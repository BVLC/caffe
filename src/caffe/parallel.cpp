#include <cstdlib>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <sstream>
#include <pthread.h>
#include <glog/logging.h>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <netdb.h>
#include <net/if.h>
#include <unistd.h>
#include <iomanip>

#include <caffe/caffe.hpp>
#include "caffe/filler.hpp"
#include "caffe/parallel.hpp"

using namespace std;

namespace caffe {

void Meter::show(std::ostream& s) const {
  ptime now = microsec_clock::local_time();
  uint64_t value = value_;
  uint64_t delta = value - last_;
  uint64_t u_sec = (now - time_).total_microseconds();
  double per_s = delta * 1e6 / (u_sec ? u_sec : 1);
  last_ = value;
  time_ = now;
  s << name_ << " " << value << " (";
  if (unit_size_)
    s << (int) (per_s * unit_size_ / (1024 * 1024)) << " mb";
  else
    s << std::setprecision(2) << per_s;
  s << "/s)";
}

//

template<typename Dtype>
static size_t len(const vector<shared_ptr<Blob<Dtype> > >& params) {
  size_t len = 0;
  for (int i = 0; i < params.size(); ++i)
    len += params[i]->count();
  return len;
}

// Align arrays to all potential chunk sizes to avoid boundary checks
template<typename Dtype>
static size_t align(const size_t len) {
  size_t m = len;
#ifndef CPU_ONLY
  m = max(m, CPUGPUSync<Dtype>::chunks(len) * CPUGPUSync<Dtype>::CHUNK);
#endif
#ifdef __linux__
  m = max(m, RawSync<Dtype>::chunks(len) * RawSync<Dtype>::CHUNK);
#endif
#ifdef INFINIBAND
  m = max(m, IBSync<Dtype>::chunks(len) * IBSync<Dtype>::CHUNK);
#endif
  return m;
}

template<typename Dtype>
Params<Dtype>::Params(const vector<shared_ptr<Blob<Dtype> > >& blobs,
                      const string& file_map)
    : len_used_(len<Dtype>(blobs)),
      len_buff_(align<Dtype>(len_used_)) {

  bool exists = false;
  if (file_map.empty()) {
    CaffeMallocHost((void**) &cpu_, len_buff_ * sizeof(Dtype));
    memset(cpu_, 0, len_buff_ * sizeof(Dtype));
  } else {
    struct stat st_buf;
    exists = stat(file_map.c_str(), &st_buf) == 0;
    int fd = open(file_map.c_str(), O_RDWR | O_CREAT,  //
                  S_IRWXU | S_IRWXG | S_IRWXO);
    CHECK(!ftruncate(fd, len_buff_ * sizeof(Dtype)));
    cpu_ = (Dtype*) mmap(NULL,  //
        len_buff_ * sizeof(Dtype),
        PROT_READ | PROT_WRITE,
        MAP_SHARED, fd, 0);
    close(fd);
  }

  Dtype* cpu = cpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->data()->size();
    // Init to current values of blobs if file doesn't already exists
    if (!exists)
      memcpy(cpu, blobs[i]->data()->cpu_data(), size);
    cpu += size / sizeof(Dtype);
    CHECK(cpu <= cpu_ + len_used_);
  }
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = cpu_ + check;
  CHECK_EQ(expect, cpu);

  iterations_ = 0;
}

template<typename Dtype>
Params<Dtype>::~Params() {
  CaffeFreeHost((void*) cpu_);
}

template<typename Dtype>
void Params<Dtype>::configure(Solver<Dtype>* solver) const {
  // Replace weights
  vector<shared_ptr<Blob<Dtype> > > &blobs = solver->net()->params();
  Dtype* cpu = cpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->data()->set_cpu_data(cpu);
    cpu += blobs[i]->data()->size() / sizeof(Dtype);
    CHECK(cpu <= cpu_ + len_used_);
  }
  // Check sizes
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = cpu_ + check;
  CHECK_EQ(expect, cpu);

  solver->iter_total(&iterations_);
}

//

#ifndef CPU_ONLY
#include <cuda_runtime.h>

template<typename Dtype>
GPUParams<Dtype>::GPUParams(const Params<Dtype>& params, int device)
    : params_(params),
      device_(device) {

  int current;
  CUDA_CHECK(cudaGetDevice(&current));
  CUDA_CHECK(cudaSetDevice(device));
  const size_t size = params.len_buff() * sizeof(Dtype);
  CUDA_CHECK(cudaMalloc((void** ) &gpu_, size));
  CUDA_CHECK(cudaMemcpy(gpu_, params.cpu(), size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaSetDevice(current));
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
  CUDA_CHECK(cudaFree((void* ) gpu_));
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  // Replace GPU weights
  vector<shared_ptr<Blob<Dtype> > > &blobs = solver->net()->params();
  Dtype* gpu = gpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->data()->set_gpu_data(gpu);
    gpu += blobs[i]->data()->size() / sizeof(Dtype);
    CHECK(gpu <= gpu_ + params_.len_used());
  }
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = gpu_ + check;
  CHECK_EQ(expect, gpu);

  solver->iter_total(&params_.iterations_);
}

//

template<typename Dtype>
GPUStream<Dtype>::GPUStream() {
  int least, greatest;
  cudaDeviceGetStreamPriorityRange(&least, &greatest);
  cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, least);
}

template<typename Dtype>
GPUStream<Dtype>::~GPUStream() {
  cudaStreamDestroy(stream_);
}

//

template<typename Dtype>
GPUSync<Dtype>::GPUSync(const GPUParams<Dtype>& params)
    : params_(params) {

  size_t size = params.params().len_buff() * sizeof(Dtype);
  Dtype* gpu = params.gpu();
  CUDA_CHECK(cudaMalloc((void** ) &gpu_last_, size));
  CUDA_CHECK(cudaMemcpy(gpu_last_, gpu, size, cudaMemcpyDeviceToDevice));
}

template<typename Dtype>
GPUSync<Dtype>::~GPUSync() {
  CUDA_CHECK(cudaFree((void* ) gpu_last_));
}

//

template<typename Dtype>
CPUGPUSync<Dtype>::CPUGPUSync(const GPUParams<Dtype>& params)
    : GPUSync<Dtype>(params),
      chunks_(chunks(params.params().len_used())),
      calls_("calls", CHUNK * sizeof(Dtype)),
      cycles_("cycles") {
}

template<typename Dtype>
CPUGPUSync<Dtype>::~CPUGPUSync() {
  stop();
}

template<typename Dtype>
void CPUGPUSync<Dtype>::run() {
  CUDA_CHECK(cudaSetDevice(this->params_.device()));
  GPUStream<Dtype> gpu_stream;
  const cudaStream_t& stream = gpu_stream.stream();

  // Current cpu values when invoking kernel, gradients on the way back
  Dtype* buf;
  Dtype* tmp;
  CUDA_CHECK(cudaMalloc((void** ) &buf, CHUNK * sizeof(Dtype)));
  CaffeMallocHost((void**) &tmp, CHUNK * sizeof(Dtype));

  const size_t len = CHUNK * sizeof(Dtype);
  // Explicit directions for readability
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  const cudaMemcpyKind get = cudaMemcpyDeviceToHost;
  uint32_t index = 0;
  Dtype* cpu = this->params_.params().cpu();
  Dtype* gpu = this->params_.gpu();
  Dtype* last = this->gpu_last_;
  uint8_t get_grads = true;

  while (!must_stop()) {
    size_t off = index * CHUNK;
    CUDA_CHECK(cudaMemcpyAsync(buf, &cpu[off], len, put, stream));
    // TODO simpler kernel
    sync_worker_kernel<Dtype>(gpu, last, &buf, &off, &buf, &get_grads, //
                              0, 1, stream, CHUNK);
    CUDA_CHECK(cudaMemcpyAsync(tmp, buf, len, get, stream));
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < CHUNK; ++i)
      cpu[off + i] += tmp[i];
    if (++index == chunks_) {
      index = 0;
      cycles_++;
    }
    calls_++;
  }

  CaffeFreeHost((void*) tmp);
  CUDA_CHECK(cudaFree((void* ) buf));
}

#endif

//

template<typename Dtype>
DistSync<Dtype>::DistSync(uint32_t nodes, uint32_t chunks)
    : nodes_(nodes),
      chunks_(chunks),
      received_(chunks),
      remaining_(chunks),
      cycles_("cycles") {

  own_start_ = own_until_ = chunk_ = 0;
}

template<typename Dtype>
void DistSync<Dtype>::dist_init(int rank) {
  own_start_ = (rank + 0) * chunks_ / nodes_;
  own_until_ = (rank + 1) * chunks_ / nodes_;
  LOG(INFO)<< "range: " << own_start_ << " " << own_until_;
  chunk_ = own_start_;

  for (uint32_t chunk = own_start_; chunk < own_until_; ++chunk) {
    received_[chunk] = true;
    remaining_--;
  }
}

template<typename Dtype>
inline int DistSync<Dtype>::chunk_master(uint32_t chunk) {
  // TODO find range without loop?
  for (int i = nodes_ - 1; i >= 0; --i) {
    uint32_t start = i * chunks_ / nodes_;
    if (start <= chunk)
      return i;
  }
  CHECK(false);
  return -1;
}

//

INSTANTIATE_CLASS(Params);
#ifndef CPU_ONLY
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(GPUSync);
INSTANTIATE_CLASS(CPUGPUSync);
#endif
INSTANTIATE_CLASS(DistSync);

#ifdef RDMA

ibv_context* IBChannel::open_device(ibv_device* ib_dev) {
  ibv_context* context = ibv_open_device(ib_dev);
  CHECK(context) << "Open context failed for " << ibv_get_device_name(ib_dev);
  return context;
}

ibv_pd* IBChannel::alloc_pd(ibv_context* context) {
  ibv_pd* pd = ibv_alloc_pd(context);
  CHECK(pd) << "Failed to allocate protection domain";
  return pd;
}

IBChannel::IBChannel(ibv_device* ib_dev)
    : context_(open_device(ib_dev)),
      pd_(alloc_pd(context_)),
      buf_send_(),
      buf_recv_(),
      mr_send_(),
      mr_recv_(),
      send_queue_(FRAMES),
      recv_queue_(FRAMES),
      sent_("sent", MTU),
      recv_("recv", MTU) {

  cq_ = ibv_create_cq(context_, FRAMES * 2, NULL, NULL, 0);
  CHECK(cq_) << "Failed to create completion queue";

  // Create queue pair
  {
    ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.send_cq = cq_;
    attr.recv_cq = cq_;
    attr.cap.max_send_wr = FRAMES;
    attr.cap.max_recv_wr = FRAMES;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_UD,

    qp_ = ibv_create_qp(pd_, &attr);
    CHECK(qp_) << "Failed to create queue pair";
  }

  // Init queue pair
  {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = PORT;
    attr.qkey = 0x11111111;

    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
    CHECK(!ibv_modify_qp(qp_, &attr, mask)) << "Failed to set QP to INIT";
  }

  // Local address
  {
    memset(&local_, 0, sizeof(local_));
    ibv_port_attr attr;
    CHECK(!ibv_query_port(context_, PORT, &attr)) << "Query port";
    CHECK(attr.active_mtu == IBV_MTU_4096);
    local_.lid = attr.lid;
    local_.qpn = qp_->qp_num;
    local_.psn = caffe_rng_rand() & 0xffffff;
  }

  // Queue pair to recv & send
  {
    struct ibv_qp_attr attr;
    attr.qp_state = IBV_QPS_RTR;  // Ready to receive
    CHECK(!ibv_modify_qp(qp_, &attr, IBV_QP_STATE)) << "QP to RTR";
    attr.qp_state = IBV_QPS_RTS;  // Ready to send
    attr.sq_psn = local_.psn;
    int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    CHECK(!ibv_modify_qp(qp_, &attr, mask)) << "QP to RTS";
  }

  for (int i = 0; i < 2 * FRAMES; ++i)
    wc_[i].wr_id = i;
}

static ib_addr mcast_init(ibv_context* context, int port, const ibv_gid* mgid) {
  mcast_parameters params;
  memset(&params, 0, sizeof(struct mcast_parameters));

  string ib_devname(ibv_get_device_name(context->device));
  params.ib_devname = const_cast<char*>(ib_devname.c_str());
  CHECK(!ibv_query_gid(context, port, 0, &params.port_gid));
  CHECK(!ibv_query_pkey(context, port, DEF_PKEY_IDX, &params.pkey));

  ibv_port_attr port_attr;
  CHECK(!ibv_query_port(context, port, &port_attr));
  params.sm_lid = port_attr.sm_lid;
  params.sm_sl = port_attr.sm_sl;
  params.ib_port = port;

  if (mgid)
    memcpy(&params.mgid.raw, &mgid->raw, 16);

  CHECK(!join_multicast_group(SUBN_ADM_METHOD_SET, &params))
      << "Failed to create multicast group";

  ib_addr addr;
  memcpy(&addr.gid.raw, &params.mgid.raw, 16);
  addr.lid = params.mlid;
  addr.qpn = QPNUM_MCAST;
  return addr;
}

ib_addr IBChannel::mcast_create() const {
  ib_addr addr = mcast_init(context_, PORT, NULL);
  addr.psn = caffe_rng_rand() & 0xffffff;
  return addr;
}

void IBChannel::mcast_join(const ib_addr& addr) const {
  mcast_init(context_, PORT, &addr.gid);
}

void IBChannel::mcast_attach_qp(const ib_addr& addr) const {
  CHECK(!ibv_attach_mcast(qp_, &addr.gid, addr.lid))
      << "Failed to attach to the multicast group";
}

void IBChannel::start(uint8_t* buf_send, size_t buf_size, bool gpu) const {
  size_t send_size = buf_send ? buf_size : FRAMES * MTU;
  size_t recv_size = FRAMES * (GRH + MTU);

  if (gpu) {
    if (buf_send) {
      buf_send_ = buf_send;
    } else {
      CUDA_CHECK(cudaMalloc((void** ) &buf_send_, send_size));
    }
    CUDA_CHECK(cudaMalloc((void** ) &buf_recv_, recv_size));
  } else {
    buf_send_ = buf_send ? buf_send : (uint8_t*) malloc(send_size);
    buf_recv_ = (uint8_t*) malloc(recv_size);
  }

  LOG(INFO)<< "range: " << hex << (uint64_t) buf_send_ << " " << (uint64_t) send_size;
  LOG(INFO)<< "range: " << hex << (uint64_t) buf_recv_ << " " << (uint64_t) recv_size;

  mr_send_ = ibv_reg_mr(pd_, buf_send_, send_size, IBV_ACCESS_LOCAL_WRITE);
  mr_recv_ = ibv_reg_mr(pd_, buf_recv_, recv_size, IBV_ACCESS_LOCAL_WRITE);
  CHECK(mr_send_ && mr_recv_) << "Failed to register memory regions";

  // Create initial requests, start the recv ones
  for (int i = 0; i < FRAMES; ++i) {
    send_queue_[i] = i;
    recv_done(i + FRAMES);
  }
  recv_queue_.clear();
}

IBChannel::~IBChannel() {
  CHECK(!ibv_destroy_qp(qp_)) << "Failed to destroy QP";
  CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";
  CHECK(!ibv_dereg_mr(mr_send_)) << "Failed to deregister MR";
  CHECK(!ibv_dereg_mr(mr_recv_)) << "Failed to deregister MR";
  CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD";
  CHECK(!ibv_close_device(context_)) << "Failed to release context";
  free(buf_send_);
  free(buf_recv_);
}

bool IBChannel::can_send() const {
  return !send_queue_.empty();
}

int IBChannel::send_init(uint8_t*& buf) const {
  int id = send_queue_.front();
  send_queue_.pop_front();
  buf = buf_send_ + id * MTU;
  return id;
}

void IBChannel::send(int id, const ib_addr& addr, uint8_t* buf,
                     uint32_t imm_data) const {
  struct ibv_sge list;
  struct ibv_send_wr wr;
  struct ibv_send_wr *bad_wr;

  list.addr = (uintptr_t) buf;
  list.length = MTU;
  list.lkey = mr_send_->lkey;

  wr.wr_id = id;
  wr.next = NULL;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.ud.ah = addr.ah;
  wr.wr.ud.remote_qpn = addr.qpn;
  wr.wr.ud.remote_qkey = 0x11111111;

  CHECK(!ibv_post_send(qp_, &wr, &bad_wr)) << "Failed send";
}

bool IBChannel::can_recv() const {
  return !recv_queue_.empty();
}

int IBChannel::recv(uint8_t*& buf, uint32_t& imm_data) const {
  recv_msg& msg = recv_queue_.front();
  int id = msg.id_;
  buf = buf_recv_ + (id - FRAMES) * (GRH + MTU) + GRH;
  imm_data = msg.imm_;
  recv_queue_.pop_front();
  return id;
}

void IBChannel::recv_done(int id) const {
  struct ibv_sge list;
  struct ibv_recv_wr wr;
  struct ibv_recv_wr* bad_wr;

  list.addr = (uintptr_t) (buf_recv_ + (id - FRAMES) * (GRH + MTU));
  list.length = GRH + MTU;
  list.lkey = mr_recv_->lkey;

  wr.wr_id = id;
  wr.next = NULL;
  wr.sg_list = &list;
  wr.num_sge = 1;

  CHECK(!ibv_post_recv(qp_, &wr, &bad_wr)) << "Failed receive";
}

void IBChannel::poll() const {
  int ne = ibv_poll_cq(cq_, FRAMES * 2, wc_);
  CHECK(ne >= 0) << "Poll CQ failed";

  for (int i = 0; i < ne; ++i) {
    CHECK(wc_[i].status == IBV_WC_SUCCESS) << "Failed status \n"
                                           << ibv_wc_status_str(wc_[i].status)
                                           << " " << wc_[i].status << " "
                                           << (int) wc_[i].wr_id << " "
                                           << wc_[i].vendor_err;

    if (wc_[i].wr_id < IBChannel::FRAMES) {
      sent_++;
      send_queue_.push_back(wc_[i].wr_id);
    } else {
      recv_++;
      CHECK(wc_[i].byte_len == GRH + MTU);
      recv_msg msg;
      msg.id_ = wc_[i].wr_id;
      msg.imm_ = wc_[i].imm_data;
      recv_queue_.push_back(msg);
    }
  }
}

//

template<typename Dtype>
IBSync<Dtype>::IBSync(const Params<Dtype>& params, int rank,
                      const IBChannel& ucast, const IBChannel& mcast,
                      const vector<ib_addr>& ucast_addrs,
                      const vector<ib_addr>& mcast_addrs)
    : DistSync<Dtype>(ucast_addrs.size(), chunks(params.len_used())),
      rank_(rank),
      ucast_(ucast),
      mcast_(mcast),
      ucast_addrs_(ucast_addrs),
      mcast_addr_(mcast_addrs[rank]) {

  for (int i = 0; i < ucast_addrs_.size(); ++i) {
    CHECK(ucast_addrs_[i].ah == NULL);
    if (i != rank) {
      struct ibv_ah_attr ah_attr;
      memset(&ah_attr, 0, sizeof ah_attr);
      ah_attr.dlid = (uint16_t) ucast_addrs[i].lid;
      ah_attr.sl = (uint8_t) 0;  // Service level
      ah_attr.src_path_bits = 0;
      ah_attr.is_global = 0;
      ah_attr.port_num = IBChannel::PORT;
      ucast_addrs_[i].ah = ibv_create_ah(ucast.pd_, &ah_attr);
      CHECK(ucast_addrs_[i].ah) << "Failed to create address handle";
    }
  }

  struct ibv_ah_attr ah_attr;
  memset(&ah_attr, 0, sizeof ah_attr);
  ah_attr.grh.dgid = mcast_addr_.gid;
  ah_attr.dlid = (uint16_t) mcast_addr_.lid;
  ah_attr.sl = (uint8_t) 0;  // Service level
  ah_attr.src_path_bits = 0;
  ah_attr.is_global = 1;
  ah_attr.port_num = IBChannel::PORT;
  mcast_addr_.ah = ibv_create_ah(mcast.pd_, &ah_attr);
  CHECK(mcast_addr_.ah) << "Failed to create address handle";

  for (int i = 0; i < mcast_addrs.size(); ++i) {
    if (i != rank) {
      mcast_.mcast_join(mcast_addrs[i]);
      mcast_.mcast_attach_qp(mcast_addrs[i]);
    }
  }

  this->dist_init(rank);
}

template<typename Dtype>
IBSync<Dtype>::~IBSync() {
  for (int i = 0; i < this->ucast_addrs_.size(); ++i) {
    if (i == rank_) {
      CHECK(!ibv_destroy_ah(this->ucast_addrs_[i].ah))
          << "Failed to destroy ucast AH";
    }
  }
  CHECK(!ibv_destroy_ah(this->mcast_addr_.ah)) << "Failed to destroy mcast AH";
}

//

template<typename Dtype>
CPUIBSync<Dtype>::CPUIBSync(const Params<Dtype>& params, int rank,
                            const IBChannel& ucast, const IBChannel& mcast,
                            const vector<ib_addr>& ucast_addrs,
                            const vector<ib_addr>& mcast_addrs)
    : IBSync<Dtype>(params, rank, ucast, mcast, ucast_addrs, mcast_addrs) {

  cpu_ = params.cpu();
  CaffeMallocHost((void**) &cpu_last_, params.len_buff() * sizeof(Dtype));
  memcpy(cpu_last_, cpu_, params.len_used() * sizeof(Dtype));
}

template<typename Dtype>
CPUIBSync<Dtype>::~CPUIBSync() {
  CaffeFreeHost((void*) cpu_last_);
}

template<typename Dtype>
void CPUIBSync<Dtype>::run() {
  // TODO
}

//

template<typename Dtype>
GPUIBSync<Dtype>::GPUIBSync(const GPUParams<Dtype>& params, int rank,
                            const IBChannel& ucast, const IBChannel& mcast,
                            const vector<ib_addr>& ucast_addrs,
                            const vector<ib_addr>& mcast_addrs)
    : GPUSync<Dtype>(params),
      IBSync<Dtype>(params.params(), rank,  //
                    ucast, mcast,  //
                    ucast_addrs, mcast_addrs) {

  gpu_ = params.gpu();
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaSetDevice(params.device()));
  size_t size = params.params().len_buff() * sizeof(Dtype);
  CUDA_CHECK(cudaMalloc((void** ) &gpu_last_, size));
  CUDA_CHECK(cudaMemcpy(gpu_last_, gpu_, size, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaSetDevice(device));
}

template<typename Dtype>
GPUIBSync<Dtype>::~GPUIBSync() {
  CUDA_CHECK(cudaFree((void* ) gpu_last_));
}

class Queue {
 public:
  Queue()
      : front_(),
        back_(),
        size_() {
  }
  void push() {
    CHECK(size_ < IBChannel::FRAMES);
    back_ = (back_ + 1) & (IBChannel::FRAMES - 1);
    size_++;
  }
  void pop() {
    CHECK(size_ > 0);
    front_ = (front_ + 1) & (IBChannel::FRAMES - 1);
    size_--;
  }

  int front_;
  int back_;
  int size_;
};

class EventQueue : Queue {
 public:
  EventQueue(const cudaStream_t& stream)
      : stream_(stream) {

    for (int i = 0; i < IBChannel::FRAMES; ++i)
      cudaEventCreateWithFlags(&items_[i].event_, cudaEventDisableTiming);
  }

  ~EventQueue() {
    for (int i = 0; i < IBChannel::FRAMES; ++i)
      cudaEventDestroy(items_[i].event_);
  }

  void record(int tag) {
    cudaEventRecord(items_[back_].event_, this->stream_);
    items_[back_].tag_ = tag;
    push();
  }

  bool query(int& tag) {
    if (size_ && cudaEventQuery(items_[front_].event_) == cudaSuccess) {
      tag = items_[front_].tag_;
      pop();
      return true;
    }
    return false;
  }

 protected:
  const cudaStream_t& stream_;
  struct item {
    cudaEvent_t event_;
    int tag_;
  };
  item items_[IBChannel::FRAMES];
};

template<typename Dtype>
void GPUIBSync<Dtype>::run() {
  CUDA_CHECK(cudaSetDevice(this->params_.device()));

  const IBChannel& ucast = this->ucast_;
  const IBChannel& mcast = this->mcast_;
  ucast.start(NULL, 0, true);
  mcast.start((uint8_t*) gpu_, (size_t) this->chunks_ * IBChannel::MTU, true);

  GPUStream<Dtype> master_stream;
  Queue master_queue;
  uint16_t master_ids_[FRAMES];
  EventQueue master_events(master_stream.stream());

  GPUStream<Dtype> worker_stream;
  Queue worker_queue;
  struct worker_item {
    int recv_id, send_id;
    Dtype* grd;
    uint32_t chunk;
  };
  worker_item worker_items[FRAMES];
  EventQueue worker_events(worker_stream.stream());

  const size_t real_size = FRAMES * sizeof(Dtype*);
  const size_t size_size = FRAMES * sizeof(size_t);
  const size_t bool_size = FRAMES * sizeof(size_t);

  Dtype** master_gpu_grds;
  size_t* master_gpu_offs;
  CUDA_CHECK(cudaMalloc((void** ) &master_gpu_grds, real_size));
  CUDA_CHECK(cudaMalloc((void** ) &master_gpu_offs, size_size));
  Dtype** master_cpu_grds;
  size_t* master_cpu_offs;
  CUDA_CHECK(cudaMallocHost((void** ) &master_cpu_grds, real_size));
  CUDA_CHECK(cudaMallocHost((void** ) &master_cpu_offs, size_size));

  Dtype** worker_gpu_pos;
  size_t* worker_gpu_offs;
  Dtype** worker_gpu_grds;
  uint8_t* worker_gpu_gets;
  CUDA_CHECK(cudaMalloc((void** ) &worker_gpu_pos, real_size));
  CUDA_CHECK(cudaMalloc((void** ) &worker_gpu_offs, size_size));
  CUDA_CHECK(cudaMalloc((void** ) &worker_gpu_grds, real_size));
  CUDA_CHECK(cudaMalloc((void** ) &worker_gpu_gets, bool_size));
  Dtype** worker_cpu_pos;
  size_t* worker_cpu_offs;
  Dtype** worker_cpu_grds;
  uint8_t* worker_cpu_gets;
  CUDA_CHECK(cudaMallocHost((void** ) &worker_cpu_pos, real_size));
  CUDA_CHECK(cudaMallocHost((void** ) &worker_cpu_offs, size_size));
  CUDA_CHECK(cudaMallocHost((void** ) &worker_cpu_grds, real_size));
  CUDA_CHECK(cudaMallocHost((void** ) &worker_cpu_gets, bool_size));

  int master_batch_start = 0;
  int master_batch_count = 0;
  int worker_batch_start = 0;
  int worker_batch_count = 0;
  const int batch = 128;  // TODO bench
  while (!this->must_stop()) {
    ucast.poll();
    mcast.poll();

    // Receive gradients for chunks for which we are master
    {
      while (ucast.can_recv()) {
        uint8_t* buf;
        uint32_t chunk;
        int id = ucast.recv(buf, chunk);
        Dtype* grd = (Dtype*) buf;
        CHECK(this->chunk_master(chunk) == this->rank_);
        size_t off = ((size_t) chunk) * IBSync<Dtype>::CHUNK;

        int index = master_queue.back_;
        master_ids_[index] = id;
        master_cpu_grds[index] = grd;
        master_cpu_offs[index] = off;
        master_queue.push();
        master_batch_count++;
      }
      // Add gradients to our weights
      if (master_batch_count >= batch) {
        CUDA_CHECK(
            cudaMemcpyAsync(master_gpu_grds, master_cpu_grds, real_size,
                            cudaMemcpyHostToDevice, master_stream.stream()));
        CUDA_CHECK(
            cudaMemcpyAsync(master_gpu_offs, master_cpu_offs, size_size,
                            cudaMemcpyHostToDevice, master_stream.stream()));
        sync_master_kernel<Dtype>(gpu_, master_gpu_grds, master_gpu_offs,
                                  master_batch_start, master_batch_count,
                                  master_stream.stream(), IBSync<Dtype>::CHUNK);
        master_events.record(master_batch_count);
        master_batch_start = master_queue.back_;
        master_batch_count = 0;
      }
    }
    // Start receiving again once kernels are done with buffers
    for (;;) {
      int batch;
      if (!master_events.query(batch)) {
        break;
      }
      for (int i = 0; i < batch; ++i) {
        int index = master_queue.front_;
        master_queue.pop();
        ucast.recv_done(master_ids_[index]);
      }
    }
    // Send absolute positions for chunks for which we are master
    while (mcast.can_send()) {
      uint8_t* buf;
      int id = mcast.send_init(buf);  // buf ignored
      size_t off = (size_t) this->chunk_ * IBSync<Dtype>::CHUNK;
      buf = (uint8_t*) (gpu_ + off);
      CHECK(id >= 0 && id < FRAMES);
      mcast.send(id, this->mcast_addr_, buf, this->chunk_);
      if (++this->chunk_ == this->own_until_) {
        this->chunk_ = this->own_start_;
        this->cycles_++;
      }
    }

    // Receive absolute positions for other chunks
    {
      while (mcast.can_recv()) {
        Dtype* pos;
        uint32_t chunk;
        int recv_id, send_id;
        size_t off;
        {
          uint8_t* buf;
          recv_id = mcast.recv(buf, chunk);
          pos = (Dtype*) buf;
          off = ((size_t) chunk) * IBSync<Dtype>::CHUNK;
        }

        // Send back the gradients if frame is available
        Dtype* grd = NULL;
        if (ucast.can_send()) {
          uint8_t* buf;
          send_id = ucast.send_init(buf);
          grd = (Dtype*) buf;
        }

        int index = worker_queue.back_;
        worker_items[index].recv_id = recv_id;
        worker_items[index].send_id = send_id;
        worker_items[index].grd = grd;
        worker_items[index].chunk = chunk;
        worker_cpu_pos[index] = pos;
        worker_cpu_offs[index] = off;
        worker_cpu_grds[index] = grd;
        worker_cpu_gets[index] = grd != NULL;
        worker_queue.push();
        worker_batch_count++;
      }
      if (worker_batch_count >= batch) {
        CUDA_CHECK(
            cudaMemcpyAsync(worker_gpu_pos, worker_cpu_pos, real_size,
                            cudaMemcpyHostToDevice, worker_stream.stream()));
        CUDA_CHECK(
            cudaMemcpyAsync(worker_gpu_offs, worker_cpu_offs, size_size,
                            cudaMemcpyHostToDevice, worker_stream.stream()));
        CUDA_CHECK(
            cudaMemcpyAsync(worker_gpu_grds, worker_cpu_grds, real_size,
                            cudaMemcpyHostToDevice, worker_stream.stream()));
        CUDA_CHECK(
            cudaMemcpyAsync(worker_gpu_gets, worker_cpu_gets, bool_size,
                            cudaMemcpyHostToDevice, worker_stream.stream()));
        sync_worker_kernel<Dtype>(gpu_, gpu_last_, worker_gpu_pos,
                                  worker_gpu_offs, worker_gpu_grds,
                                  worker_gpu_gets, worker_batch_start,
                                  worker_batch_count, worker_stream.stream(),
                                  IBSync<Dtype>::CHUNK);
        worker_events.record(worker_batch_count);
        worker_batch_start = worker_queue.back_;
        worker_batch_count = 0;
      }
    }
    for (;;) {
      int batch;
      if (!worker_events.query(batch)) {
        break;
      }
      for (int i = 0; i < batch; ++i) {
        int index = worker_queue.front_;
        worker_queue.pop();
        int recv_id = worker_items[index].recv_id;
        int send_id = worker_items[index].send_id;
        Dtype* grd = worker_items[index].grd;
        uint32_t chunk = worker_items[index].chunk;

        mcast.recv_done(recv_id);
        if (grd) {
          int master = this->chunk_master(chunk);
          CHECK(master != this->rank_);
          ib_addr& a = this->ucast_addrs_[master];
          ucast.send(send_id, a, (uint8_t*) grd, chunk);
        }
        if (this->remaining_ > 0 && !this->received_[chunk]) {
          this->received_[chunk] = true;
          this->remaining_--;
        }
      }
    }
  }
}

INSTANTIATE_CLASS(IBSync);
INSTANTIATE_CLASS(CPUIBSync);
INSTANTIATE_CLASS(GPUIBSync);

#endif

#ifdef __linux__

// Parse MAC address to byte array
// TODO remove optional ':' chars
static uint8_t* parse_mac(const char* str) {
  uint8_t* bytes = (uint8_t*) malloc(ETH_ALEN);
  for (int i = 0; i < ETH_ALEN; ++i) {
    int value;
    sscanf(str + 2 * i, "%02x", &value);
    bytes[i] = value;
  }
  return bytes;
}

static vector<uint8_t*> parse_macs(const vector<string>& macs) {
  vector<uint8_t*> res;
  for (int i = 0; i < macs.size(); ++i)
    res.push_back(parse_mac(macs[i].c_str()));
  return res;
}

// Adapter name from MAC address
static string adapter(const uint8_t* mac) {
  int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  CHECK(s != -1);

  // Iterate over adapters
  struct ifconf ifc;
  char buf[1024];
  ifc.ifc_len = sizeof(buf);
  ifc.ifc_buf = buf;
  CHECK(ioctl(s, SIOCGIFCONF, &ifc) != -1);
  struct ifreq* it = ifc.ifc_req;
  const struct ifreq* const end = it + (ifc.ifc_len / sizeof(struct ifreq));

  // Look for a MAC match
  struct ifreq ifr;
  for (; it != end; ++it) {
    strcpy(ifr.ifr_name, it->ifr_name);
    CHECK(!ioctl(s, SIOCGIFHWADDR, &ifr));
    if (!memcmp(mac, ifr.ifr_hwaddr.sa_data, ETH_ALEN))
      return string(it->ifr_name);
  }
  return "";
}

static int local(const vector<uint8_t*>& macs) {
  for (int i = 0; i < macs.size(); ++i) {
    string a = adapter(macs[i]);
    if (!a.empty())
      return i;
  }
  CHECK(0) << "Local machine not part of given MAC addresses.";
  return -1;
}

//

Ring::Ring(const string& adapter, int protocol_send, int protocol_recv)
    : adapter_(adapter),  //
      socket_(socket(PF_PACKET, SOCK_RAW, htons(protocol_recv))),  //
      sent_("sent", ETH_FRAME_LEN),
      recv_("recv", ETH_FRAME_LEN) {

  const int s = socket_;
  CHECK(s != -1) << "Cannot open raw socket, make sure to run as root or to "
                 << "set the capability on the executable: "
                 << "sudo setcap cap_net_raw+ep <prog.bin>" << endl;

  // TODO look at this
  //  s_ifr.ifr_mtu = c_mtu;
  //  /* update the mtu through ioctl */
  //  ec = ioctl(fd_socket, SIOCSIFMTU, &s_ifr);
  //  if(ec == -1)
  //  {
  //    perror("iotcl");
  //  return EXIT_FAILURE;
  //  }

  // Get adapter info
  struct ifreq ifr;
  strcpy(ifr.ifr_name, adapter.c_str());
  CHECK(ioctl(s, SIOCGIFINDEX, &ifr) != -1);
  int index = ifr.ifr_ifindex;
  CHECK(ioctl(s, SIOCGIFHWADDR, &ifr) != -1);
  uint8_t* mac = (uint8_t*) ifr.ifr_hwaddr.sa_data;

  // Bind to interface
  struct sockaddr_ll addr;
  memset(&addr, 0, sizeof(struct sockaddr_ll));
  addr.sll_family = AF_PACKET;
  addr.sll_protocol = htons(protocol_recv);
  addr.sll_ifindex = index;
  CHECK(bind(s, (struct sockaddr* ) &addr, sizeof(struct sockaddr_ll)) != -1);

  // Setup ring buffer
  struct tpacket_req req;
  req.tp_frame_size = FRAME_SIZE;
  req.tp_frame_nr = FRAME_NR;
  req.tp_block_size = FRAME_SIZE * FRAME_NR;
  req.tp_block_nr = BLOCK_NR;
  CHECK(setsockopt(s, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req)) >= 0);
  CHECK(setsockopt(s, SOL_PACKET, PACKET_TX_RING, &req, sizeof(req)) >= 0);
  uint32_t size = req.tp_block_size * req.tp_block_nr;
  int prot = PROT_READ | PROT_WRITE;
  map_recv_ = (uint8_t*) mmap(0, 2 * size, prot, MAP_SHARED, s, 0);
  map_send_ = map_recv_ + size;
  CHECK(map_recv_ != (void* ) -1);

  // Pre-fill send frames with sender address and protocol
  const __be16 protocol = htons(protocol_send);
  for (int i = 0; i < FRAME_NR; i++) {
    struct tpacket_hdr* hdr;
    hdr = (struct tpacket_hdr*) (map_send_ + FRAME_SIZE * i);
    hdr->tp_len = ETH_FRAME_LEN;
    uint8_t* eth = ((uint8_t*) hdr) + TPACKET_ALIGN(sizeof(struct tpacket_hdr));
    memcpy(eth + ETH_ALEN, mac, ETH_ALEN);
    memcpy(eth + ETH_ALEN * 2, &protocol, 2);
  }
}

Ring::~Ring() {
  shutdown(socket_, 2);
}

bool Ring::can_send(int frame, struct tpacket_hdr*& hdr) {
  hdr = (struct tpacket_hdr*) (map_send_ + FRAME_SIZE * frame);
  int status = (volatile uint32_t) hdr->tp_status;
  CHECK(!(status & TP_STATUS_WRONG_FORMAT));
  return status == TP_STATUS_AVAILABLE;
}

ethhdr* Ring::send_init(const struct tpacket_hdr* hdr) {
  uint8_t* eth = ((uint8_t*) hdr) + TPACKET_ALIGN(sizeof(struct tpacket_hdr));
  return (struct ethhdr*) eth;
}

void Ring::send(struct tpacket_hdr* hdr) {
  hdr->tp_status = TP_STATUS_SEND_REQUEST;
  sent_++;
}

bool Ring::can_recv(int frame, struct tpacket_hdr*& hdr) {
  hdr = (struct tpacket_hdr*) (map_recv_ + FRAME_SIZE * frame);
  int status = (volatile uint32_t) hdr->tp_status;
  CHECK(!(status & TP_STATUS_COPY));
  return status & TP_STATUS_USER;
}

ethhdr* Ring::recv(const struct tpacket_hdr* hdr) {
  return (struct ethhdr*) ((uint8_t*) hdr + hdr->tp_mac);
}

void Ring::recv_done(struct tpacket_hdr* hdr) {
  hdr->tp_status = TP_STATUS_KERNEL;
  recv_++;
}

void Ring::socket_stats(uint64_t& received, uint64_t& dropped) {
  struct tpacket_stats st;
  unsigned int len = sizeof(st);
  int s = socket_;
  CHECK(!getsockopt(s, SOL_PACKET, PACKET_STATISTICS, (char* ) &st, &len));
  received = st.tp_packets;
  dropped = st.tp_drops;
}

//

template<typename Dtype>
RawSync<Dtype>::RawSync(const Params<Dtype>& params,
                        const vector<string>& mac_addresses,
                        const vector<string>& secondary_macs)
    : DistSync<Dtype>(mac_addresses.size(), chunks(params.len_used())),
      masters_(parse_macs(mac_addresses)),
      workers_(
          secondary_macs.size() ?
              parse_macs(secondary_macs) : parse_macs(mac_addresses)),
      others_(),
      master_(adapter(this->masters_[local(this->masters_)]), 0x73A, 0x73B),
      worker_(adapter(this->workers_[local(this->workers_)]), 0x73B, 0x73A) {

  int rank = local(this->masters_);
  ostringstream s;
  s << "Raw socket - node: " << rank << ", ";
  if (secondary_macs.size()) {
    CHECK(master_.adapter() != worker_.adapter());
    CHECK(rank == local(this->workers_));
    s << "adapters: " << master_.adapter() << ", " << worker_.adapter() << endl;
  } else {
    CHECK(master_.adapter() == worker_.adapter());
    s << "adapter: " << master_.adapter() << endl;
  }
  LOG(INFO)<< s.str();

  cpu_ = params.cpu();
  CaffeMallocHost((void**) &cpu_last_, params.len_buff() * sizeof(Dtype));
  memcpy(cpu_last_, cpu_, params.len_used() * sizeof(Dtype));

  for (int i = 0; i < workers_.size(); ++i)
    if (i != rank)
      others_.push_back(workers_[i]);

  this->dist_init(rank);
}

template<typename Dtype>
RawSync<Dtype>::~RawSync() {
  CaffeFreeHost((void*) cpu_last_);
}

template<typename Dtype>
inline void RawSync<Dtype>::next() {
  if (++other_ == others_.size()) {
    other_ = 0;
    if (++this->chunk_ == this->own_until_) {
      this->chunk_ = this->own_start_;
      this->cycles_++;
    }
  }
}

template<typename Dtype>
void RawSync<Dtype>::run() {
  struct tpacket_hdr* hdr;
  struct tpacket_hdr* hdr_send;
  // TODO split over two threads? compact wire format?
  for (;;) {
    // Receive and add gradients for chunks for which we are master
    for (int f = 0; f < Ring::FRAME_NR; f++) {
      if (master_.can_recv(f, hdr)) {
        ethhdr* eth = master_.recv(hdr);
        uint8_t* data = (uint8_t*) eth + ETH_HLEN;
        uint32_t chunk = ((uint32_t*) &(data[MSG_CHUNK]))[0];
        size_t off = ((size_t) chunk) * CHUNK;
        Dtype* grads = (Dtype*) &(data[MSG_DATA]);
        for (size_t i = 0; i < CHUNK; ++i)
          this->cpu_[off + i] += grads[i];
        master_.recv_done(hdr);
      }
    }

    // Send absolute positions for chunks for which we are master
    // TODO allow broadcast addresses on private networks instead of
    // iterating over workers
    for (int f = 0; f < Ring::FRAME_NR; f++) {
      if (master_.can_send(f, hdr)) {
        uint32_t peer = this->other_;
        uint32_t chnk = this->chunk_;
        ethhdr* eth = master_.send_init(hdr);
        memcpy(eth->h_dest, (void*) this->others_[peer], ETH_ALEN);
        uint8_t* data = (uint8_t*) eth + ETH_HLEN;
        ((uint32_t*) &(data[MSG_CHUNK]))[0] = chnk;
        Dtype* pos = (Dtype*) &(data[MSG_DATA]);
        size_t off = (size_t) chnk * CHUNK;
        memcpy(pos, this->cpu_ + off, CHUNK * sizeof(Dtype));
        master_.send(hdr);
        this->next();
      }
    }
    send(master_.sock(), NULL, 0, MSG_DONTWAIT);

    // Receive absolute positions for other chunks
    for (int f = 0; f < Ring::FRAME_NR; f++) {
      if (worker_.can_recv(f, hdr)) {
        ethhdr* eth = worker_.recv(hdr);
        uint8_t* data = (uint8_t*) eth + ETH_HLEN;
        uint32_t chunk = ((uint32_t*) &(data[MSG_CHUNK]))[0];
        size_t off = ((size_t) chunk) * CHUNK;
        Dtype* pos = (Dtype*) &(data[MSG_DATA]);

        // Send back the gradients if frame is available
        Dtype* grads = NULL;
        if (worker_.can_send(f, hdr_send)) {
          ethhdr* eth_send = worker_.send_init(hdr_send);
          uint8_t* m = this->masters_[this->chunk_master(chunk)];
          memcpy(eth_send->h_dest, (void*) m, ETH_ALEN);
          uint8_t* data_send = (uint8_t*) eth_send + ETH_HLEN;
          ((uint32_t*) &(data_send[MSG_CHUNK]))[0] = chunk;
          grads = (Dtype*) &(data_send[MSG_DATA]);
        }

        for (size_t i = 0; i < CHUNK; ++i) {
          Dtype d = this->cpu_[off + i] - this->cpu_last_[off + i];
          // If gradient is sent, reset last_ to cpu_, otherwise keep them apart
          if (grads) {
            grads[i] = d;
            this->cpu_last_[off + i] = pos[i] + d;
            this->cpu_[off + i] = this->cpu_last_[off + i];
          } else {
            this->cpu_last_[off + i] = pos[i];
            this->cpu_[off + i] = this->cpu_last_[off + i] + d;
          }
        }

        worker_.recv_done(hdr);
        if (grads)
          worker_.send(hdr_send);

        if (this->remaining_ > 0 && !this->received_[chunk]) {
          this->received_[chunk] = true;
          this->remaining_--;
        }
      }
    }
    send(worker_.sock(), NULL, 0, MSG_DONTWAIT);
  }
}

INSTANTIATE_CLASS(RawSync);

#endif
}
