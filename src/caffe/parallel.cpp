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
#include <cuda_runtime.h>
#include <unistd.h>
#include <iomanip>

#include <caffe/caffe.hpp>
#include "caffe/filler.hpp"
#include "caffe/parallel.hpp"

using namespace std;

namespace caffe {

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
  size_t m = 0;
  m = max(m, GPUSync<Dtype>::chunks(len) * GPUSync<Dtype>::CHUNK);
  //m = max(m, UDPSync<Dtype>::chunks(len) * UDPSync<Dtype>::CHUNK);
#ifdef __linux__
  m = max(m, RawSync<Dtype>::chunks(len) * RawSync<Dtype>::CHUNK);
#endif
  return m;
}

template<typename Dtype>
Params<Dtype>::Params(const vector<shared_ptr<Blob<Dtype> > >& blobs, const string& file_map) :
    len_used_(len<Dtype>(blobs)), len_buff_(align<Dtype>(len_used_)) {

  bool exists = false;
  if (file_map.empty()) {
    CaffeMallocHost((void**) &cpu_, len_buff_ * sizeof(Dtype));
    memset(cpu_, 0, len_buff_ * sizeof(Dtype));
  } else {
    struct stat st_buf;
    exists = stat(file_map.c_str(), &st_buf) == 0;
    int fd = open(file_map.c_str(), O_RDWR | O_CREAT, //
        S_IRWXU | S_IRWXG | S_IRWXO);
    ftruncate(fd, len_buff_ * sizeof(Dtype));
    cpu_ = (Dtype*) mmap(NULL, //
        len_buff_ * sizeof(Dtype),
        PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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

//

template<typename Dtype>
CPUSync<Dtype>::CPUSync(Params<Dtype>& params, Solver<Dtype>& solver) :
    Sync<Dtype>(params) {
  // Replace weights
  vector<shared_ptr<Blob<Dtype> > > &blobs = solver.net()->params();
  Dtype* cpu = params.cpu();
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->data()->set_cpu_data(cpu);
    cpu += blobs[i]->data()->size() / sizeof(Dtype);
    CHECK(cpu <= params.cpu() + params.len_used());
  }
  // Check sizes
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = params.cpu() + check;
  CHECK_EQ(expect, cpu);

  params.replace_iteration_counter(solver);
}

//

void Meter::show(std::ostream& s) {
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

static int device() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

template<typename Dtype>
GPUSync<Dtype>::GPUSync(Params<Dtype>& params, Solver<Dtype>& solver) :
    Sync<Dtype>(params), //
    device_(device()), chunks_(chunks(params.len_used())), //
    cpu_(params.cpu()), //
    calls_("calls", CHUNK * sizeof(Dtype)), cycles_("cycles") {

  const size_t size = params.len_buff() * sizeof(Dtype);

  CUDA_CHECK(cudaMalloc((void** ) &gpu_, size));
  CUDA_CHECK(cudaMemcpy(gpu_, params.cpu(), size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void** ) &last_, size));
  CUDA_CHECK(cudaMemcpy(last_, gpu_, size, cudaMemcpyDeviceToDevice));

  // Replace GPU weights
  vector<shared_ptr<Blob<Dtype> > > &blobs = solver.net()->params();
  Dtype* gpu = gpu_;
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->data()->set_gpu_data(gpu);
    gpu += blobs[i]->data()->size() / sizeof(Dtype);
    CHECK(gpu <= gpu_ + params.len_used());
  }
  size_t check = 0;
  for (int i = 0; i < blobs.size(); ++i)
    check += blobs[i]->count();
  Dtype* expect = gpu_ + check;
  CHECK_EQ(expect, gpu);

  params.replace_iteration_counter(solver);
}

template<typename Dtype>
GPUSync<Dtype>::~GPUSync() {
  CUDA_CHECK(cudaFree((void* ) last_));
  CUDA_CHECK(cudaFree((void* ) gpu_));
}

template<typename Dtype>
void GPUSync<Dtype>::run() {
  CUDA_CHECK(cudaSetDevice(device_));
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Current cpu values when invoking kernel, gradients on the way back
  Dtype* chunk;
  Dtype* tmp;
  CUDA_CHECK(cudaMalloc((void** ) &chunk, CHUNK * sizeof(Dtype)));
  CaffeMallocHost((void**) &tmp, CHUNK * sizeof(Dtype));

  const size_t csize = CHUNK * sizeof(Dtype);
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  const cudaMemcpyKind get = cudaMemcpyDeviceToHost;
  uint32_t index = 0;

  for (;;) {
    const size_t off = index * CHUNK;
    CUDA_CHECK(cudaMemcpyAsync(chunk, &cpu_[off], csize, put, stream));
    GPUSync_kernel<Dtype>(gpu_, last_, chunk, off, stream);
    CUDA_CHECK(cudaMemcpyAsync(tmp, chunk, csize, get, stream));
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < CHUNK; ++i)
      cpu_[off + i] += tmp[i];
    if (++index == chunks_) {
      index = 0;
      cycles_++;
    }
    calls_++;
  }

  CaffeFreeHost((void*) tmp);
  CUDA_CHECK(cudaFree((void* ) chunk));
  cudaStreamDestroy(stream);
}

//

template<typename Dtype, typename Node>
DistSync<Dtype, Node>::DistSync(const Params<Dtype>& params, //
    vector<Node> masters, vector<Node> workers, uint32_t chunks) :
    Sync<Dtype>(params), masters_(masters), workers_(workers), //
    others_(), chunks_(chunks), cpu_(params.cpu()), //
    received_(chunks), remaining_(chunks), //
    cycles_("cycles") {

  CaffeMallocHost((void**) &last_, params.len_buff() * sizeof(Dtype));
  memcpy(last_, cpu_, params.len_used() * sizeof(Dtype));

  own_start_ = own_until_ = chunk_ = other_ = 0;
}

template<typename Dtype, typename Node>
DistSync<Dtype, Node>::~DistSync() {
  CaffeFreeHost((void*) last_);
}

template<typename Dtype, typename Node>
void DistSync<Dtype, Node>::Init(int local) {
  for (int i = 0; i < workers_.size(); ++i)
    if (i != local)
      others_.push_back(workers_[i]);
  own_start_ = (local + 0) * chunks_ / masters_.size();
  own_until_ = (local + 1) * chunks_ / masters_.size();
  chunk_ = own_start_;

  LOG(INFO)<< "range: " <<own_start_<<" "<<own_until_;

  for (uint32_t chunk = own_start_; chunk < own_until_; ++chunk) {
    received_[chunk] = true;
    remaining_--;
  }
}

template<typename Dtype, typename Node>
inline int DistSync<Dtype, Node>::master(uint32_t chunk) {
  // TODO find range without loop?
  for (int i = masters_.size() - 1; i >= 0; --i) {
    uint32_t start = i * chunks_ / masters_.size();
    if (start <= chunk)
      return i;
  }
  CHECK(false);
  return -1;
}

template<typename Dtype, typename Node>
inline void DistSync<Dtype, Node>::next() {
  if (++other_ == others_.size()) {
    other_ = 0;
    if (++chunk_ == own_until_) {
      chunk_ = own_start_;
      cycles_++;
    }
  }
}

//

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(Sync);
INSTANTIATE_CLASS(CPUSync);
INSTANTIATE_CLASS(GPUSync);
template class DistSync<float, uint8_t*> ;
template class DistSync<double, uint8_t*> ;

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

Ring::Ring(const string& adapter, int protocol_send, int protocol_recv) :
    adapter_(adapter), //
    socket_(socket(PF_PACKET, SOCK_RAW, htons(protocol_recv))), //
    sent_("sent", ETH_FRAME_LEN), recv_("recv", ETH_FRAME_LEN) {
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

bool Ring::recv(int frame, struct tpacket_hdr*& hdr) {
  hdr = (struct tpacket_hdr*) (map_recv_ + FRAME_SIZE * frame);
  int status = (volatile uint32_t) hdr->tp_status;
  CHECK(!(status & TP_STATUS_COPY));
  return status & TP_STATUS_USER;
}

ethhdr* Ring::recv_init(const struct tpacket_hdr* hdr) {
  return (struct ethhdr*) ((uint8_t*) hdr + hdr->tp_mac);
}

void Ring::recv_done(struct tpacket_hdr* hdr) {
  hdr->tp_status = TP_STATUS_KERNEL;
  recv_++;
}

bool Ring::send(int frame, struct tpacket_hdr*& hdr) {
  hdr = (struct tpacket_hdr*) (map_send_ + FRAME_SIZE * frame);
  int status = (volatile uint32_t) hdr->tp_status;
  CHECK(!(status & TP_STATUS_WRONG_FORMAT));
  return status == TP_STATUS_AVAILABLE;
}

ethhdr* Ring::send_init(const struct tpacket_hdr* hdr) {
  uint8_t* eth = ((uint8_t*) hdr) + TPACKET_ALIGN(sizeof(struct tpacket_hdr));
  return (struct ethhdr*) eth;
}

void Ring::send_done(struct tpacket_hdr* hdr) {
  hdr->tp_status = TP_STATUS_SEND_REQUEST;
  sent_++;
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
RawSync<Dtype>::RawSync(const Params<Dtype>& params, //
    const vector<string>& mac_addresses, //
    const vector<string>& secondary_macs) :
    DistSync<Dtype, uint8_t*>(params, parse_macs(mac_addresses), //
        secondary_macs.size() ?
            parse_macs(secondary_macs) : parse_macs(mac_addresses), //
        chunks(params.len_used())), //
    master_(adapter(this->masters_[local(this->masters_)]), 0x73A, 0x73B), //
    worker_(adapter(this->workers_[local(this->workers_)]), 0x73B, 0x73A) {

  int node = local(this->masters_);
  ostringstream s;
  s << "Raw socket - node: " << node << ", ";
  if (secondary_macs.size()) {
    CHECK(master_.adapter() != worker_.adapter());
    CHECK(node == local(this->workers_));
    s << "adapters: " << master_.adapter() << ", " << worker_.adapter() << endl;
  } else {
    CHECK(master_.adapter() == worker_.adapter());
    s << "adapter: " << master_.adapter() << endl;
  }
  LOG(INFO)<< s.str();

  DistSync<Dtype, uint8_t*>::Init(node);
}

template<typename Dtype>
void RawSync<Dtype>::run() {
  struct tpacket_hdr* hdr;
  struct tpacket_hdr* hdr_send;
  // TODO split over two threads? compact wire format?
  for (;;) {
    // Receive and add gradients for chunks for which we are master
    for (int f = 0; f < Ring::FRAME_NR; f++) {
      if (master_.recv(f, hdr)) {
        ethhdr* eth = master_.recv_init(hdr);
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
      if (master_.send(f, hdr)) {
        uint32_t peer = this->other_;
        uint32_t chnk = this->chunk_;
        ethhdr* eth = master_.send_init(hdr);
        memcpy(eth->h_dest, (void*) this->others_[peer], ETH_ALEN);
        uint8_t* data = (uint8_t*) eth + ETH_HLEN;
        ((uint32_t*) &(data[MSG_CHUNK]))[0] = chnk;
        Dtype* pos = (Dtype*) &(data[MSG_DATA]);
        size_t off = (size_t) chnk * CHUNK;
        memcpy(pos, this->cpu_ + off, CHUNK * sizeof(Dtype));
        master_.send_done(hdr);
        this->next();
      }
    }
    send(master_.sock(), NULL, 0, MSG_DONTWAIT);

    // Receive absolute positions for other chunks
    for (int f = 0; f < Ring::FRAME_NR; f++) {
      if (worker_.recv(f, hdr)) {
        ethhdr* eth = worker_.recv_init(hdr);
        uint8_t* data = (uint8_t*) eth + ETH_HLEN;
        uint32_t chunk = ((uint32_t*) &(data[MSG_CHUNK]))[0];
        size_t off = ((size_t) chunk) * CHUNK;
        Dtype* pos = (Dtype*) &(data[MSG_DATA]);

        // Send back the gradients if frame is available
        Dtype* grads = NULL;
        if (worker_.send(f, hdr_send)) {
          ethhdr* eth_send = worker_.send_init(hdr_send);
          uint8_t* m = this->masters_[DistSync<Dtype, uint8_t*>::master(chunk)];
          memcpy(eth_send->h_dest, (void*) m, ETH_ALEN);
          uint8_t* data_send = (uint8_t*) eth_send + ETH_HLEN;
          ((uint32_t*) &(data_send[MSG_CHUNK]))[0] = chunk;
          grads = (Dtype*) &(data_send[MSG_DATA]);
        }

        for (size_t i = 0; i < CHUNK; ++i) {
          Dtype d = this->cpu_[off + i] - this->last_[off + i];
          // If gradient is sent, reset last_ to cpu_, otherwise keep them apart
          if (grads) {
            grads[i] = d;
            this->last_[off + i] = pos[i] + d;
            this->cpu_[off + i] = this->last_[off + i];
          } else {
            this->last_[off + i] = pos[i];
            this->cpu_[off + i] = this->last_[off + i] + d;
          }
        }

        worker_.recv_done(hdr);
        if (grads)
          worker_.send_done(hdr_send);

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
