#ifndef CAFFE_PARALLEL_H_
#define CAFFE_PARALLEL_H_

#include <netdb.h>
#include <ctime>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/internal_thread.hpp"

using std::deque;
using boost::dynamic_bitset;
using boost::posix_time::ptime;
using boost::posix_time::microsec_clock;

// The following classes enable parallel training, over multiple CPU cores,
// GPUs, and machines. Gradients are measured and propagated between solvers
// asynchronously from backprop, to independently max out both networking
// and compute resources. Only data-parallel training is supported. Models
// can be trained in parallel without modification.

namespace caffe {

// Helper to write components running in their own threads
class Threaded : public InternalThread {
 public:
  Threaded()
      : InternalThread() {
  }

  virtual void start() {
    this->StartInternalThread();
  }
  virtual void stop() {
    this->StopInternalThread();
  }

  virtual void run() = 0;

 protected:
  void InternalThreadEntry() {
    run();
  }

DISABLE_COPY_AND_ASSIGN(Threaded);
};

// Helper for perf metrics
class Meter {
 public:
  // If unit_size is specified, meter will display bandwidth as size * count/s
  Meter(const string& name, uint64_t unit_size = 0)
      : name_(name),
        unit_size_(unit_size),  //
        value_(),
        last_(),
        time_(microsec_clock::local_time()) {
  }

  inline uint64_t value() const {
    return value_;
  }
  inline void value(uint64_t value) {
    value_ = value;
  }
  inline void operator++(int) {
    value_++;
  }

  void show(std::ostream& s) const;

 protected:
  const string name_;
  const uint64_t unit_size_;
  mutable uint64_t value_, last_;
  mutable ptime time_;  // TODO find a monotonic clock

DISABLE_COPY_AND_ASSIGN(Meter);
};

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. E.g. Params ensures
// that all parameters are allocated in one consecutive array, that the buffers
// are sufficiently long for chuncking alignments, and potentially other future
// requirements. Also keep track of the total iterations on those weights, to get
// correct hyper-parameters schedules across multiple solvers. TODO keep track
// of total iterations also between machines.
template<typename Dtype>
class Params {
 public:
  // Allocate a buffer compatible with the given blobs, optionally mapped to a
  // file (/dev/shm) for multi-process configurations or debugging.
  Params(const vector<shared_ptr<Blob<Dtype> > >& blobs,  //
      const string& file_map = "");
  virtual ~Params();

  inline size_t len_used() const {
    return len_used_;
  }
  inline size_t len_buff() const {
    return len_buff_;
  }
  inline Dtype* cpu() const {
    return cpu_;
  }
  inline int iterations() {
    return iterations_;
  }
  inline void iterations(int value) {
    iterations_ = value;
  }

  // Replaces solvers parameters by the shared buffer. Solvers then run on
  // the same weights without synchronization (Hogwild). See hogwild.cpp in
  // /examples for details and BLAS requirements.
  void configure(Solver<Dtype>* solver) const;

 protected:
  const size_t len_used_;       // Actually used
  const size_t len_buff_;       // Allocated aligned to potential chunks
  Dtype* cpu_;
  mutable int iterations_;      // Total iterations across solvers

  template<typename U>
  friend class GPUParams;

DISABLE_COPY_AND_ASSIGN(Params);
};

#ifndef CPU_ONLY

// Params on a GPU
template<typename Dtype>
class GPUParams {
 public:
  GPUParams(const Params<Dtype>& params, int device);
  virtual ~GPUParams();
  void configure(Solver<Dtype>* solver) const;

  inline const Params<Dtype>& params() const {
    return params_;
  }
  inline int device() const {
    return device_;
  }
  inline Dtype* gpu() const {
    return gpu_;
  }

 protected:
  const Params<Dtype>& params_;
  const int device_;
  Dtype* gpu_;

DISABLE_COPY_AND_ASSIGN(GPUParams);
};

template<typename Dtype>
class GPUStream {
 public:
  GPUStream();
  virtual ~GPUStream();

  const cudaStream_t& stream() const {
    return stream_;
  }

 protected:
  cudaStream_t stream_;

DISABLE_COPY_AND_ASSIGN(GPUStream);
};

// Base class for GPU synchronization.
template<typename Dtype>
class GPUSync {
 protected:
  GPUSync(const GPUParams<Dtype>& params);
  virtual ~GPUSync();

  const GPUParams<Dtype>& params_;
  Dtype* gpu_last_;

DISABLE_COPY_AND_ASSIGN(GPUSync);
};

// Syncs params between CPU and GPU memory.
template<typename Dtype>
class CPUGPUSync :  //
    public GPUSync<Dtype>,  //
    public Threaded {

 public:
  CPUGPUSync(const GPUParams<Dtype>& params);

  virtual ~CPUGPUSync();

  void run();

  inline const Meter& calls() const {
    return calls_;
  }
  inline const Meter& cycles() {
    return cycles_;
  }

  static size_t chunks(const size_t len) {
    return (len + CHUNK - 1) / CHUNK;
  }

  // TODO bench, auto tune?
  static const int CHUNK = 262144;

 protected:
  void push(uint32_t chunk);

  const uint32_t chunks_;

  // Perf counters
  Meter calls_, cycles_;
};

template<typename Dtype>
void sync_master_kernel(Dtype* gpu, Dtype** grds, size_t* offs,  //
                        int batch_start, int batch_count,  //
                        const cudaStream_t& stream, size_t chunk);

template<typename Dtype>
void sync_worker_kernel(Dtype* gpu, Dtype* last, Dtype** pos, size_t* offs,
                        Dtype** grads, uint8_t* get_grads,  //
                        int batch_start, int batch_count,  //
                        const cudaStream_t& stream, size_t chunk);

#endif

// Base class for distributed sync
template<typename Dtype>
class DistSync {
 public:
  inline const Meter& cycles() const {
    return cycles_;
  }
  bool ready() {
    return remaining_ == 0;
  }

 protected:
  DistSync(uint32_t nodes, uint32_t chunks);
  virtual ~DistSync() {
  }
  void dist_init(int local);

  // Master node for a given chunk
  inline int chunk_master(uint32_t chunk);

  const uint32_t nodes_;
  const uint32_t chunks_;

  uint32_t own_start_;  // Start of range of chunks for which this node is master
  uint32_t own_until_;  // End of this range
  uint32_t chunk_;      // Current chunk sent by master

  // Startup book-keeping, we need to know when nodes are in sync
  // TODO replace by transfer of initial weights?
  dynamic_bitset<> received_;
  uint32_t remaining_;

  // Perf counter
  Meter cycles_;

DISABLE_COPY_AND_ASSIGN(DistSync);
};

#ifdef RDMA
#include <infiniband/verbs.h>
#include <infiniband/umad.h>
#include "caffe/util/multicast_resources.hpp"

struct ib_addr {
  ibv_gid gid;  // Only used for multicast addresses
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
  ibv_ah* ah;
};

template<typename Dtype>
class IBSync;

class IBChannel {
 public:
  IBChannel(ibv_device* ib_dev);
  ~IBChannel();
  inline const ib_addr& address() const {
    return local_;
  }

  ib_addr mcast_create() const;
  void mcast_join(const ib_addr& addr) const;
  void mcast_attach_qp(const ib_addr& addr) const;

  void start(uint8_t* buf_send, size_t buf_size, bool gpu = false) const;

  inline const string adapter() const {
    return string(context_->device->dev_name);
  }

  // Stats
  inline const Meter& sent() const {
    return sent_;
  }
  inline const Meter& recv() const {
    return recv_;
  }

  static const int MTU = 4096;  // TODO get at runtime
  static const int FRAMES = 1024;  // TODO bench

 protected:
  // TODO make port configurable
  static const int PORT = 1;

  bool can_send() const;
  int send_init(uint8_t*& buf) const;
  void send(int id, const ib_addr& addr, uint8_t* buf, uint32_t imm_data) const;

  bool can_recv() const;
  int recv(uint8_t*& buf, uint32_t& imm_data) const;
  void recv_done(int id) const;

  void poll() const;

  static ibv_context* open_device(ibv_device* ib_dev);
  static ibv_pd* alloc_pd(ibv_context*);

  // TODO align recv buffers to CACHE_LINE_SIZE     (64) - GRH
  static const int GRH = 40;  // Global Routing Header

  ibv_context* context_;
  ibv_pd* pd_;
  ib_addr local_;
  ibv_cq* cq_;
  ibv_qp* qp_;

  mutable uint8_t* buf_send_;
  mutable uint8_t* buf_recv_;
  mutable ibv_mr* mr_send_;
  mutable ibv_mr* mr_recv_;

  struct recv_msg {
    uint32_t id_;
    uint32_t imm_;
  };

  mutable ibv_wc wc_[IBChannel::FRAMES * 2];
  mutable deque<uint16_t> send_queue_;
  mutable deque<recv_msg> recv_queue_;

  mutable Meter sent_, recv_;

  template<typename U>
  friend class IBSync;
  template<typename U>
  friend class CPUIBSync;
  template<typename U>
  friend class GPUIBSync;

DISABLE_COPY_AND_ASSIGN(IBChannel);
};

// Synchronization over InfiniBand
template<typename Dtype>
class IBSync : public DistSync<Dtype>, public Threaded {
 public:
  inline const IBChannel& ucast() const {
    return ucast_;
  }
  inline const IBChannel& mcast() const {
    return mcast_;
  }

  static size_t chunks(const size_t len) {
    return (len + CHUNK - 1) / CHUNK;
  }

  static const int CHUNK = IBChannel::MTU / sizeof(Dtype);

 protected:
  IBSync(const Params<Dtype>& params, int rank,  //
         const IBChannel& ucast,  //
         const IBChannel& mcast,  //
         const vector<ib_addr>& ucast_addrs,  //
         const vector<ib_addr>& mcast_addrs);
  ~IBSync();

  const int rank_;
  const IBChannel& ucast_;
  const IBChannel& mcast_;
  vector<ib_addr> ucast_addrs_;
  ib_addr mcast_addr_;
};

// InfiniBand to and from host memory
template<typename Dtype>
class CPUIBSync : public IBSync<Dtype> {
 public:
  CPUIBSync(const Params<Dtype>& params, int rank,
            const IBChannel& ucast,
            const IBChannel& mcast,  //
            const vector<ib_addr>& ucast_addrs,
            const vector<ib_addr>& mcast_addrs);
  ~CPUIBSync();
  void run();

 protected:
  Dtype* cpu_;
  Dtype* cpu_last_;
};

#ifndef CPU_ONLY

// InfiniBand to and from GPU memory
template<typename Dtype>
class GPUIBSync : public GPUSync<Dtype>, public IBSync<Dtype> {
 public:
  GPUIBSync(const GPUParams<Dtype>& params, int rank,
            const IBChannel& ucast,
            const IBChannel& mcast,  //
            const vector<ib_addr>& ucast_addrs,
            const vector<ib_addr>& mcast_addrs);
  ~GPUIBSync();
  void run();

  static const int FRAMES = IBChannel::FRAMES;

 protected:
  Dtype* gpu_;
  Dtype* gpu_last_;
};

#endif // not CPU_ONLY

#endif // RDMA

#ifdef __linux__

#include <linux/if_ether.h>
#include <linux/if_packet.h>

// User-space networking ring buffer.
class Ring {
 public:
  Ring(const string& adapter, int protocol_send, int protocol_recv);

  ~Ring();

  inline bool can_send(int frame, struct tpacket_hdr*& hdr);
  inline ethhdr* send_init(const struct tpacket_hdr* hdr);
  inline void send(struct tpacket_hdr* hdr);

  inline bool can_recv(int frame, struct tpacket_hdr*& hdr);
  inline ethhdr* recv(const struct tpacket_hdr* hdr);
  inline void recv_done(struct tpacket_hdr* hdr);

  inline const string& adapter() const {
    return adapter_;
  }
  inline int sock() {
    return socket_;
  }

  // Stats
  inline const Meter& sent() const {
    return sent_;
  }
  inline const Meter& recv() const {
    return recv_;
  }
  void socket_stats(uint64_t& received, uint64_t& dropped);

  static const int FRAME_SIZE = 2048;  // TODO bench
  static const int FRAME_NR = 32;
  static const int BLOCK_NR = 1;

 protected:
  const string adapter_;
  const int socket_;
  const uint8_t* map_recv_;
  const uint8_t* map_send_;

  Meter sent_, recv_;

DISABLE_COPY_AND_ASSIGN(Ring);
};

// Synchronization using raw sockets and user-space networking. Can be a very
// efficient alternative to RDMA if not available, but cannot read and write
// directly to GPU memory.
//  C.f. https://www.kernel.org/doc/Documentation/networking/packet_mmap.txt
template<typename Dtype>
class RawSync : public DistSync<Dtype>, public Threaded {
 public:
  RawSync(const Params<Dtype>& params,  //
      const vector<string>& mac_addresses,  //
      const vector<string>& secondary_macs);

  ~RawSync();

  void run();

  inline const Ring& master() const {
    return master_;
  }
  inline const Ring& worker() const {
    return worker_;
  }

  static size_t chunks(const size_t len) {
    return (len + CHUNK - 1) / CHUNK;
  }

  // Offsets of parts of a message
  static const int MSG_CHUNK = 0;
  static const int MSG_DATA = sizeof(Dtype);

  static const int CHUNK = (ETH_DATA_LEN - MSG_DATA) / sizeof(Dtype);

 protected:
  // Next chunk
  inline void next();

  // Currently all nodes are both masters and workers for some chunks,
  // so the two vectors should be equal. If machines have two adapters,
  // workers can point to the secondary adapters for better performance.
  const vector<uint8_t*> masters_;
  const vector<uint8_t*> workers_;
  vector<uint8_t*> others_;   // Workers without the local one
  uint32_t other_;            // Current node the chunk is sent to

  Ring master_;
  Ring worker_;

  Dtype* cpu_;
  Dtype* cpu_last_;
};

#endif // __linux__
}

#endif
