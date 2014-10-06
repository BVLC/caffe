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

using boost::dynamic_bitset;
using boost::posix_time::ptime;
using boost::posix_time::microsec_clock;

// The following classes enable parallel training, over multiple CPUs,
// GPUs, and machines. Gradients are measured and propagated between solvers
// asynchronously from backprop, to independently max out both networking
// and compute resources. Only data-parallel training is supported. Models
// can be trained in parallel without modification.

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. E.g. Params ensures
// that all parameters are allocated in one consecutive array, that the buffers
// are sufficiently long for chuncking alignments, and potentially other future
// requirements. Also keep track of the total iterations on those weights, to get
// correct hyper-parameters schedules across multiple solvers.
template<typename Dtype>
class Params {
public:
  // Allocate a buffer compatible with the given blobs, optionally mapped to a
  // file (/dev/shm) for multi-process configurations or debugging.
  Params(const vector<shared_ptr<Blob<Dtype> > >& blobs, //
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
  void replace_iteration_counter(Solver<Dtype>& solver) {
    solver.iter_total(&iterations_);
  }

protected:
  const size_t len_used_;       // Actually used
  const size_t len_buff_;       // Allocated aligned to potential chunks
  Dtype* cpu_;
  int iterations_;              // Total iterations across solvers

DISABLE_COPY_AND_ASSIGN(Params);
};

// Base class for parameter synchronization
template<typename Dtype>
class Sync {
public:
  Sync(const Params<Dtype>& params) :
      params_(params) {
  }

protected:
  const Params<Dtype>& params_;

DISABLE_COPY_AND_ASSIGN(Sync);
};

// Allows multiple CPU solvers to train a net in parallel. This class doesn't
// actively exchange gradients, it simply replaces all solvers parameters by
// a shared buffer from Params. Solvers run on the same weights without
// synchronization (Hogwild), which works better in practice than in theory.
// See hogwild.cpp in /examples for details and BLAS requirements.
template<typename Dtype>
class CPUSync: public Sync<Dtype> {
public:
  CPUSync(Params<Dtype>& params, Solver<Dtype>& solver);

  virtual ~CPUSync() {
  }
};

// Helper to write components running in their own threads.
// TODO merge with Caffe Thread
class PThread {
public:
  PThread() {
    memset(&thread_, 0, sizeof(pthread_t));
  }

  virtual ~PThread() {
  }

  // In case a component needs to run on an existing thread
  void init_as_current() {
    thread_ = pthread_self();
  }

  virtual void start() {
    CHECK(!pthread_create(&thread_, NULL, run, this));
  }

  virtual void run() = 0;

protected:
  static void* run(void* ptr) {
    PThread* t = (PThread*) ptr;
    t->run();
    return NULL;
  }

  pthread_t thread_;

DISABLE_COPY_AND_ASSIGN(PThread);
};

// Helper for perf metrics
class Meter {
public:
  // If unit_size is specified, meter will display bandwidth as size * count/s
  Meter(const string& name, uint64_t unit_size = 0) :
      name_(name), unit_size_(unit_size), //
      value_(), last_(), time_(microsec_clock::local_time()) {
  }

  inline uint64_t value() const {
    return value_;
  }
  inline void value(uint64_t value) {
    value_ = value;
  }

  Meter& operator++() {
    value_++;
    return *this;
  }
  Meter operator++(int) {
    Meter tmp(*this);
    operator++();
    return tmp;
  }

  void show(std::ostream& s);

protected:
  const string name_;
  const uint64_t unit_size_;
  uint64_t value_, last_;
  ptime time_; // TODO find a monotonic clock
};

// Syncs params between the host and GPU memory. Does not currently use GPU
// to GPU communication to avoid allocating additional buffers on the GPUs.
// The host keeps track of the last values to evaluate the gradient.
template<typename Dtype>
class GPUSync: public Sync<Dtype>, public PThread {
public:
  // TODO bench, auto tune?
  static const int CHUNK = 0x100000;

  GPUSync(Params<Dtype>& params, Solver<Dtype>& solver);

  virtual ~GPUSync();

  void run();
  void kernel(size_t off);

  inline Meter calls() const {
    return calls_;
  }
  inline Meter cycles() {
    return cycles_;
  }

  static size_t chunks(const size_t len) {
    return (len + CHUNK - 1) / CHUNK;
  }

protected:
  void push(uint32_t chunk);

  const int device_;
  const uint32_t chunks_;

  Dtype* cpu_;
  Dtype* gpu_;
  Dtype* last_;

  // Perf counters
  Meter calls_, cycles_;
};

template<typename Dtype>
void GPUSync_kernel(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off, cudaStream_t& stream);

// Base class for distributed sync
template<typename Dtype, typename Node>
class DistSync: public Sync<Dtype> {
public:
  DistSync(const Params<Dtype>& params, //
      vector<Node> masters, vector<Node> workers, //
      uint32_t chunks);

  virtual ~DistSync();

  inline Meter cycles() const {
    return cycles_;
  }

  bool ready() {
    return remaining_ == 0;
  }

protected:
  void Init(int local);

  // Master node for a given chunk
  inline int master(uint32_t chunk);

  // Next chunk
  inline void next();

  // Currently all nodes are both masters and workers for some chunks,
  // so the two vectors should be equal. If machines have two adapters,
  // workers can point to the secondary adapters for better performance.
  const vector<Node> masters_;
  const vector<Node> workers_;
  vector<Node> others_;             // Workers without the local one

  // int32 to have same size as floats on the wire
  const uint32_t chunks_;

  uint32_t own_start_; // Start of range of chunks for which this node is master
  uint32_t own_until_; // End of this range
  uint32_t chunk_;     // Current chunk sent by master
  uint32_t other_;     // Current node the chunk is sent to

  Dtype* cpu_;
  Dtype* last_;

  // Startup book-keeping, we need to know when nodes are in sync
  // TODO replace by transfer of initial weights?
  dynamic_bitset<> received_;
  uint32_t remaining_;

  // Perf counter
  Meter cycles_;
};

#ifdef __linux__

#include <linux/if_ether.h>
#include <linux/if_packet.h>

// User-space networking ring buffer.
class Ring {
public:
  static const int FRAME_SIZE = 2048; // TODO bench
  static const int FRAME_NR = 32;
  static const int BLOCK_NR = 1;

  Ring(const string& adapter, int protocol_send, int protocol_recv);

  ~Ring();

  inline bool recv(int frame, struct tpacket_hdr*& hdr);
  inline ethhdr* recv_init(const struct tpacket_hdr* hdr);
  inline void recv_done(struct tpacket_hdr* hdr);

  inline bool send(int frame, struct tpacket_hdr*& hdr);
  inline ethhdr* send_init(const struct tpacket_hdr* hdr);
  inline void send_done(struct tpacket_hdr* hdr);

  inline const string& adapter() const {
    return adapter_;
  }
  inline int sock() {
    return socket_;
  }

  // Stats
  inline Meter sent() const {
    return sent_;
  }
  inline Meter recv() const {
    return recv_;
  }
  void socket_stats(uint64_t& received, uint64_t& dropped);

protected:
  const string adapter_;
  const int socket_;
  const uint8_t* map_recv_;
  const uint8_t* map_send_;

  Meter sent_, recv_;

  DISABLE_COPY_AND_ASSIGN(Ring);
};

// High performance synchronization using raw sockets and user-space networking
//  C.f. https://www.kernel.org/doc/Documentation/networking/packet_mmap.txt
template<typename Dtype>
class RawSync: public DistSync<Dtype, uint8_t*>, public PThread {
public:
  // Offsets of parts of a message
  static const int MSG_CHUNK = 0;
  static const int MSG_DATA = sizeof(Dtype);

  static const int CHUNK = (ETH_DATA_LEN - MSG_DATA) / sizeof(Dtype);

  RawSync(const Params<Dtype>& params,//
      const vector<string>& mac_addresses,//
      const vector<string>& secondary_macs);

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

protected:
  Ring master_;
  Ring worker_;
};

#endif
}

#endif
