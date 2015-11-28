#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <sys/socket.h>

#include "caffe/caffe.hpp"
#include "caffe/filler.hpp"
#include "caffe/parallel.hpp"
#include "base.hpp"

#ifdef RDMA
#include <infiniband/verbs.h>

using namespace std;

// Trains a net on multiple boxes over InfiniBand or RoCE. RDMA addresses
// are exchanged over a socket, first launch a server instance, then clients:

// Server: rdma.bin <nbr_clients> port solver.prototxt <gpu 0>:<gpu n>:<cpus>
// Client: rdma.bin      <server> port solver.prototxt <gpu 0>:<gpu n>:<cpus>

// e.g. for 4 machines with 4 GPUs each:
// rdma.bin        3 4444 examples/parallel/mnist_solver.prototxt 0:1:2:3:0
// then 3 times:
// rdma.bin <server> 4444 examples/parallel/mnist_solver.prototxt 0:1:2:3:0

// Monitors solvers and network
class IBMonitor : public Monitor {
 public:
  IBMonitor(Params<float>& params, const vector<SolverContext*>& solvers,
            const vector<IBSync<float>*> syncs,
            vector<GPUParams<float>*> gpu_params)
      : Monitor(params, solvers),
        syncs_(syncs),
        gpu_params_(gpu_params) {
  }

  void stats(const IBChannel& c, ostream& s) {
    s << c.adapter() + " ";
    c.sent().show(s);
    s << ", ";
    c.recv().show(s);
  }

  void run() {
    sleep(5);
    time_t start = time(0);
    void* d0;
    void* d1;
    size_t len = gpu_params_[0]->params().len_buff();
    size_t size = len * sizeof(float);
    CaffeMallocHost(&d0, size);
    CaffeMallocHost(&d1, size);
    for (;;) {
      sleep(2);

      ostringstream s;
      step(&s);

      for (int i = 0; i < syncs_.size(); ++i) {
        s << "RDMA " << i << ": ucast ";
        stats(syncs_[i]->ucast(), s);
        s << ", mcast ";
        stats(syncs_[i]->mcast(), s);
        s << ", ";
        syncs_[i]->cycles().show(s);
        s << "\n";
      }
      LOG(INFO)<< s.str();
      LOG(INFO)<< "Training time: " << (time(0) - start);
    }
  }

  const vector<IBSync<float>*> syncs_;
  const vector<GPUParams<float>*> gpu_params_;
};

static void exch_server(const int clients, const char* port,
                        vector<ib_addr>* ucast_addrs,
                        vector<ib_addr>* mcast_addrs);
static int exch_client(const char* server, const char* port,
                       vector<ib_addr>* ucast_addrs,
                       vector<ib_addr>* mcast_addrs);

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();

  if (argc != 5) {
    printf("Usage: ib.bin <server/nbr_clients> port solver_proto_file "  //
        "[gpu_id][:gpu_id][...]:cpu_cores\n");
    return 1;
  }

  const int clients = atoi(argv[1]);
  const bool server = clients != 0;
  char* host = argv[1];
  char* port = argv[2];

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[3], &solver_param);

  vector<string> procs;
  boost::split(procs, argv[4], boost::is_any_of(":"));
  vector<int> gpus;
  for (int i = 0; i < procs.size() - 1; ++i)
    gpus.push_back(atoi(procs[i].c_str()));
  int cores = atoi(procs[procs.size() - 1].c_str());

  // Get IB device

  ibv_device** dev_list;
  ibv_device* ib_dev;
  dev_list = ibv_get_device_list(NULL);
  CHECK(dev_list) << "No IB devices found";
  ib_dev = dev_list[0];
  CHECK(ib_dev) << "No IB devices found";

  // Create IB channels for exchanging positions and gradients

  const int channels = gpus.size() + (cores > 0 ? 1 : 0);
  vector<IBChannel*> ucast(channels);
  vector<IBChannel*> mcast(channels);
  vector<ib_addr> ucast_addrs(channels);
  vector<ib_addr> mcast_addrs(channels);
  for (int i = 0; i < channels; ++i) {
    ucast[i] = new IBChannel(ib_dev);
    mcast[i] = new IBChannel(ib_dev);
    ucast_addrs[i] = ucast[i]->address();
    mcast_addrs[i] = mcast[i]->mcast_create();
  }

  // Exchange IB addresses

  int rank;
  if (server) {
    if (clients > 0)
      exch_server(clients, port, &ucast_addrs, &mcast_addrs);
    rank = 0;
  } else {
    rank = exch_client(host, port, &ucast_addrs, &mcast_addrs);
  }

  // Create main solver (first GPU if available, or CPU)

  if (gpus.size())
    solver_param.set_device_id(gpus[0]);
  else
    solver_param.set_solver_mode(SolverParameter::CPU);
  SGDSolver<float> main(solver_param, rank != 0);

  Params<float> params(main.net()->params());  //, "/dev/shm/test");

  // Create syncs

  bool sync = true;
  vector<GPUParams<float>*> gpu_params;
  vector<IBSync<float>*> syncs;
  for (int i = 0; i < gpus.size(); ++i) {
    gpu_params.push_back(new GPUParams<float>(params, gpus[i]));
    syncs.push_back(new GPUIBSync<float>(*gpu_params.back(), rank + i,  //
                                         *ucast[i],  //
                                         *mcast[i],  //
                                         ucast_addrs,  //
                                         mcast_addrs));
    if (sync)
      syncs.back()->start();
  }
  if (cores > 0) {
    syncs.push_back(new CPUIBSync<float>(params, rank + gpus.size(),  //
                                         *ucast[gpus.size()],  //
                                         *mcast[gpus.size()],  //
                                         ucast_addrs,  //
                                         mcast_addrs));
    syncs.back()->start();
  }

  // Wait for weights to be in sync

  LOG(INFO)<< "Waiting for other boxes\n";
  bool ready = false;
  while (sync && !ready) {
    sleep(1);
    ready = true;
    for (int i = 0; i < syncs.size(); ++i) {
      if (!syncs[i]->ready()) {
        ready = false;
      }
    }
  }
  LOG(INFO)<< "Start training\n";

  // Create contexts
  vector<SolverContext*> contexts(gpus.size() + cores);
  if (gpus.size()) {
#ifndef CPU_ONLY
    contexts[0] = new GPUContext(params, solver_param, gpu_params[0], &main);
#else
    NO_GPU;
#endif
  } else {
    contexts[0] = new CPUContext(params, solver_param, &main);
  }
#ifndef CPU_ONLY
  // GPUs
  for (int i = 1; i < gpus.size(); ++i) {
    solver_param.set_device_id(gpus[i]);
    contexts[i] = new GPUContext(params, solver_param, gpu_params[i]);
    contexts[i]->start();
  }
#endif
  // CPUs
  solver_param.set_solver_mode(SolverParameter::CPU);
  for (int i = max(1, (int) gpus.size()); i < gpus.size() + cores; ++i) {
    contexts[i] = new CPUContext(params, solver_param);
    contexts[i]->start();
  }

  // Start monitor
  IBMonitor monitor(params, contexts, syncs, gpu_params);
  monitor.start();

  // Run main on current thread
  contexts[0]->run();

  monitor.stop();
  LOG(INFO)<< "Monitor stop\n";

  for (int i = 1; i < contexts.size(); ++i)
    contexts[i]->stop();

  for (int i = 1; i < contexts.size(); ++i)
    delete contexts[i];

  ibv_free_device_list(dev_list);
}

// Exchange addresses through socket, c.f. IB perftest

static void exch_server(const int clients, const char* port,
                        vector<ib_addr>* ucast_addrs,
                        vector<ib_addr>* mcast_addrs) {
  struct addrinfo *res, *t;
  struct addrinfo hints;
  memset(&hints, 0, sizeof hints);
  hints.ai_flags = AI_PASSIVE;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  int n = getaddrinfo(NULL, port, &hints, &res);
  if (n < 0) {
    fprintf(stderr, "%s for port %s\n", gai_strerror(n), port);
    return;
  }
  int s = -1;
  for (t = res; t; t = t->ai_next) {
    s = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (s >= 0) {
      int n = 1;
      setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);
      if (!bind(s, t->ai_addr, t->ai_addrlen))
        break;
      close(s);
      s = -1;
    }
  }
  freeaddrinfo(res);
  if (s < 0) {
    fprintf(stderr, "Couldn't listen to port %s\n", port);
    return;
  }

  printf("Listening to port %s\n", port);
  listen(s, 1);
  vector<int> connections(clients);
  vector<int> ranks(clients);

  for (int i = 0; i < connections.size(); ++i) {
    connections[i] = accept(s, NULL, 0);
    if (connections[i] < 0) {
      fprintf(stderr, "accept() failed\n");
      return;
    }
    LOG(INFO)<< "Client " << i << " of " << connections.size() << " connected\n";
    int count;
    CHECK(read(connections[i], &count, sizeof(int)) == sizeof(int));
    ranks[i] = ucast_addrs->size();
    ucast_addrs->resize(ranks[i] + count);
    mcast_addrs->resize(ranks[i] + count);
    int bytes = sizeof(ib_addr) * count;
    CHECK(read(connections[i], &ucast_addrs->at(ranks[i]), bytes) == bytes);
    CHECK(read(connections[i], &mcast_addrs->at(ranks[i]), bytes) == bytes);
  }

  for (int i = 0; i < connections.size(); ++i) {
    int count = ucast_addrs->size();
    CHECK(write(connections[i], &ranks[i], sizeof(int)) == sizeof(int));
    CHECK(write(connections[i], &count, sizeof(int)) == sizeof(int));
    int bytes = sizeof(ib_addr) * count;
    CHECK(write(connections[i], &ucast_addrs->at(0), bytes) == bytes);
    CHECK(write(connections[i], &mcast_addrs->at(0), bytes) == bytes);
    close(connections[i]);
  }

  close(s);
}

static int exch_client(const char* server, const char* port,
                       vector<ib_addr>* ucast_addrs,
                       vector<ib_addr>* mcast_addrs) {
  addrinfo *res;
  addrinfo hints;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  int n = getaddrinfo(server, port, &hints, &res);
  if (n < 0) {
    fprintf(stderr, "%s for %s:%s\n", gai_strerror(n), server, port);
    return -1;
  }
  int s = -1;
  for (addrinfo* t = res;; t = t->ai_next) {
    s = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (s >= 0) {
      if (!connect(s, t->ai_addr, t->ai_addrlen))
        break;
      close(s);
      s = -1;
    }
  }
  freeaddrinfo(res);
  if (s < 0) {
    fprintf(stderr, "Couldn't connect to %s:%s\n", server, port);
    return -1;
  }
  LOG(INFO)<< "Connected to server\n";

  int bytes, rank, count = ucast_addrs->size();
  CHECK(write(s, &count, sizeof(int)) == sizeof(int));
  bytes = sizeof(ib_addr) * count;
  CHECK(write(s, &ucast_addrs->at(0), bytes) == bytes);
  CHECK(write(s, &mcast_addrs->at(0), bytes) == bytes);

  CHECK(read(s, &rank, sizeof(int)) == sizeof(int));
  CHECK(read(s, &count, sizeof(int)) == sizeof(int));
  ucast_addrs->resize(count);
  mcast_addrs->resize(count);
  bytes = sizeof(ib_addr) * count;
  CHECK(read(s, &ucast_addrs->at(0), bytes) == bytes);
  CHECK(read(s, &mcast_addrs->at(0), bytes) == bytes);
  return rank;
}

#else
int main(int argc, char *argv[]) {
}
#endif

