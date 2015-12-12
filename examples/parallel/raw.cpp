#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <caffe/net.hpp>
#include <caffe/parallel.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/solver.hpp>
#include <caffe/util/io.hpp>
#include <glog/logging.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "base.hpp"

using namespace std;
using namespace caffe;

// Trains a net over multiple boxes through high perf. networking. C.f RawSync in
// parallel.h. The application must be launched on each box with the local CPUs & GPUs
// to use, and the list of MAC addresses of all adapters in the cluster. The MAC
// list must be the same on all boxes.
//
// Example launch on GPU 0, GPU 1, 4 cores on two machines:
// make -j
// (single thread BLAS is only needed for CPU training, c.f. hogwild.cpp)
// export LD_LIBRARY_PATH=<single thread BLAS>:/usr/local/lib:/usr/local/cuda/lib64
// export GLOG_logtostderr=1
// build/examples/parallel/raw.bin examples/parallel/mnist_solver.prototxt 0:1:4 002590ca9998:002590ca9956

#ifdef __linux__

// Monitors solvers and network
class RawMonitor : public Monitor {
 public:
  RawMonitor(Params<float>& params, const vector<SolverContext*>& solvers,
             RawSync<float>& raw)
      : Monitor(params, solvers),
        raw_(raw) {
  }

  void stats(const Ring& r, ostream& s) {
    s << r.adapter() + " ";
    r.sent().show(s);
    s << ", ";
    r.recv().show(s);
  }

  void run() {
    time_t start = time(0);
    for (;;) {
      sleep(10);

      ostringstream s;
      step(&s);

      s << "raw: ";
      stats(raw_.master(), s);
      s << ", ";
      stats(raw_.worker(), s);
      s << ", ";
      raw_.cycles().show(s);
      s << "\n";
      LOG(INFO)<< s.str();
      LOG(INFO)<< "Training time: " << (time(0) - start);
    }
  }

  const RawSync<float>& raw_;
};

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();

  if (argc < 4 || argc > 5) {
    printf("Usage: raw.bin solver_proto_file "  //
        "[gpu_id][:gpu_id][...]:cpu_cores "
        "mac_address[:mac_address][:...] [secondary_mac][:secondary_mac][:...]\n");
    printf("Raw socket is a privileged operation, either run as root or "  //
        "set the capability on the executable: "
        "sudo setcap cap_net_raw+ep raw.bin\n");
    return 1;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  vector<string> procs;
  boost::split(procs, argv[2], boost::is_any_of(":"));
  vector<int> gpus;
  for (int i = 0; i < procs.size() - 1; ++i)
    gpus.push_back(atoi(procs[i].c_str()));
  int cores = atoi(procs[procs.size() - 1].c_str());

  vector<string> macs;
  boost::split(macs, argv[3], boost::is_any_of(":"));

  vector<string> secs;
  if (argc == 5)
    boost::split(secs, argv[4], boost::is_any_of(":"));

  // Set main solver to first GPU if available, or CPU
  if (gpus.size())
    solver_param.set_device_id(gpus[0]);
  else
    solver_param.set_solver_mode(SolverParameter::CPU);
  SGDSolver<float> main(solver_param);

  // Shared network weights
  Params<float> params(main.net()->params(), "/dev/shm/test");

  // Raw socket synchronization
  RawSync<float> raw(params, macs, secs);
  raw.start();

  LOG(INFO)<< "Waiting for other boxes\n";
  while (!raw.ready())
    sleep(1);
  LOG(INFO)<< "Start training\n";

  // Create contexts
  vector<SolverContext*> contexts(gpus.size() + cores);
  if (gpus.size()) {
#ifndef CPU_ONLY
    contexts[0] = new CPUGPUContext(params, solver_param, &main);
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
    contexts[i] = new CPUGPUContext(params, solver_param);
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
  RawMonitor monitor(params, contexts, raw);
  monitor.start();

  // Run main on current thread
  contexts[0]->run();

  monitor.stop();
  LOG(INFO)<< "Monitor stop\n";

  for (int i = 1; i < contexts.size(); ++i)
    contexts[i]->stop();

  for (int i = 1; i < contexts.size(); ++i)
    delete contexts[i];
}

#endif
