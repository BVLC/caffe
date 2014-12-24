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
#include <cstdlib>
#include <string>
#include <vector>

#include "base.hpp"

using namespace std;
using namespace caffe;

#ifndef CPU_ONLY

// Trains a net on multiple GPUs on one box. C.f. GPUSync in parallel.h.
//
// Example launch on GPU 0 and 1:
// make -j
// export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64
// export GLOG_logtostderr=1
// build/examples/parallel/gpus.bin examples/parallel/mnist_solver.prototxt 0:1

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();

  if (argc != 3) {
    printf("Usage: gpus.bin solver_proto_file gpu_id[:gpu_id][...]\n");
    return 1;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  vector<int> gpus;
  vector<string> gpu_strings;
  boost::split(gpu_strings, argv[2], boost::is_any_of(":"));
  for (int i = 0; i < gpu_strings.size(); ++i)
    gpus.push_back(atoi(gpu_strings[i].c_str()));

  solver_param.set_device_id(gpus[0]);
  SGDSolver<float> main(solver_param);

  // Shared network weights
  Params<float> params(main.net()->params());

  // Create contexts
  vector<SolverContext*> solvers(gpus.size());
  solvers[0] = new CPUGPUContext(params, solver_param, &main);
  for (int i = 1; i < gpus.size(); ++i) {
    solver_param.set_device_id(gpus[i]);
    solvers[i] = new CPUGPUContext(params, solver_param);
    solvers[i]->start();
  }

  // Start monitor
  Monitor monitor(params, solvers);
  monitor.start();

  // Run main on current thread
  solvers[0]->run();

  monitor.stop();
  LOG(INFO)<< "Monitor stop\n";

  for (int i = 1; i < solvers.size(); ++i)
    solvers[i]->stop();

  for (int i = 1; i < solvers.size(); ++i)
    delete solvers[i];
}

#else
int main(int argc, char *argv[]) {
}
#endif

