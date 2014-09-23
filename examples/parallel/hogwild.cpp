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
#include <sys/socket.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <netdb.h>
#include <cuda_runtime.h>

#include <caffe/caffe.hpp>
#include "caffe/filler.hpp"
#include "caffe/parallel.hpp"
#include "base.hpp"

using namespace std;
using namespace caffe;

// Trains a net in parallel on multiple CPU cores. C.f. CPUSync in parallel.h.
//
// Your BLAS library needs to let the application manage its threads, e.g.
// for OpenBLAS, compile with no threading (USE_THREAD = 0 in Makefile.rule).
// Performance is linear at first, but then plateaus on large nets as the number
// of cores is increased, probably as the CPU runs out of memory bandwidth.
//
// Example launch on 4 cores:
// make -j
// export LD_LIBRARY_PATH=<single thread BLAS>:/usr/local/lib:/usr/local/cuda/lib64
// export GLOG_logtostderr=1
// build/examples/parallel/hogwild.bin examples/parallel/mnist_solver.prototxt 4

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::google::InstallFailureSignalHandler();

  if (argc < 2 || argc > 3) {
    printf("Usage: hogwild.bin solver_proto_file [number_of_cores]\n");
    return 1;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  int cores = argc == 3 ? atoi(argv[2]) : sysconf(_SC_NPROCESSORS_ONLN);

  // Override in code so that proto file can be shared with other examples
  solver_param.set_solver_mode(SolverParameter::CPU);

  // Main solver
  SGDSolver<float> main(solver_param);

  // Shared network weights
  Params<float> params(main.net()->params());

  // Create contexts
  vector<SolverContext*> solvers(cores);
  solvers[0] = new CPUContext(params, main);
  for (int i = 1; i < cores; ++i) {
    solvers[i] = new CPUContext(params, solver_param, i);
    solvers[i]->start();
  }

  // Start monitor
  Monitor monitor(params, solvers);
  monitor.start();

  // Run main on current thread
  solvers[0]->run();
}
