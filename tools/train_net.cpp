// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <cstring>
#include <ctime>

#include "caffe/caffe.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2 || argc > 3) {
    LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
    return 1;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  boost::posix_time::ptime  start_timer =
           boost::posix_time::microsec_clock::local_time();

  SGDSolver<float> solver(solver_param);
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver.Solve(argv[2]);
  } else {
    solver.Solve();
  }
  LOG(INFO) << "Solver: Optimization Done.";

  boost::posix_time::ptime  stop_timer =
          boost::posix_time::microsec_clock::local_time();
  uint64_t elapsed_sec=(stop_timer - start_timer).total_milliseconds()/1000;
  LOG(INFO) << "Elapsed time, sec: " << elapsed_sec;

  return 0;
}
