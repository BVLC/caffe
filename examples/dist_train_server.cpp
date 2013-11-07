// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 3) {
    LOG(ERROR) << "Usage: dist_train_server solver_proto_file (server|client) [resume_point_file]";
    return 0;
  }

  //Caffe::SetDevice(0);
  Caffe::set_mode(Caffe::CPU);

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  shared_ptr<Solver<float> > solver;
  if (strcmp(argv[2], "server") == 0) {
    solver.reset(new DistributedSolverParamServer<float>(solver_param));
  } else if (strcmp(argv[2], "client") == 0) {
    solver.reset(new DistributedSolverParamClient<float>(solver_param));
  }

  if (argc == 4) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver->Solve(argv[2]);
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
