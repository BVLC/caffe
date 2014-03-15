// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly finetune a network.
// Usage:
//    finetune_net solver_proto_file pretrained_net

#include <cuda_runtime.h>

#include <string>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: finetune_net solver_proto_file pretrained_net";
    return 1;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  LOG(INFO) << "Loading from " << argv[2];
  solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  solver.Solve();
  LOG(INFO) << "Optimization Done.";

  return 0;
}
