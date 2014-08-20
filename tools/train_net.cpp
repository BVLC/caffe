#include "caffe/caffe.hpp"

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
  shared_ptr<Solver<float> > solver(GetSolver<float>(solver_param));
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver->Solve(argv[2]);
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
