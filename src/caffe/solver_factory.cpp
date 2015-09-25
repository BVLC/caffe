#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  SolverParameter_SolverType type = param.solver_type();

  switch (type) {
  case SolverParameter_SolverType_SGD:
    return new SGDSolver<Dtype>(param);
  case SolverParameter_SolverType_NESTEROV:
    return new NesterovSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAGRAD:
    return new AdaGradSolver<Dtype>(param);
  case SolverParameter_SolverType_RMSPROP:
    return new RMSPropSolver<Dtype>(param);
  case SolverParameter_SolverType_ADADELTA:
    return new AdaDeltaSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAM:
    return new AdamSolver<Dtype>(param);
  default:
    LOG(FATAL) << "Unknown SolverType: " << type;
  }
  return (Solver<Dtype>*) NULL;
}

template Solver<float>* GetSolver(const SolverParameter& param);
template Solver<double>* GetSolver(const SolverParameter& param);

}  // namespace caffe
