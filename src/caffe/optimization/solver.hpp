#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

namespace caffe {

class Solver {
 public:
  explicit Solver(const SolverParameter& param)
      : param_(param) {}
  void Solve(Net* net);

 protected:
  SolverParameter param_;
};

}  // namspace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_