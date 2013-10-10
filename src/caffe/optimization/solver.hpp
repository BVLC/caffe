// Copyright Yangqing Jia 2013

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <vector>

namespace caffe {

template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param)
      : param_(param) {}
  // The main entry of the solver function.
  void Solve(Net<Dtype>* net);
  virtual ~Solver() {}

 protected:
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  void Snapshot(bool is_final = false);
  SolverParameter param_;
  int iter_;
  Net<Dtype>* net_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};


template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) {}

 protected:
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;
};


}  // namspace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
