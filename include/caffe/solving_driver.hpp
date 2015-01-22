#ifndef CAFFE_OPTIMIZATION_SOLVING_DRIVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVING_DRIVER_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe {
template <typename Dtype>
class SolvingDriver {
 private:
  shared_ptr<Solver<Dtype> > solver_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;

  void TestAll();
  void Test(int test_net_id);
  void Snapshot() { solver_->Snapshot(); }

 public:
  explicit SolvingDriver(const SolverParameter& param);
  explicit SolvingDriver(const string& param_file);
  explicit SolvingDriver(shared_ptr<Solver<Dtype> > solver);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();
  // The main entry of the driver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);
  inline shared_ptr<Net<Dtype> > net() { return solver_->net(); }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() const { return solver_->iter(); }

  const SolverParameter& param() const { return solver_->param(); }

  DISABLE_COPY_AND_ASSIGN(SolvingDriver);
};
}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVING_DRIVER_HPP_
