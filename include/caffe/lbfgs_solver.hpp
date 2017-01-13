#ifndef CAFFE_LBFGS_SOLVER_HPP_
#define CAFFE_LBFGS_SOLVER_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using L-BFGS.
 * This implementation is based on minFunc, by Marc Schmidt [1]
 *
 * [1] https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
 */
template <typename Dtype>
class LBFGSSolver : public Solver<Dtype> {
 public:
  explicit LBFGSSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit LBFGSSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "LBFGS"; }

 protected:
  void PreSolve();
  virtual void ApplyUpdate();
  virtual void CollectGradients();
  virtual void UpdateHistory();
  virtual void ComputeInitialHessianApprox();
  virtual void ComputeDirection();
  virtual void ComputeStep();
  virtual void UpdateNet();
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  shared_ptr<Blob<Dtype> > gradients_prev_;
  shared_ptr<Blob<Dtype> > gradients_;
  shared_ptr<Blob<Dtype> > direction_;
  vector<shared_ptr<Blob<Dtype> > > s_history_, y_history_;
  vector<Dtype> rho_history_;
  int start_, end_, n_;
  Dtype h0_;
  Dtype step_;

  DISABLE_COPY_AND_ASSIGN(LBFGSSolver);
};

}  // namespace caffe

#endif  // CAFFE_LBFGS_SOLVER_HPP_
