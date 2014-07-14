// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>
#include "caffe/iter_callback.hpp"
#include "caffe/default_solver_actions.hpp"

namespace caffe {

template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(typename IterCallback<Dtype>::Type iter_callback);
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }

 protected:
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue(const IterActions<Dtype>& actions) = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  TrainingStats<Dtype> TestAll();
  TestResult<Dtype> Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;

  SolverParameter param_;
  int iter_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  typename IterCallback<Dtype>::Type callback_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

template <typename Dtype>
class SGDSolverEx : public Solver<Dtype> {
 public:
  explicit SGDSolverEx(const SolverParameter& param )
      : Solver<Dtype>(param) {}
  explicit SGDSolverEx(const string& param_file )
      : Solver<Dtype>(param_file) {}

 protected:
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue(const IterActions<Dtype>& actions);
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;

  DISABLE_COPY_AND_ASSIGN(SGDSolverEx);
};

template <typename Dtype>
class SGDSolver : public SGDSolverEx<Dtype> {
 public:
    explicit SGDSolver(const SolverParameter& param)
      : SGDSolverEx<Dtype>(param),
        handler_(param) {}
    explicit SGDSolver(const string& param_file)
      : SGDSolverEx<Dtype>(param_file),
        handler_() {
        handler_ = DefaultSolverActions<Dtype>(
                    this->param_);
    }
    void Solve(const char* resume_file = 0) {
        bool is_null = (resume_file == 0);
        std::string file_string;
        if (!is_null) {
            file_string = std::string(resume_file);
        }
        handler_.SetResumeFile(file_string);
        Solver<Dtype>::Solve(handler_);
    }
 protected:
    DefaultSolverActions<Dtype> handler_;
    DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
