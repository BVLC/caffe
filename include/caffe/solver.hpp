// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <string>
#include <vector>

namespace caffe {

template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  virtual ~Solver() {}
  inline shared_ptr<Net<Dtype> > net() { return net_; }

 protected:
  // PreSolve is run before any solving iteration starts, allowing one to
  // put up some scaffold.
  virtual void PreSolve() {}
  // Get the update value for the current iteration.
  virtual void ComputeUpdateValue() = 0;
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  // The test routine
  void TestAll();
  void Test(const int test_net_id = 0);
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

  DISABLE_COPY_AND_ASSIGN(Solver);
};


template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) {}
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) {}

 protected:
  virtual void PreSolve();
  Dtype GetLearningRate();
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;
  // update maintains update related data
  vector<shared_ptr<Blob<Dtype> > > update_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
     : SGDSolver<Dtype>(param) {}
  explicit AdaGradSolver(const string& param_file)
     : SGDSolver<Dtype>(param_file) {}
  virtual ~AdaGradSolver() {}

 protected:
    virtual void ComputeUpdateValue();
    virtual Dtype GetLearningRate();

    DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
     : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
     : SGDSolver<Dtype>(param_file) {}
  virtual ~NesterovSolver() {}

 protected:
    virtual void ComputeUpdateValue();

    DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

// Solver factory
template <typename Dtype>
Solver<Dtype>* GetSolver(const SolverParameter& param) {
  const SolverParameter_SolverType& type = param.solver_type();
  switch (type) {
  case SolverParameter_SolverType_SGD:
    return new SGDSolver<Dtype>(param);
  case SolverParameter_SolverType_ADAGRAD:
    return new AdaGradSolver<Dtype>(param);
  case SolverParameter_SolverType_NESTEROV:
    return new NesterovSolver<Dtype>(param);
  default:
    LOG(FATAL) << "Unknown Solver Type: " << type;
  }
  // just to suppress old compiler warnings.
  return (Solver<Dtype>*)(NULL);
}


}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
