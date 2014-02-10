// Copyright Yangqing Jia 2013

#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <vector>

namespace caffe {

template <typename Dtype>
class TerminationCriterion;
  
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param);
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  virtual ~Solver() {}
  inline Net<Dtype>* net() { return net_.get(); }

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
  Dtype Test();
  virtual void SnapshotSolverState(SolverState* state) = 0;
  // The Restore function implements how one should restore the solver to a
  // previously snapshotted state. You should implement the RestoreSolverState()
  // function that restores the state from a SolverState protocol buffer.
  void Restore(const char* resume_file);
  virtual void RestoreSolverState(const SolverState& state) = 0;

  SolverParameter param_;
  int iter_;
  shared_ptr<Net<Dtype> > net_;
  shared_ptr<Net<Dtype> > test_net_;
  shared_ptr<TerminationCriterion<Dtype > > termination_criterion_;

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
  virtual void SnapshotSolverState(SolverState * state);
  virtual void RestoreSolverState(const SolverState& state);
  // history maintains the historical momentum data.
  vector<shared_ptr<Blob<Dtype> > > history_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class TerminationCriterion {
public:
  TerminationCriterion() : criterion_met_(false) {};
  
  virtual bool IsCriterionMet() {return criterion_met_;};

  virtual void NotifyTestAccuracy(Dtype test_accuracy) = 0;

  virtual void NotifyIteration(int iteration) = 0;
protected:
  bool criterion_met_;
};
  
template <typename Dtype>
class MaxIterTerminationCriterion : public TerminationCriterion<Dtype> {
public:
  MaxIterTerminationCriterion(int max_iter) : max_iter_(max_iter) {};

  virtual void NotifyTestAccuracy(Dtype test_accuracy) {};
  
  virtual void NotifyIteration(int iteration);
private:
  int max_iter_;
};
  
template <typename Dtype>
class TestAccuracyTerminationCriterion : public TerminationCriterion<Dtype> {
public:
  TestAccuracyTerminationCriterion(int test_accuracy_stop_countdown) :
    test_accuracy_stop_countdown_(test_accuracy_stop_countdown),
    count_down_(test_accuracy_stop_countdown),
    best_accuracy_(0.) {};
  
  virtual void NotifyTestAccuracy(Dtype test_accuracy);
  
  virtual void NotifyIteration(int iteration) {};
  
private:
  const int test_accuracy_stop_countdown_;
  Dtype best_accuracy_;
  int count_down_;
};
  

}  // namspace caffe

#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
