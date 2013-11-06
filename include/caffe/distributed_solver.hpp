// Copyright Yangqing Jia 2013
// This implements the distributed solver in two classes: the parameter server
// that holds the parameters and the actual solver that does computation.

// Copyright Yangqing Jia 2013

#ifndef CAFFE_OPTIMIZATION_DISTRIBUTED_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_DISTRIBUTED_SOLVER_HPP_

#include <vector>

#include "caffe/solver.hpp"


namespace caffe {

template <typename Dtype>
class DistributedSolverParamServer : public Solver<Dtype> {
 public:
  DistributedSolverParamServer(const SolverParameter& param)
      : Solver<Dtype>(param) {}
  virtual void Solve(const char* resume_file = NULL);

 protected:
  // The distributed solver does not do solving itself.
  virtual void ComputeUpdateValue() {}
  // The distributed solver has nothing to snapshot.
  virtual void SnapshotSolverState(SolverState* state) {}
  virtual void RestoreSolverState(const SolverState& state) {}
  // The function that implements the actual communication.
  void ReceiveAndSend();

  int next_snapshot_;
};


template <typename Dtype>
class DistributedSolverParamClient : public SGDSolver<Dtype> {
 public:
  DistributedSolverParamClient(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {
    CHECK_GT(param.communication_interval(), 0);
    next_send_iter_ = param.communication_interval();
  }
  virtual void Solve(const char* resume_file = NULL);
 protected:
  virtual void PreSolve() { SGDSolver<Dtype>::PreSolve(); }
  virtual void ComputeUpdateValue();
  void SendAndReceive(bool just_receive = false);

  int next_send_iter_;
  int next_display_;
};


}  // namespace caffe

#endif  // CAFFE_OPTIMIZATION_DISTRIBUTED_SOLVER_HPP_
