
#ifndef MULTI_NODE_FC_THREAD_H_
#define MULTI_NODE_FC_THREAD_H_

#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class FcThread : public WorkerThread<Dtype>
{
public:
  FcThread() { }
  virtual ~FcThread() { }

  virtual void Run();

protected:

protected:
  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver, const SolverParameter& solver_param) {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *fc_root = (SGDSolver<Dtype> *)root_solver;

    return new SGDSolver<Dtype>(solver_param, fc_root);
  }
  
  /// for fc layers we don't do parameter sharing
  virtual Solver<Dtype> *NewSolver(Solver<Dtype> *proot, const SolverParameter& solver_param) {
    boost::mutex::scoped_lock lock(this->new_solver_mutex_);

    Solver<Dtype> *new_solver = CreateSolver(proot, solver_param);
    if (new_solver == NULL) {
      return new_solver;
    }

    this->new_solver_cnt_++;
    LOG(INFO) << "created " << this->new_solver_cnt_ << " solvers";
    return new_solver;
  }


protected:
  shared_ptr<Msg> FcForward(shared_ptr<Msg> m);
  void FcBackward(shared_ptr<Msg> m, vector<shared_ptr<Msg> >& replies, bool copy_diff);

DISABLE_COPY_AND_ASSIGN(FcThread);
};


//the last part of FC layers
template <typename Dtype>
class FcLossThread : public FcThread<Dtype>
{
public:
  FcLossThread() { }
  virtual ~FcLossThread() { }
  virtual void Run();

protected:
  static boost::atomic_int iter_;

DISABLE_COPY_AND_ASSIGN(FcLossThread);
};


//for updating the FC parameters
template <typename Dtype>
class FcParamThread : public WorkerThread<Dtype>
{
public:
  FcParamThread() {
    train_iter_ = 0;
    test_node_id_ = -1;
    sub_updates_ = 0;
  }

  virtual void Run();

protected:
  void SendParam(shared_ptr<Msg> m);

  void UpdateParam(shared_ptr<Msg> m);
  
  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver, const SolverParameter& solver_param) {
    return NULL;
  }
  
  void SendNotify();

protected:
  int train_iter_;
  int test_node_id_;

  // number of updates from sub solvers
  int sub_updates_;

DISABLE_COPY_AND_ASSIGN(FcParamThread);
};

} //end caffe

#endif


