
#ifndef MULTI_NODE_FC_THREAD_H_
#define MULTI_NODE_FC_THREAD_H_

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
  virtual Solver<Dtype> *CreateSolver() {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *fc_root = (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const SolverParameter& param = NodeEnv::Instance()->SolverParam();

    return new SGDSolver<Dtype>(param, fc_root);
  }


protected:
  shared_ptr<Msg> FcForward(shared_ptr<Msg> m);
  void FcBackward(shared_ptr<Msg> m, vector<shared_ptr<Msg> >& replies);

DISABLE_COPY_AND_ASSIGN(FcThread);
};


//the last part of FC layers
template <typename Dtype>
class FcEndThread : public FcThread<Dtype>
{
public:
  FcEndThread() { }
  virtual ~FcEndThread() { }
  virtual void Run();

protected:
  static boost::atomic_int iter_;

DISABLE_COPY_AND_ASSIGN(FcEndThread);
};


//for updating the FC parameters
template <typename Dtype>
class FcParamThread : public WorkerThread<Dtype>
{
public:
  FcParamThread() { }

  virtual void Run();

protected:
  void UpdateParam(shared_ptr<Msg> m);
  
  virtual Solver<Dtype> *CreateSolver() {
    return NULL;
  }

DISABLE_COPY_AND_ASSIGN(FcParamThread);
};

} //end caffe

#endif


