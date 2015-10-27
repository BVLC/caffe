

#ifndef MULTI_NODE_CONV_THREAD_H_
#define MULTI_NODE_CONV_THREAD_H_

#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class ConvThread : public WorkerThread<Dtype>
{

public:
  ConvThread() { 
  
  }

  virtual ~ConvThread() {

  }

  virtual void Run();

protected:
  shared_ptr<Msg> ConvForward();
  int ConvBackward(shared_ptr<Msg> m);
  
  virtual Solver<Dtype> *CreateSolver() {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *conv_root = (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const SolverParameter& param = NodeEnv::Instance()->SolverParam();

    return new WorkerSolver<Dtype>(param, conv_root);
  }

  int64_t NewConvId() {
    boost::mutex::scoped_lock lock(conv_id_mutex_);
     
    int64_t orig_id = conv_id_;
    conv_id_++;

    return orig_id;
  }

protected:
  static int64_t conv_id_;
  static boost::mutex conv_id_mutex_;

DISABLE_COPY_AND_ASSIGN(ConvThread);
};


//for collect and update parameters in conv threads
template <typename Dtype>
class ConvParamThread : public WorkerThread<Dtype>
{

public:
  ConvParamThread() { 
  
  }
  
  virtual ~ConvParamThread() {

  }

  virtual void Run();

protected:
  virtual Solver<Dtype> *CreateSolver() {
    return NULL;
  }

protected:
  //update gradient
  int PutGradient(shared_ptr<Msg> m);
  //update parameter got from parameter server
  int UpdateParam(shared_ptr<Msg> m);

DISABLE_COPY_AND_ASSIGN(ConvParamThread);
};

} //end caffe

#endif


