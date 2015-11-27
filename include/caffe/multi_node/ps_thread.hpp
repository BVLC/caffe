

#ifndef MULTI_NODE_PS_THREAD_H_
#define MULTI_NODE_PS_THREAD_H_

#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"


namespace caffe {
/**
*Threads work in parameter server nodes
*Get param gradients from convolution nodes
* and update parameters using SGD solver
*/
template <typename Dtype>
class PSThread : public WorkerThread<Dtype>
{
public:
  PSThread() {
    ps_solver_ = NULL;
  }

  virtual void Run();

protected:
  void UpdateParam(shared_ptr<Msg> m);
  void SendParam(shared_ptr<Msg> m);

  virtual Solver<Dtype> *CreateSolver() {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *ps_root = (SGDSolver<Dtype> *)NodeEnv::Instance()->GetRootSolver();
    const SolverParameter& param = NodeEnv::Instance()->SolverParam();

    return new SGDSolver<Dtype>(param, ps_root);
  }

protected:
  SGDSolver<Dtype> *ps_solver_;
};

} //end caffe

#endif


