

#ifndef MULTI_NODE_CONV_THREAD_H_
#define MULTI_NODE_CONV_THREAD_H_

#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/multi_node/node_env.hpp"

namespace caffe {

template <typename Dtype>
class ConvThread : public WorkerThread<Dtype>
{

public:
  ConvThread(const vector<int>& clocks) {
    ps_ids_ = NodeEnv::Instance()->ps_ids();
    ps_clocks_ = clocks;

    // init PS id map
    for (int i = 0; i < ps_ids_.size(); i++) {
      ps_id_map_[ps_ids_[i]] = i;
    }
  }

  virtual ~ConvThread() {

  }

  virtual void Run();

protected:
  shared_ptr<Msg> ConvForward();
  int ConvBackward(shared_ptr<Msg> m);
  
  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver, const SolverParameter& solver_param) {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *conv_root = (SGDSolver<Dtype> *)root_solver;

    return new SGDSolver<Dtype>(solver_param, conv_root);
  }

  int64_t NewConvId() {
    boost::mutex::scoped_lock lock(conv_id_mutex_);
     
    int64_t orig_id = conv_id_;
    conv_id_++;

    return orig_id;
  }
  
  // get a new version of parameters from PS
  void SyncWithPS();

protected:
  static int64_t conv_id_;
  static boost::mutex conv_id_mutex_;
  
  /// id of parameter servers
  vector<int> ps_ids_;

  // mapping PS ids for quick references
  map<int, int> ps_id_map_;

  // clock at each parameter server
  vector<int> ps_clocks_;

DISABLE_COPY_AND_ASSIGN(ConvThread);
};


//for collect and update parameters in conv threads
template <typename Dtype>
class ConvParamThread : public WorkerThread<Dtype>
{

public:
  ConvParamThread() { 
    ps_ids_ = NodeEnv::Instance()->ps_ids(); 
  }
  
  virtual ~ConvParamThread() {

  }

  virtual void Run();

protected:
  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver, const SolverParameter& solver_param) {
    return NULL;
  }

protected:
  //update gradient
  int PutGradient(shared_ptr<Msg> m);
  //update parameter got from parameter server
  int UpdateParam(shared_ptr<Msg> m);

protected:
  /// id of parameter servers
  vector<int> ps_ids_;

DISABLE_COPY_AND_ASSIGN(ConvParamThread);
};

} //end caffe

#endif


