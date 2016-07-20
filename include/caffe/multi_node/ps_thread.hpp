

#ifndef MULTI_NODE_PS_THREAD_H_
#define MULTI_NODE_PS_THREAD_H_

#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/multi_node/node_env.hpp"

namespace caffe {
/**
* Threads work in parameter server nodes
* Get param gradients from convolution nodes
* and update parameters using SGD solver
*/
template <typename Dtype>
class PSThread : public WorkerThread<Dtype>
{
public:
  PSThread() {
    ps_solver_ = NULL;
    pseudo_solver_ = NULL;
    iter_ = 0;
    staleness_ = NodeEnv::Instance()->get_staleness();
    num_workers_ = NodeEnv::Instance()->num_workers();
  }

  virtual void Run();

protected:
  void SendUpdates();
 
  /// return the min clock in the clients
  int MinClock();

  int GetBatchSize(shared_ptr<Net<Dtype> > net) {
    const vector<Blob<Dtype>*>& out_blobs = net->output_blobs();
    CHECK_GT(out_blobs.size(), 0);

    return out_blobs[0]->shape(0);
  }

  /// register a new client to PS
  void RegisterNode(shared_ptr<Msg> m);

  /// BSP stype, return true if we are ready to broadcast the parameter to clients
  void UpdateParam(shared_ptr<Msg> m);

  void SendParam(shared_ptr<Net<Dtype> > net, int dst, int clock);

  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver, const SolverParameter& solver_param) {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *ps_root = (SGDSolver<Dtype> *)root_solver;

    return new SGDSolver<Dtype>(solver_param, ps_root);
  }

protected:
  SGDSolver<Dtype> *ps_solver_;
  SGDSolver<Dtype> *pseudo_solver_;
  int iter_;

  // map from client id to its array index
  map<int, int> client_idx_map_;

  // 
  vector<int> client_ids_;

  // the clock at each client
  vector<int> client_clocks_;

  // whether the client needs update from PS
  vector<bool> client_need_update_;
  
  // store the messages from clients
  vector<shared_ptr<Msg> > client_msgs_;

  // use to store the square results
  vector<shared_ptr<Blob<Dtype> > > sqr_blobs_;
  
  // blobs filled with 1.0, used to get sum by dot
  vector<shared_ptr<Blob<Dtype> > > eye_blobs_;

  // allowed staleness of PS
  // -1 means doesn't check staleness
  int staleness_;
  
  // the number of conv. workers
  int num_workers_;
};

} //end caffe

#endif


