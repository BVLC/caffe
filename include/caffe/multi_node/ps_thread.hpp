

#ifndef MULTI_NODE_PS_THREAD_H_
#define MULTI_NODE_PS_THREAD_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"


namespace caffe {
/**
* Threads work in parameter server nodes
* Get param gradients from convolution nodes
* and update parameters using SGD solver
*/
template <typename Dtype>
class PSThread : public WorkerThread<Dtype> {
 public:
  PSThread() {
    ps_solver_ = NULL;
    iter_ = 0;
    updated_layers_ = 0;
    staleness_ = NodeEnv::Instance()->get_staleness();
    num_workers_ = NodeEnv::Instance()->num_workers();
    max_iter_ = NodeEnv::Instance()->SolverParam().max_iter();
    test_node_ = -1;
  }

  virtual void Run();

 protected:
  int SendUpdates(int layer_id);

  int GetBatchSize(shared_ptr<Net<Dtype> > net) {
    const vector<Blob<Dtype>*>& out_blobs = net->output_blobs();
    CHECK_GT(out_blobs.size(), 0);

    return out_blobs[0]->shape(0);
  }

  /// update a layer
  void UpdateLayer(int layer_id);

  /// broadcast parameters of a layer
  void BroadcastLayer(int layer_id);

  /// register a new client to PS
  void RegisterNode(shared_ptr<Msg> m);

  /// BSP stype
  // return true if we are ready to broadcast the parameter to clients
  int UpdateParam(shared_ptr<Msg> m);

  void SendParam(shared_ptr<Net<Dtype> > net,
                 const vector<string>& layer_names,
                 int dst, int clock);

  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver,
                                      const SolverParameter& solver_param) {
    Caffe::set_root_solver(false);
    SGDSolver<Dtype> *ps_root = (SGDSolver<Dtype> *)root_solver;

    return new SGDSolver<Dtype>(solver_param, ps_root);
  }


 protected:
  SGDSolver<Dtype> *ps_solver_;

  int iter_;

  // map from client id to its array index
  map<int, int> client_idx_map_;

  // zmq id of each client
  vector<int> client_ids_;

  // store the messages from clients
  vector<vector<shared_ptr<Msg> > > client_msgs_;

  // number of gradients updated
  int updated_layers_;

  // number of learnalbe layers
  int num_learnable_layers_;

  // clock of each message
  vector<vector<int> > msg_clocks_;

  // allowed staleness of PS
  // -1 means doesn't check staleness
  int staleness_;

  // the number of conv. workers
  int num_workers_;

  // maximun iterations to be executed
  int max_iter_;

  // the id of test node
  int test_node_;
};

}  // end namespace caffe

#endif


