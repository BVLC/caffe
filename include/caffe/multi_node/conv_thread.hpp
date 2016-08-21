

#ifndef MULTI_NODE_CONV_THREAD_H_
#define MULTI_NODE_CONV_THREAD_H_

#include <boost/thread/barrier.hpp>

#include <map>
#include <string>
#include <vector>

#include "caffe/multi_node/node_env.hpp"
#include "caffe/multi_node/worker_thread.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {


template <typename Dtype>
class ConvThread : public WorkerThread<Dtype> {
 public:
  ConvThread() {
    param_solver_ = NULL;
    num_sub_solvers_ = NodeEnv::Instance()->num_sub_solvers();
  }

  virtual ~ConvThread() { }

  virtual void Run();

  static void InitFullSolver(SGDSolver<Dtype> *psolver) {
    CHECK(full_solver_ == NULL);

    full_solver_ = psolver;
  }

  static void InitBarrier(int num_threads) {
    CHECK(pconv_barrier_ == NULL);

    pconv_barrier_ = new boost::barrier(num_threads);
  }

 protected:
  void ConvForward();

  int NewConvId() {
    boost::mutex::scoped_lock lock(conv_id_mutex_);

    int orig_id = conv_id_;
    conv_id_++;

    return orig_id;
  }

  // get the solver according to message id
  WorkerSolver<Dtype> *PrepareBwdSolver(shared_ptr<Msg> m);

  // backward and sync, return true if backward is done
  bool ConvBackward(WorkerSolver<Dtype> *pconv, bool peek_next);

  bool SyncedBackward(WorkerSolver<Dtype> *prev_solver,
                      int prev_idx,
                      shared_ptr<Msg> m);

  // inform the parameter thread to update layer i
  void SendLayer(int layer_id);

  // do forward to a layer
  void ForwardLayer(shared_ptr<Net<Dtype> > conv_net, int layer_id);

  // do backward to a layer
  void BackwardLayer(shared_ptr<Net<Dtype> > conv_net, int layer_id);

 protected:
  typedef unordered_map<int64_t, shared_ptr<vector<shared_ptr<Msg> > > >
                                                                      MsgMap;
  // store and merge backward messages from multiple gateways
  MsgMap msg_id_to_buf_;

  // the pointer of solver which is used to store gradients
  WorkerSolver<Dtype> *param_solver_;

 protected:
  int num_sub_solvers_;

 protected:
  static int conv_id_;
  static boost::mutex conv_id_mutex_;

  static boost::barrier *pconv_barrier_;

  // the solver that has the full batch
  static SGDSolver<Dtype> *full_solver_;

DISABLE_COPY_AND_ASSIGN(ConvThread);
};


// for collect and update parameters in conv threads
template <typename Dtype>
class ConvParamThread : public WorkerThread<Dtype> {
 public:
  explicit ConvParamThread(const vector<int>& clocks) {
    ps_ids_ = NodeEnv::Instance()->ps_ids();
    ps_clocks_ = clocks;
    ps_updates_.resize(ps_ids_.size());

    // init PS maps
    for (int i = 0; i < ps_ids_.size(); i++) {
      ps_id_map_[ps_ids_[i]] = i;
      const vector<string>& ps_layers =
                            NodeEnv::Instance()->FindPSLayer(ps_ids_[i]);
      for (int j = 0; j < ps_layers.size(); j++) {
        const string& layer_name = ps_layers[j];
        layer_to_ps_id_[layer_name] = ps_ids_[i];
      }
      ps_updates_[i] = 0;
    }

    // init forward blobs and ids
    fwd_blobs_ = NodeEnv::Instance()->forward_blobs();
    fwd_ids_ = NodeEnv::Instance()->forward_ids();

    gateway_ids_ = NodeEnv::Instance()->gateway_ids();
    gateway_blobs_ = NodeEnv::Instance()->gateway_blobs();
    num_param_update_ = 0;
  }

  virtual ~ConvParamThread() { }

  // get a new version of parameters from PS
  void SyncWithPS();

  // process the forward messages sent by conv threads
  void ProcessForward(shared_ptr<Msg> m);

  // process the backward message from gateways
  void ProcessBackward(shared_ptr<Msg> m);

  // merge the activations as a single message
  void SendActivations();

  // sync one layer with PS
  void SyncLayer(int layer_id);

  virtual void Run();

 protected:
  virtual Solver<Dtype> *CreateSolver(const Solver<Dtype> *root_solver,
                                      const SolverParameter& solver_param) {
    return NULL;
  }

 protected:
  // update gradient
  int PutGradient(shared_ptr<Msg> m);

  // update parameter got from parameter server
  int UpdateParam(shared_ptr<Msg> m);

 protected:
  /// id of parameter servers
  vector<int> ps_ids_;

  // mapping PS ids for quick references
  map<int, int> ps_id_map_;

  // clock at each parameter server
  vector<int> ps_clocks_;

  // number of updates from a parameter server
  vector<int> ps_updates_;

  // number of layers which have learnable blobs
  int num_learnable_layers_;

  // map a layer to the id of PS
  map<string, int> layer_to_ps_id_;

  vector<int> fwd_ids_;
  vector<vector<string> > fwd_blobs_;

  vector<int> gateway_ids_;
  vector<vector<string> > gateway_blobs_;

  // record the number of updates from conv. workers
  vector<int> layer_updates_;

  vector<shared_ptr<Msg> > fwd_msgs_;

  // number of parameter update got from parameter server
  int num_param_update_;

  typedef unordered_map<int, shared_ptr<vector<shared_ptr<Msg> > > > ConvIdMap;

  // find the array of forward messages with a conv_id
  ConvIdMap conv_id_to_vec_;

  // find the array of backward messages with a conv_id
  ConvIdMap bwd_id_to_vec_;

DISABLE_COPY_AND_ASSIGN(ConvParamThread);
};

}  // end namespace caffe

#endif


