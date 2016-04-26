#include <glog/logging.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "boost/make_shared.hpp"
#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/guaranteed_comm.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"
#include "caffe/multinode/SynchronousParamClient.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

namespace {

struct TerminatedHandler {
  virtual bool terminated() = 0;
};

struct LayerState {
  enum Enum {
    calculating,
    updating
  };

  Enum state;
  uint32_t version;
  boost::mutex mtx;
  boost::condition_variable cond;

  LayerState() : state(calculating), version(0u) {
  }
  LayerState(const LayerState& other)
    : state(other.state)
    , version(other.version) {
  }

  void move_to(Enum next_state) {
    {
      boost::mutex::scoped_lock lock(mtx);
      state = next_state;
    }
    cond.notify_all();
  }

  int wait_till(TerminatedHandler* handler, Enum till_state) {
    boost::mutex::scoped_lock lock(mtx);
    int ret = 0;
    while (state != till_state) {
      boost::system_time timeout
        = boost::get_system_time() + boost::posix_time::milliseconds(100);
      cond.timed_wait(lock, timeout);
      ++ret;
      if (handler->terminated()) {
        std::terminate();
      }
    }
    return ret;
  }

  void set_version(uint32_t new_version) {
    boost::mutex::scoped_lock lock(mtx);
    version = new_version;
  }

  uint32_t get_version() {
    boost::mutex::scoped_lock lock(mtx);
    return version;
  }
};

}  // namespace

template<typename Dtype>
struct SynchronousParamSyncingImpl
    : TerminatedHandler
    , BlobSyncInfo::Handler
    , InternalThread {
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<internode::Daemon> comm;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<internode::Waypoint> waypoint;

  shared_ptr<BlobAccessor<Dtype> > blob_accessor;
  shared_ptr<BlobConstInfo> const_info;
  shared_ptr<BlobSyncInfo> sync_info;
  shared_ptr<BlobKeyChain<Dtype> > keychain;
  shared_ptr<BlobComms<Dtype> > comms;

  std::vector<LayerState> layers;
  LayerState init;

  boost::mutex mtx;
  bool terminated_;

  SynchronousParamSyncingImpl(shared_ptr<Solver<Dtype> > solver,
                              string address,
                              int num_of_threads)
    : solver(solver)
    , comm(internode::create_communication_daemon())
    , codec(BlobCodec<Dtype>::create_codec(
        solver->param().multinode_param(), true))
    , waypoint(internode::configure_client(comm, address, codec->packet_size()))
    , blob_accessor(BlobInfoFactory<Dtype>::create_blob_accessor(solver))
    , const_info(BlobInfoFactory<Dtype>::create_const_info(
        solver, codec->max_elements_per_part()))
    , sync_info(BlobInfoFactory<Dtype>::create_sync_info(const_info))
    , keychain(BlobKeyChain<Dtype>::create_empty(const_info->layers()))
    , comms(
        BlobComms<Dtype>::create(blob_accessor,
          const_info, sync_info, waypoint, codec, keychain,
          typename BlobComms<Dtype>::Settings(
            BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
          num_of_threads))
    , layers(const_info->layers())
    , terminated_(false) {
    init.move_to(LayerState::updating);
    sync_info->register_synced_handler(this);
    waypoint->register_receive_handler(comms.get());
    comms->send_iter_size(solver->param().iter_size());

    internode::create_timer(
      comm,
      500000,
      boost::bind(&SynchronousParamSyncingImpl::tick, this),
      true);
  }

  void tick() {
  }

  // called from comm thread
  virtual bool terminated() {
    boost::mutex::scoped_lock lock(mtx);
    terminated_ =
      terminated_ || (solver->GetRequestedAction() != SolverAction::NONE);
    return terminated_;
  }

  virtual void synced(int layer_id, int blob_id, int part, uint32_t version) {
  }

  virtual void synced(int layer_id, uint32_t version) {
    VLOG(2) << "layer " << layer_id
               << " is in synced with version " << version;
    comms->cancel(layer_id, version - 1);
    layers.at(layer_id).set_version(version);
    layers.at(layer_id).move_to(LayerState::calculating);
  }

  virtual void synced(uint32_t version) {
    VLOG(2) << "net is synced with version: " << version;
    init.move_to(LayerState::calculating);
  }

  virtual void InternalThreadEntry() {
    while (!terminated()) {
      internode::run_one(comm);
    }
  }

  // called from solver thread
  void calculate(int layer_id) {
    if (!const_info->needs_syncing(layer_id)) return;
    VLOG(3)  << "waiting for layer " << layer_id;
    int waited = layers.at(layer_id).wait_till(this, LayerState::calculating);

    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    for (int j = 0; j < param_ids.size(); ++j) {
      solver->net()->ClearParamDiffs(param_ids[j]);
    }

    if (waited > 1) {
      VLOG(1) << "waited on calculating layer " << layer_id
              << " " << (waited / 10.0) << "seconds";
    }
  }

  void update(int layer_id) {
    if (!const_info->needs_syncing(layer_id)) return;
    VLOG(3) << "backward ready for layer " << layer_id;
    layers.at(layer_id).move_to(LayerState::updating);
    comms->push(layer_id, layers.at(layer_id).get_version());
  }
};

template<typename Dtype>
SynchronousParamClient<Dtype>::SynchronousParamClient(
        boost::shared_ptr<Solver<Dtype> > solver,
        string param_server_addr,
        int num_of_threads)
    : solver_(boost::make_shared<MultiSolver<Dtype> >(
        solver, (Caffe::mode() != Caffe::GPU)))
    , sync(new SynchronousParamSyncingImpl<Dtype>(
        solver, param_server_addr, num_of_threads)) {
  solver->param().set_disabled_update(true);
}

template<typename Dtype>
SynchronousParamClient<Dtype>::~SynchronousParamClient() {
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_start() {
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_start(int layer_id) {
  sync->calculate(layer_id);
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_forward_finished(int layer_id) {
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_gradients_ready() {
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_backward_start(int layer_id) {
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_gradients_ready(int layer_id) {
  sync->update(layer_id);
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::run() {
  sync->StartInternalThread();
  LOG(INFO) << "waiting for layers to synchronize";
  int time = sync->init.wait_till(sync.get(), LayerState::calculating);
  VLOG(1) << "layers are synchronized: waited "
          << (time / 10.0) << " seconds";
  solver_->add_callback(this);
  solver_->Solve();
}

INSTANTIATE_CLASS(SynchronousParamClient);

}  // namespace caffe

