#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
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
#include "caffe/multinode/SynchronousParamClient.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

namespace {

struct TerminatedHandler {
  virtual bool terminated() = 0;
};

class LayerState {
  enum Enum {
    waiting_for_update,
    calculating,
    sending_gradients
  };

  Enum state;
  int iter;
  int iter_size;
  boost::mutex mtx;
  boost::condition_variable cond;

  boost::mutex gradient_mtx;
  boost::mutex param_mtx;

 public:
  LayerState() : state(waiting_for_update), iter(0), iter_size(0) {
  }

  LayerState(const LayerState& other)
    : state(waiting_for_update), iter(0), iter_size(0) {
  }

  void move_to_calculating(int iter_id) {
    {
      boost::mutex::scoped_lock lock(mtx);
      state = calculating;
      iter = iter_id;
    }
    cond.notify_all();
  }
  void move_to_calculating() {
    move_to_calculating(get_iter());
  }
  void move_to_waiting_for_update() {
    {
      boost::mutex::scoped_lock lock(mtx);
      state = waiting_for_update;
    }
    cond.notify_all();
  }
  void move_to_sending() {
    {
      boost::mutex::scoped_lock lock(mtx);
      state = sending_gradients;
    }
    cond.notify_all();
  }

  int get_iter() {
    boost::mutex::scoped_lock lock(mtx);
    return iter;
  }

  void wait_till_sending(TerminatedHandler* handler) {
    boost::mutex::scoped_lock lock(mtx);
    while (state != sending_gradients) {
      boost::system_time timeout
        = boost::get_system_time() + boost::posix_time::milliseconds(100);
      cond.timed_wait(lock, timeout);
      if (handler->terminated()) {
        std::terminate();
      }
    }
  }

  void wait_till_updating(TerminatedHandler* handler) {
    boost::mutex::scoped_lock lock(mtx);
    while (state != waiting_for_update) {
      boost::system_time timeout
        = boost::get_system_time() + boost::posix_time::milliseconds(100);
      cond.timed_wait(lock, timeout);
      if (handler->terminated()) {
        std::terminate();
      }
    }
  }

  int wait_till_calculating(TerminatedHandler* handler) {
    boost::mutex::scoped_lock lock(mtx);
    int ret = 0;
    while (state != calculating) {
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
};

struct CalculationState {
  std::vector<LayerState> layers;
  std::vector<bool> needs_syncing;

  template <typename Net>
  explicit CalculationState(const Net& net)
    : layers(net.layers().size())
    , needs_syncing(net.layers().size(), false) {
    for (int i = 0; i < net.layers().size(); ++i) {
      needs_syncing[i] = !net.get_layer_learnable_param_ids(i).empty();
    }
  }
};

template <typename Dtype>
class ReceiveThread : public InternalThread,
                      public internode::Waypoint::Handler {
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<Net<Dtype> > net;
  boost::shared_ptr<internode::Daemon> comm;
  boost::shared_ptr<internode::Waypoint> waypoint;
  shared_ptr<CalculationState> state;
  shared_ptr<BlobCodec<Dtype> > codec;
  std::vector<std::vector<bool> > received_blobs;
  TerminatedHandler* terminate_handler;
 public:
  ReceiveThread(shared_ptr<Solver<Dtype> > solver,
                boost::shared_ptr<internode::Daemon> comm,
                boost::shared_ptr<internode::Waypoint> param_waypoint,
                shared_ptr<CalculationState> state,
                shared_ptr<BlobCodec<Dtype> > codec,
                TerminatedHandler* terminate_handler)
    : solver(solver)
    , net(solver->net())
    , comm(comm)
    , waypoint(param_waypoint)
    , state(state)
    , codec(codec)
    , received_blobs(net->layers().size())
    , terminate_handler(terminate_handler) {
    waypoint->register_receive_handler(this);
    for (int i = 0; i < net->layers().size(); ++i) {
      received_blobs[i].resize(net->layers()[i]->blobs().size(), false);
    }
  }

  void reset_received_blobs(int layer_id) {
    for (int j = 0; j < received_blobs[layer_id].size(); ++j) {
      received_blobs[layer_id][j] = false;
    }
  }

  bool received_all_blobs(int layer_id) {
    for (int j = 0; j < received_blobs[layer_id].size(); ++j) {
      if (!received_blobs[layer_id][j]) return false;
    }
    return true;
  }

  virtual void received(char* data, size_t size, internode::RemoteId) {
    BlobUpdate msg;
    if (!deserialize(data, size, &msg)) return;

    Blob<Dtype>* blob =
      net->layers().at(msg.layer_id())->blobs().at(msg.blob_id()).get();

    state->layers.at(msg.layer_id()).wait_till_updating(terminate_handler);

    codec->decode(msg, blob, BlobCodec<Dtype>::PARAMS, 1.0, 0.0);
    if (Caffe::mode() == Caffe::GPU) {
      blob->gpu_data();
    }
    VLOG(2) << "received update for blob: " << msg.blob_id()
            << " of layer " << msg.layer_id()
            << " with version " << msg.version()
            << " data size: " << msg.data().size();

    received_blobs[msg.layer_id()][msg.blob_id()] = true;
    if (received_all_blobs(msg.layer_id())) {
      reset_received_blobs(msg.layer_id());
      state->layers.at(msg.layer_id()).move_to_calculating(msg.version());
    }
  }

  virtual void InternalThreadEntry() {
    while (!terminate_handler->terminated()) {
      internode::poll_one(comm);
    }
  }
};

template <typename Dtype>
class SendThread : public InternalThread {
  const int layer_id;
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<internode::Waypoint> waypoint;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<CalculationState> state;
  vector<int> param_ids;
  TerminatedHandler* terminate_handler;

 public:
  SendThread(int layer_id,
             shared_ptr<Solver<Dtype> > solver,
             shared_ptr<internode::Waypoint> waypoint,
             shared_ptr<BlobCodec<Dtype> > codec,
             shared_ptr<CalculationState> state,
             TerminatedHandler* terminate_handler)
    : layer_id(layer_id)
    , solver(solver)
    , waypoint(waypoint)
    , codec(codec)
    , state(state)
    , param_ids(solver->net()->get_layer_learnable_param_ids(layer_id))
    , terminate_handler(terminate_handler) {
  }

  virtual void InternalThreadEntry() {
    if (solver->net()->get_layer_learnable_param_ids(layer_id).empty()) {
      return;
    }

    while (!terminate_handler->terminated()) {
      state->layers.at(layer_id).wait_till_sending(terminate_handler);

      Layer<Dtype>& layer = *solver->net()->layers()[layer_id];
      for (int j = 0; j < layer.blobs().size(); ++j) {
        BlobUpdate msg;
        msg.set_version(state->layers.at(layer_id).get_iter());
        msg.set_iters(solver->param().iter_size());
        msg.set_layer_id(layer_id);
        msg.set_blob_id(j);

        Blob<Dtype>* blob = layer.blobs()[j].get();
        codec->encode(&msg, blob, BlobCodec<Dtype>::GRADS, 0);
        string str = serialize(msg);

        VLOG(3) << "[sending thread] sending blob "
                << j << " of layer " << layer_id;
        waypoint->send(str.c_str(), str.size());
        VLOG(2) << "[sending thread] sent blob " << j << " from layer "
                << layer_id << " of version: " << msg.version();
      }

      for (int j = 0; j < param_ids.size(); ++j) {
        solver->net()->ClearParamDiffs(param_ids[j]);
      }
      if (Caffe::mode() == Caffe::GPU) {
        for (int j = 0; j < layer.blobs().size(); ++j) {
          layer.blobs()[j].get()->gpu_diff();
        }
      }
      state->layers.at(layer_id).move_to_waiting_for_update();
    }
  }
};

}  // namespace

template<typename Dtype>
struct SynchronousParamSyncingImpl : TerminatedHandler {
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<Net<Dtype> > net;
  shared_ptr<internode::Daemon> comm;
  shared_ptr<internode::Waypoint> waypoint;
  shared_ptr<CalculationState> state;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<ReceiveThread<Dtype> > receiving;
  vector<shared_ptr<SendThread<Dtype> > > sending;

  boost::mutex mtx;
  bool terminated_;

  SynchronousParamSyncingImpl(shared_ptr<Solver<Dtype> > solver,
                              string address)
    : solver(solver)
    , net(solver->net())
    , comm(internode::create_communication_daemon())
    , waypoint(internode::configure_client(comm, address))
    , state(new CalculationState(*net))
    , codec(BlobCodec<Dtype>::create_codec(solver->param().multinode_param()))
    , receiving(
        new ReceiveThread<Dtype>(solver, comm, waypoint, state, codec, this))
    , sending(state->layers.size())
    , terminated_(false) {
    for (int i = 0; i < sending.size(); ++i) {
      sending[i].reset(
        new SendThread<Dtype>(i, solver, waypoint, codec, state, this));
    }
  }

  bool terminated() {
    boost::mutex::scoped_lock lock(mtx);
    terminated_ =
      terminated_ || (solver->GetRequestedAction() != SolverAction::NONE);
    return terminated_;
  }

  void start() {
    for (int i = 0; i < sending.size(); ++i) {
      sending[i]->StartInternalThread();
    }
    receiving->StartInternalThread();
  }
};

template<typename Dtype>
SynchronousParamClient<Dtype>::SynchronousParamClient(
        boost::shared_ptr<Solver<Dtype> > solver,
        string param_server_addr)
    : solver_(boost::make_shared<MultiSolver<Dtype> >(solver))
    , sync(new SynchronousParamSyncingImpl<Dtype>(solver, param_server_addr))
    , iters(0) {
}

template<typename Dtype>
SynchronousParamClient<Dtype>::~SynchronousParamClient() {
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_start() {
  VLOG(3) << "[solver thread] starting forward/backward";
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::on_start(int layer_id) {
  if (!sync->state->needs_syncing.at(layer_id)) return;

  VLOG(4)  << "[solver thread] waiting for layer " << layer_id;
  int waited = sync->state->layers.at(layer_id)
    .wait_till_calculating(sync.get());

  int layer_iter = sync->state->layers.at(layer_id).get_iter();

  if (waited > 1) {
    VLOG(1) << "[solver thread] waited on calculating layer " << layer_id
      << " " << (waited / 10.0) << "seconds"
      << ", the iteration is " << layer_iter;
  }
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
  if (!sync->state->needs_syncing.at(layer_id)) return;
  sync->state->layers.at(layer_id).move_to_sending();
  VLOG(4) << "[solver thread] backward ready for layer " << layer_id;
}

template<typename Dtype>
void SynchronousParamClient<Dtype>::run() {
  sync->start();
  solver_->add_callback(this);
  solver_->Solve();
}

INSTANTIATE_CLASS(SynchronousParamClient);

}  // namespace caffe

