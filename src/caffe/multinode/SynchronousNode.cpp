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
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"
#include "caffe/multinode/SynchronousNode.hpp"
#include "caffe/MultiSolver.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

namespace {

using internode::Waypoint;
using internode::TreeWaypoint;
using internode::RemoteId;

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

  string str_state() {
    boost::mutex::scoped_lock lock(mtx);
    if (state == updating) return "updating";
    return "calculating";
  }
};

#define MLOG(lvl) VLOG(lvl) << "[proc " \
                            << TreeWaypoint::get_instance()->id() << "] "

template <bool IsUp>
class UpDownWaypoint : public Waypoint {
  TreeWaypoint* waypoint;
  RemoteId id_;

 public:
  explicit UpDownWaypoint(RemoteId id)
    : waypoint(TreeWaypoint::get_instance())
    , id_(id) {
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    if (IsUp) {
      waypoint->async_send_to_parent(buffer, size, callback);
    } else {
      waypoint->async_send_to_children(buffer, size, callback);
    }
  }

  virtual void register_receive_handler(Handler* handler) {
    throw std::runtime_error("unexpected call");
  }

  virtual RemoteId id() const {
    return id_;
  }

  virtual string address() const {
    return boost::lexical_cast<string>(waypoint->id());
  }
  virtual bool guaranteed_comm() const {
    return true;
  }

  virtual size_t max_packet_size() const {
    return UINT_MAX;
  }
};

template <typename Dtype>
class SynchronousSync : public InternalThread
                      , public TerminatedHandler
                      , public TreeWaypoint::Handler {
  boost::mutex mtx;
  bool terminated_;
  TreeWaypoint* waypoint;
  boost::shared_ptr<Solver<Dtype> > solver;
  const int local_iters;
  int total_iters;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<Waypoint> up_waypoint;
  shared_ptr<Waypoint> down_waypoint;
  shared_ptr<BlobInfo<Dtype> > up_info;
  shared_ptr<BlobInfo<Dtype> > down_info;
 public:
  shared_ptr<BlobKeyChain<Dtype> > keychain;
 private:
  shared_ptr<BlobComms<Dtype> > comms_up;
  shared_ptr<BlobComms<Dtype> > comms_down;
  std::vector<LayerState> layers;

  struct ParentSyncHandler : BlobSyncInfo::Handler {
    SynchronousSync<Dtype>* instance;
    explicit ParentSyncHandler(SynchronousSync<Dtype>* instance)
      : instance(instance) {
    }

    virtual void synced(int layer_id, uint32_t version) {
      instance->synced_parameters(layer_id, version);
    }

    virtual void synced(uint32_t version) {
      instance->synced_parameters(version);
    }

    virtual void synced(int layer_id, int blob_id, int part, uint32_t version) {
      instance->synced_parameters(layer_id, blob_id, part, version);
    }
  } parent_sync;

  struct ChildrenSyncHandler : BlobSyncInfo::Handler {
    SynchronousSync<Dtype>* instance;
    explicit ChildrenSyncHandler(SynchronousSync<Dtype>* instance)
      : instance(instance) {
    }

    virtual void synced(int layer_id, uint32_t version) {
      instance->synced_gradients(layer_id, version);
    }

    virtual void synced(uint32_t version) {
      instance->synced_gradients(version);
    }

    virtual void synced(int layer_id, int blob_id, int part, uint32_t version) {
      instance->synced_gradients(layer_id, blob_id, part, version);
    }
  } children_sync;

  virtual bool terminated() {
    #ifdef USE_MPI
      return false;
    #else
      boost::mutex::scoped_lock lock(mtx);
      terminated_ =
        terminated_ || (solver->GetRequestedAction() != SolverAction::NONE);
      return terminated_;
    #endif
  }

  virtual void InternalThreadEntry() {
    while (!terminated()) {
      internode::poll_one(waypoint->get_daemon());
    }
  }

  void tick() const {
  }

  virtual void received_from_parent(char* buffer, size_t size) {
    comms_up->received(buffer, size, up_waypoint.get());
  }
  virtual void received_from_child(char* buffer, size_t size, RemoteId id) {
    UpDownWaypoint<false> down_waypoint(id);
    comms_down->received(buffer, size, &down_waypoint);
  }

  void ready_params_if_root(uint32_t version) {
    if (is_root()) {
      for (int i = 0; i < layers.size(); ++i) {
        layers[i].set_version(version);
        ready_local(up_info->get_sync_info(), i, version, "up");
      }
    }
  }

 public:
  SynchronousSync(TreeWaypoint* waypoint,
                  boost::shared_ptr<Solver<Dtype> > solver)
    : terminated_(false)
    , waypoint(waypoint)
    , solver(solver)
    , local_iters(solver->param().iter_size())
    , total_iters(0)
    , codec(BlobCodec<Dtype>::create_codec(
        solver->param().multinode_param(), true))
    , up_waypoint(new UpDownWaypoint<true>(waypoint->parent()))
    , down_waypoint(new UpDownWaypoint<false>(waypoint->id()))
    , up_info(new BlobInfo<Dtype>(solver, codec->max_elements_per_part()))
    , down_info(new BlobInfo<Dtype>(solver, codec->max_elements_per_part()))
    , keychain(BlobKeyChain<Dtype>::create(up_info->get_const_info()->layers()))
    , comms_up(BlobComms<Dtype>::create(
        solver, up_info, up_waypoint, codec, keychain,
        typename BlobComms<Dtype>::Settings(
          BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
        0))
    , comms_down(BlobComms<Dtype>::create(
        solver, down_info, down_waypoint, codec, keychain,
        typename BlobComms<Dtype>::Settings(
          BlobEncoding::PARAMS, BlobEncoding::GRADS, 1.0, 1.0),
        0))
    , layers(up_info->get_const_info()->layers())
    , parent_sync(this)
    , children_sync(this) {
    waypoint->set_buffer_size(codec->packet_size());
    if (!is_root()) solver->param().clear_snapshot();
    MLOG(1) << "initialized sync node with parent: " << waypoint->parent()
      << ", and num of children " << waypoint->children().size();

    waypoint->register_receive_handler(this);

    up_info->get_sync_info()->register_synced_handler(&parent_sync);
    down_info->get_sync_info()->register_synced_handler(&children_sync);

    up_info->get_sync_info()->add_remote(waypoint->parent());

    down_info->get_sync_info()->add_remote(waypoint->id());
    for (int i = 0; i < waypoint->children().size(); ++i)
      down_info->get_sync_info()->add_remote(waypoint->children()[i]);

    for (int i = 0; i < layers.size(); ++i) {
      layers[i].move_to(LayerState::updating);
    }
  }

  void init() {
    ready_params_if_root(1);
    solver->set_iter(1);
    create_timer(
      waypoint->get_daemon(),
      500000,
      boost::bind(&SynchronousSync::tick, this),
      true);
    if (is_root()) return;
  }

  bool is_root() const {
    return waypoint->id() == waypoint->parent();
  }

  bool is_leaf() const {
    return waypoint->children().empty();
  }

  virtual void synced_gradients(int layer_id,
                                int blob_id,
                                int part,
                                uint32_t version) {
    set_iter_size();
    if (!is_root()) {
      comms_up->cancel(layer_id, version - 1);
      comms_up->push(layer_id, blob_id, part, version);
    }
  }

  void set_iter_size() {
    {
      boost::mutex::scoped_lock lock(mtx);
      if (total_iters == solver->param().iter_size())
        return;
      total_iters = down_info->get_sync_info()->get_total_iters();
    }
    for (int i = 0; i < up_info->get_const_info()->layers(); ++i) {
      keychain->lock(i);
    }
    solver->param().set_iter_size(
      down_info->get_sync_info()->get_total_iters());
    comms_up->set_iter_size(solver->param().iter_size());
    for (int i = 0; i < up_info->get_const_info()->layers(); ++i) {
      keychain->unlock(i);
    }
  }

  virtual void synced_gradients(int layer_id, uint32_t version) {
    MLOG(1) << "layer " << layer_id
               << " gradients are in synced with version " << version;
    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    if (is_root()) {
      set_iter_size();
      keychain->lock(layer_id);
      for (int j = 0; j < param_ids.size(); ++j) {
        solver->ApplyUpdate(param_ids[j]);
      }
      keychain->unlock(layer_id);

      // required to sync params for the layer (and all its parts)
      ready_local(up_info->get_sync_info(), layer_id, version + 1, "up");
    }
  }

  virtual void synced_gradients(uint32_t version) {
    MLOG(1) << "net gradients are synced with version: " << version;
    if (is_root()) {
      if ((solver->param().test_interval() > 0)
          && (version % solver->param().test_interval() == 0)
          && ((version > 0) || (solver->param().test_initialization()))) {
        solver->TestAll();
      }
    }
  }

  virtual void ready_local(boost::shared_ptr<BlobSyncInfo> sync,
                           int layer_id,
                           uint32_t version,
                           std::string what) {
    DLOG(INFO) << what << " ready local layer " << layer_id
               << " of version " << version;
    for (int i = 0; i < up_info->get_const_info()->blobs(layer_id); ++i) {
      for (int j = 0; j < up_info->get_const_info()->parts(layer_id, i); ++j) {
        sync->received(waypoint->id(), layer_id, i, j,
          version, local_iters);
      }
    }
  }

  // triggered by sync from param or by apply update in root
  // requires (ready_params_if_root, ready_local(up_info, ...))
  // in internal/leaf requires also (received) from parent (comms_up)
  virtual void synced_parameters(int layer_id,
                                 int blob_id,
                                 int part,
                                 uint32_t version) {
    DLOG(INFO) << "part (" << layer_id << ", " << blob_id << ", " << part << ")"
            << " is in ready with version " << version;
    if (!is_leaf()) {
      // pushes params down the tree
      comms_down->cancel(layer_id, version - 1);
      comms_down->push(layer_id, blob_id, part, version);
    }
  }

  // triggered by sync from param or by apply update in root
  // in root triggered by (ready_params_if_root, ready_local(up_info, ...))
  // in internal/leaf triggered by (received)
  virtual void synced_parameters(int layer_id, uint32_t version) {
    MLOG(1) << "layer " << layer_id
            << " params are ready with version " << version;
    layers.at(layer_id).set_version(version);
    // allows calculation to continue with the layer
    layers.at(layer_id).move_to(LayerState::calculating);
  }

  virtual void synced_parameters(uint32_t version) {
    MLOG(1) << "net parameters are synced with version: " << version;
  }

  // called from solver thread
  void update(int layer_id) {
    if (!up_info->get_const_info()->needs_syncing(layer_id)) return;
    DLOG(INFO) << "backward ready for layer " << layer_id
      << " with version: " << layers[layer_id].get_version();
    layers.at(layer_id).move_to(LayerState::updating);
    ready_local(down_info->get_sync_info(),
                layer_id,
                layers[layer_id].get_version(),
                "down");
  }

  // called from solver thread
  void calculate(int layer_id) {
    if (!up_info->get_const_info()->needs_syncing(layer_id)) return;
    DLOG(INFO)  << "waiting for layer " << layer_id;
    int waited = layers.at(layer_id).wait_till(this, LayerState::calculating);

    // definitely the blob is not used at the moment
    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    for (int j = 0; j < param_ids.size(); ++j)
      solver->net()->ClearParamDiffs(param_ids[j]);

    if (waited > 0) {
      MLOG(1) << "waited on calculating layer " << layer_id
              << " " << (waited / 10.0) << "seconds";
    }
  }
};

}  // namespace

template <typename Dtype>
class SynchronousNode<Dtype>::Impl : public MultiSolver<Dtype>::Callback {
  boost::shared_ptr<MultiSolver<Dtype> > solver;
  TreeWaypoint* waypoint;
  SynchronousSync<Dtype> sync;

 public:
  Impl(boost::shared_ptr<Solver<Dtype> > solver)
    : solver(boost::make_shared<MultiSolver<Dtype> >(solver))
    , waypoint(TreeWaypoint::get_instance())
    , sync(waypoint, solver) {
  }

  void run() {
    DLOG(INFO) << "[proc " << waypoint->id() << "] solving";
    solver->add_callback(this);
    sync.init();
    solver->Solve();
  }

  void on_start() {
    if (!sync.is_started()) {
      sync.StartInternalThread();
    }
  }

  void on_start(int layer_id) {
    sync.calculate(layer_id);
    sync.keychain->lock(layer_id);
  }

  void on_forward_finished(int layer_id) {
    sync.keychain->unlock(layer_id);
  }

  void on_gradients_ready() {
  }

  void on_backward_start(int layer_id) {
    sync.keychain->lock(layer_id);
  }

  void on_gradients_ready(int layer_id) {
    sync.keychain->unlock(layer_id);
    sync.update(layer_id);
  }
};

template<typename Dtype>
SynchronousNode<Dtype>::SynchronousNode(shared_ptr<Solver<Dtype> > solver, int)
  : impl(boost::make_shared<Impl>(solver)) {
  solver->param().set_disabled_update(true);
}

template<typename Dtype>
void SynchronousNode<Dtype>::run() {
#ifndef USE_MPI
  LOG(ERROR) << "can't run mpi based training without configured MPI";
  return;
#endif
  impl->run();
}

INSTANTIATE_CLASS(SynchronousNode);

}  // namespace caffe

