#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
#include <glog/logging.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
#include "caffe/util/device_alternate.hpp"

namespace caffe {

#define CLOG(arg) \
  LOG(arg) << "[" << TreeWaypoint::get_instance()->id() << "] "
#define CVLOG(arg) \
  VLOG(arg) << "[" << TreeWaypoint::get_instance()->id() << "] "
#define CDLOG(arg) \
  DLOG(arg) << "[" << TreeWaypoint::get_instance()->id() << "] "

namespace {

using internode::Waypoint;
using internode::TreeWaypoint;
using internode::RemoteId;

struct TerminatedHandler {
  virtual bool terminated() = 0;
  virtual void on_wake_up() = 0;
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

  LayerState() : state(updating), version(0u) {
  }
  LayerState(const LayerState& other)
    : state(other.state)
    , version(other.version) {
  }

  void move_to(Enum next_state) {
    boost::mutex::scoped_lock lock(mtx);
    state = next_state;
    cond.notify_all();
  }

  void wake_up() {
    boost::mutex::scoped_lock lock(mtx);
    cond.notify_all();
  }

  int wait_till(TerminatedHandler* handler, uint32_t wait_for_version) {
    boost::mutex::scoped_lock lock(mtx);
    int ret = 0;
    while ((state != calculating) || (version != wait_for_version)) {
      boost::system_time timeout
        = boost::get_system_time() + boost::posix_time::milliseconds(10);
      cond.timed_wait(lock, timeout);
      ++ret;
      if (handler->terminated()) {
        return 0;
      }
      lock.unlock();
      handler->on_wake_up();
      lock.lock();
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
                      , public TreeWaypoint::Handler
                      , public BlobComms<Dtype>::IterSizeHandler {
  typedef boost::unordered_map<RemoteId, int> IterSizeInfo;
  IterSizeInfo children_iter_size;

  boost::mutex mtx;
  bool terminated_;
  int total_iters;
  TreeWaypoint* waypoint;
  boost::shared_ptr<Solver<Dtype> > solver;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<Waypoint> up_waypoint;
  shared_ptr<Waypoint> down_waypoint;
  shared_ptr<BlobAccessor<Dtype> > blob_accessor;
  shared_ptr<BlobConstInfo> const_info;
  shared_ptr<BlobSyncInfo> up_sync;
  shared_ptr<BlobSyncInfo> down_sync;
  boost::thread::id main_thread_id;
  boost::thread::id solver_thread_id;
  int snapshot_per_iters;
  vector<pair<int, uint32_t> > layers_to_update;
 public:
  shared_ptr<BlobKeyChain<Dtype> > keychain;
 public:
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
    boost::mutex::scoped_lock lock(mtx);
    return terminated_;
  }

  virtual void terminate() {
    boost::mutex::scoped_lock lock(mtx);
    if (!is_root())
      comms_up.finish_all_tasks();
    terminated_ = true;
  }

  virtual void InternalThreadEntry() {
    CLOG(INFO) << "Comm thread started " << is_leaf() << " " << is_root();
    create_timer(
      waypoint->get_daemon(),
      5000,
      boost::bind(&SynchronousSync::tick, this),
      true);
    main_thread_id = boost::this_thread::get_id();

    if (is_leaf()) {
      if (is_root()) {
        push_all_params_down(solver->iter());
      } else {
        comms_up->send_iter_size(total_iters);
      }
    }
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

 public:
  SynchronousSync(TreeWaypoint* waypoint,
                  boost::shared_ptr<Solver<Dtype> > root_solver)
    : terminated_(false)
    , total_iters(root_solver->param().iter_size())
    , waypoint(waypoint)
    , solver(root_solver)
    , codec(BlobCodec<Dtype>::create_codec(
        solver->param().multinode_param(), true))
    , up_waypoint(new UpDownWaypoint<true>(waypoint->parent()))
    , down_waypoint(new UpDownWaypoint<false>(waypoint->id()))
    , blob_accessor(BlobInfoFactory<Dtype>::create_blob_accessor(solver))
    , const_info(BlobInfoFactory<Dtype>::create_const_info(
        solver, codec->max_elements_per_part()))
    , up_sync(BlobInfoFactory<Dtype>::create_sync_info(const_info))
    , down_sync(BlobInfoFactory<Dtype>::create_sync_info(const_info))
    , main_thread_id(boost::this_thread::get_id())
    , snapshot_per_iters(solver->param().snapshot())
    , keychain(BlobKeyChain<Dtype>::create(const_info->layers()))
    , comms_up(BlobComms<Dtype>::create(blob_accessor,
        const_info, up_sync, up_waypoint, codec, keychain,
        typename BlobComms<Dtype>::Settings(
          BlobEncoding::GRADS, BlobEncoding::PARAMS, 1.0, 0.0),
        0))
    , comms_down(BlobComms<Dtype>::create(blob_accessor,
        const_info, down_sync, down_waypoint, codec, keychain,
        typename BlobComms<Dtype>::Settings(
          BlobEncoding::PARAMS, BlobEncoding::GRADS, 1.0, 1.0),
        0))
    , layers(const_info->layers())
    , parent_sync(this)
    , children_sync(this) {
    waypoint->set_buffer_size(codec->packet_size());
    if (!is_root()) solver->param().clear_snapshot();
    if (!is_root()) solver->param().clear_snapshot_after_train();
    CVLOG(1) << "initialized sync node with parent: " << waypoint->parent()
      << ", and num of children " << waypoint->children().size();

    comms_down->register_iter_size_handler(this);
    waypoint->register_receive_handler(this);

    up_sync->register_synced_handler(&parent_sync);
    down_sync->register_synced_handler(&children_sync);

    up_sync->add_remote(waypoint->parent());

    down_sync->add_remote(waypoint->id());
    for (int i = 0; i < waypoint->children().size(); ++i)
      down_sync->add_remote(waypoint->children()[i]);

    if (solver->iter() == 0)
      solver->set_iter(1);
    solver->param().set_iter_size(total_iters);
    for (int i = 0; i < layers.size(); ++i) {
      layers[i].set_version(solver->iter());
      CDLOG(INFO) << "layer " << i << " move to updating";
      layers[i].move_to(LayerState::updating);
    }
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
    if (!is_root()) {
      comms_up->push(layer_id, blob_id, part, version);
    }
  }

  virtual void synced_gradients(int layer_id, uint32_t version) {
    CVLOG(2) << "layer " << layer_id
               << " gradients are in synced with version " << version;
    if (is_root()) {
      boost::mutex::scoped_lock lock(mtx);
      layers_to_update.push_back(make_pair(layer_id, version));
    }
    layers.at(layer_id).wake_up();
  }

  virtual void synced_gradients(uint32_t version) {
    CVLOG(2) << "net gradients are synced with version: " << version;
  }

  virtual void ready_local(boost::shared_ptr<BlobSyncInfo> sync,
                           int layer_id,
                           uint32_t version) {
    for (int i = 0; i < const_info->blobs(layer_id); ++i) {
      for (int j = 0; j < const_info->parts(layer_id, i); ++j) {
        sync->received(waypoint->id(), layer_id, i, j, version);
      }
    }
  }

  // This part is called from Comm thread only
  void push_all_params_down(uint32_t version) {
    CHECK(main_thread_id == boost::this_thread::get_id());
    CHECK(is_root());
    for (int i = 0; i < layers.size(); ++i) {
      if (!const_info->needs_syncing(i)) continue;
      if (!is_leaf())
        comms_down->push(i, version);
      CDLOG(INFO) << "layer " << i << " move to calculating "
        << "version " << version;
      layers.at(i).set_version(version);
      layers.at(i).move_to(LayerState::calculating);
    }
  }

  virtual void received_iter_size(RemoteId from, int iters) {
    CHECK(main_thread_id == boost::this_thread::get_id());
    int update = iters - children_iter_size[from];
    children_iter_size[from] = iters;
    total_iters += update;
    CDLOG(INFO) << "received_iter_size: " << total_iters;
    if (children_iter_size.size() != waypoint->children().size()) {
      return;
    }
    solver->param().set_iter_size(total_iters);
    if (is_root()) {
      push_all_params_down(solver->iter());
      CLOG(INFO) << "initialized root of cluster with nodes: "
                << waypoint->total_nodes()
                << " and the total iter size is: " << total_iters;
    } else {
      CVLOG(2) << "iter size of the subtree from this node is: " << iters;
      comms_up->send_iter_size(total_iters);
    }
  }

  virtual void synced_parameters(int layer_id,
                                 int blob_id,
                                 int part,
                                 uint32_t version) {
    CHECK(main_thread_id == boost::this_thread::get_id());
    CDLOG(INFO) << "part (" << layer_id << ", " << blob_id << ", " << part
      << ")" << " is in ready with version " << version;
    keychain->lock(layer_id);
    if ((blob_id == 0) && (part == 0)) {
      vector<int> param_ids =
        solver->net()->get_layer_learnable_param_ids(layer_id);
      for (int j = 0; j < param_ids.size(); ++j)
        solver->net()->ClearParamDiffs(param_ids[j]);
    }
    keychain->unlock(layer_id);
    if (!is_leaf()) {
      // pushes params down the tree
      comms_down->push(layer_id, blob_id, part, version);
    }
  }

  virtual void synced_parameters(int layer_id, uint32_t version) {
    CHECK(main_thread_id == boost::this_thread::get_id());
    CVLOG(2) << "layer " << layer_id
            << " params are ready with version " << version;


    // allows calculation to continue with the layer
    CDLOG(INFO) << "layer " << layer_id
               << " move to calculating version " << version;
    layers.at(layer_id).set_version(version);
    layers.at(layer_id).move_to(LayerState::calculating);
  }

  virtual void synced_parameters(uint32_t version) {
    CHECK(main_thread_id == boost::this_thread::get_id());
    CVLOG(2) << "net parameters are synced with version: " << version;
  }

  // Everything below is called from Solver thread only
  void apply_updates(int layer_id, uint32_t version) {
    CHECK(boost::this_thread::get_id() == solver_thread_id);
    keychain->lock(layer_id);

    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    for (int i = 0; i < param_ids.size(); ++i) {
      solver->ApplyUpdate(param_ids[i]);
    }
    for (int j = 0; j < param_ids.size(); ++j)
      solver->net()->ClearParamDiffs(param_ids[j]);
    keychain->unlock(layer_id);

    version++;

    // allows calculation to continue with the layer
    CDLOG(INFO) << "layer " << layer_id << " move to calculating "
               << "version " << version;
    layers.at(layer_id).set_version(version);
    layers.at(layer_id).move_to(LayerState::calculating);

    if (!is_leaf()) {
      comms_down->push(layer_id, version);
      CDLOG(INFO) << "layer " << layer_id << " pushed down";
    }
    CDLOG(INFO) << "layer " << layer_id << " updated";
  }

  void apply_updates() {
    CHECK(boost::this_thread::get_id() == solver_thread_id);
    vector<pair<int, uint32_t> > to_update;
    {
      boost::mutex::scoped_lock lock(mtx);
      to_update.swap(layers_to_update);
    }
    if (to_update.size() > 0)
      CDLOG(INFO) << "apply_updates: " << to_update.size();
    for (int i = 0; i < to_update.size(); ++i)
      apply_updates(to_update[i].first, to_update[i].second);
  }

  void set_solver_thread() {
    solver_thread_id = boost::this_thread::get_id();
  }

  virtual void on_wake_up() {
    CHECK(boost::this_thread::get_id() == solver_thread_id);
    apply_updates();
  }

  void prepare_update(int layer_id) {
    CHECK(boost::this_thread::get_id() == solver_thread_id);
    if (!const_info->needs_syncing(layer_id)) return;
    CDLOG(INFO) << "backward ready for layer " << layer_id
      << " with version: " << layers[layer_id].get_version();
    CDLOG(INFO) << "layer " << layer_id << " move to updating "
      << "version " << solver->iter();
    layers.at(layer_id).move_to(LayerState::updating);
    ready_local(down_sync, layer_id, layers[layer_id].get_version());
  }

  void prepare_for_calculation(int layer_id) {
    CHECK(boost::this_thread::get_id() == solver_thread_id);
    if (!const_info->needs_syncing(layer_id)) return;

    CDLOG(INFO) << "waiting for layer " << layer_id
                << " in version " << solver->iter();
    int waited = layers.at(layer_id).wait_till(this, solver->iter());

    if (waited > 0) {
      CVLOG(1) << "waited on layer " << layer_id
              << " version " << solver->iter()
              << " " << (waited / 10.0) << "seconds";
    }
  }

  void wait_till_updated() {
    CHECK(boost::this_thread::get_id() == solver_thread_id);
    for (int i = 0; i < const_info->layers(); ++i) {
      if (!const_info->needs_syncing(i)) continue;
      prepare_for_calculation(i);
      layers.at(i).wait_till(this, solver->iter());
    }
  }

  void check_snapshot() {
    CHECK(boost::this_thread::get_id() == solver_thread_id)
      << boost::this_thread::get_id()
      << " " << solver_thread_id;
    if (!is_root()) return;
    if ((snapshot_per_iters != 0)
        && (solver->iter() % snapshot_per_iters == 0)) {
      wait_till_updated();
      solver->Snapshot();
    }
  }
};

}  // namespace

template <typename Dtype>
class SynchronousNode<Dtype>::Impl : public MultiSolver<Dtype>::Callback {
  boost::shared_ptr<MultiSolver<Dtype> > solver;
  SynchronousSync<Dtype> sync;
  bool initialized_;

  vector<Dtype> partial_checksums;

 public:
  Impl(boost::shared_ptr<Solver<Dtype> > solver)
    : solver(boost::make_shared<MultiSolver<Dtype> >(solver))
    , sync(TreeWaypoint::get_instance(), solver)
    , initialized_(false) {
  }

  void snapshot() {
    if (sync.is_root()) {
      sync.apply_updates();
      solver->root_solver()->Snapshot();
    }
  }

  void run() {
    CLOG(INFO) << "[proc " << TreeWaypoint::get_instance()->id() << "] solving";
    solver->add_callback(this);
    solver->Solve();
    if (sync.is_root()) {
      sync.wait_till_updated();
      solver->root_solver()->Snapshot();
    }
    sync.terminate();
    sync.StopInternalThread();
  }

  void on_start() {
    if (!initialized_) {
      sync.set_solver_thread();
      sync.StartInternalThread();
      initialized_ = true;
    }
    sync.check_snapshot();
    CDLOG(INFO) << "started iteration " << solver->root_solver()->iter();
  }

  void on_start(int layer_id) {
    CDLOG(INFO) << "started forward of layer " << layer_id;
    sync.apply_updates();
    sync.prepare_for_calculation(layer_id);
  }

  void on_forward_finished(int layer_id) {
    CDLOG(INFO) << "finished forward of layer " << layer_id;
  }

  void on_gradients_ready() {
    CDLOG(INFO) << "finished iteration " << solver->root_solver()->iter();
  }

  void on_backward_start(int layer_id) {
    CDLOG(INFO) << "calculating gradients of layer " << layer_id;
    sync.keychain->lock(layer_id);
  }

  void on_gradients_ready(int layer_id) {
    sync.keychain->unlock(layer_id);
    sync.prepare_update(layer_id);
    sync.apply_updates();
    CDLOG(INFO) << "ready gradients of layer " << layer_id;
  }
};

template<typename Dtype>
SynchronousNode<Dtype>::SynchronousNode(shared_ptr<Solver<Dtype> > solver, int)
  : impl(boost::make_shared<Impl>(solver)) {
  solver->param().set_disabled_update(true);
  solver->param().clear_test_interval();
  solver->param().clear_snapshot();
  solver->param().clear_snapshot_after_train();
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

