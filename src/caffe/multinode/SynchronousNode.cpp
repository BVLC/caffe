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
    #ifdef USE_MPI
      return terminated_;
    #else
      terminated_ =
        terminated_ || (solver->GetRequestedAction() != SolverAction::NONE);
      return terminated_;
    #endif
  }

  virtual void terminate() {
    boost::mutex::scoped_lock lock(mtx);
    terminated_ = true;
  }

  virtual void InternalThreadEntry() {
    create_timer(
      waypoint->get_daemon(),
      5000,
      boost::bind(&SynchronousSync::tick, this),
      true);
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
        ready_local(up_sync, i, version, "up");
      }
    }
  }

 public:
  SynchronousSync(TreeWaypoint* waypoint,
                  boost::shared_ptr<Solver<Dtype> > solver)
    : terminated_(false)
    , total_iters(solver->param().iter_size())
    , waypoint(waypoint)
    , solver(solver)
    , codec(BlobCodec<Dtype>::create_codec(
        solver->param().multinode_param(), true))
    , up_waypoint(new UpDownWaypoint<true>(waypoint->parent()))
    , down_waypoint(new UpDownWaypoint<false>(waypoint->id()))
    , blob_accessor(BlobInfoFactory<Dtype>::create_blob_accessor(solver))
    , const_info(BlobInfoFactory<Dtype>::create_const_info(
        solver, codec->max_elements_per_part()))
    , up_sync(BlobInfoFactory<Dtype>::create_sync_info(const_info))
    , down_sync(BlobInfoFactory<Dtype>::create_sync_info(const_info))
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
    MLOG(1) << "initialized sync node with parent: " << waypoint->parent()
      << ", and num of children " << waypoint->children().size();

    comms_down->register_iter_size_handler(this);
    waypoint->register_receive_handler(this);

    up_sync->register_synced_handler(&parent_sync);
    down_sync->register_synced_handler(&children_sync);

    up_sync->add_remote(waypoint->parent());

    down_sync->add_remote(waypoint->id());
    for (int i = 0; i < waypoint->children().size(); ++i)
      down_sync->add_remote(waypoint->children()[i]);

    for (int i = 0; i < layers.size(); ++i) {
      layers[i].move_to(LayerState::updating);
    }
    solver->set_iter(1);
    if (is_leaf()) {
      if (is_root()) {
        ready_params_if_root(1);
      } else {
        comms_up->send_iter_size(total_iters);
      }
    }
  }

  bool is_root() const {
    return waypoint->id() == waypoint->parent();
  }

  bool is_leaf() const {
    return waypoint->children().empty();
  }

  virtual void received_iter_size(RemoteId from, int iters) {
    boost::mutex::scoped_lock lock(mtx);
    int update = iters - children_iter_size[from];
    children_iter_size[from] = iters;
    total_iters += update;
    DLOG(INFO) << "received_iter_size: " << total_iters;
    if (children_iter_size.size() != waypoint->children().size()) {
      return;
    }
    if (is_root()) {
      ready_params_if_root(1);
      LOG(INFO) << "initialized root of cluster with nodes: "
                << waypoint->total_nodes()
                << " and the total iter size is: " << total_iters;
    } else {
      MLOG(2) << "iter size of the subtree from this node is: " << iters;
      comms_up->send_iter_size(total_iters);
    }
  }

  virtual void synced_gradients(int layer_id,
                                int blob_id,
                                int part,
                                uint32_t version) {
    if (!is_root()) {
      comms_up->cancel(layer_id, version - 1);
      comms_up->push(layer_id, blob_id, part, version);
    }
  }

  void dump_weights(int i, uint32_t version, string prefix = "") {
    vector<int> param_ids = solver->net()->get_layer_learnable_param_ids(i);
    if (param_ids.empty()) return;

    for (int j = 0; j < solver->net()->layers()[i]->blobs().size(); ++j) {
      string dir =
        solver->param().has_dump_dir() ? solver->param().dump_dir() : ".";
      CHECK_GT(dir.size(), 0);
      if (dir[dir.size() - 1] != '/') dir += "/";
      string layer_name = solver->net()->layers()[i]->layer_param().name();
      for (int k = 0; k < layer_name.size(); ++k)
        if (layer_name[k] == '/')
          layer_name[k] = '_';
      string filename = "dump_" + (prefix.size() ? (prefix + "_") : string())
        + boost::lexical_cast<string>(version)
        + "_" + boost::lexical_cast<string>(waypoint->id())
        + "_" + layer_name + "_"
        + boost::lexical_cast<string>(j) + ".dump";
      string path = dir + filename;
      std::ofstream dump(path.c_str(), std::ios::binary);

      Blob<Dtype>* blob = solver->net()->layers()[i]->blobs()[j].get();
      const char* data = reinterpret_cast<const char*>(blob->cpu_diff());
      const size_t size = blob->count() * sizeof(Dtype);
      dump.write(data, size);
      dump.close();
    }
  }

  virtual void synced_gradients(int layer_id, uint32_t version) {
    MLOG(1) << "layer " << layer_id
               << " gradients are in synced with version " << version;
    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    if (is_root()) {
      keychain->lock(layer_id);

      if (solver->param().dump_checksums()) {
        int blobs = solver->net()->layers()[layer_id]->blobs().size();
        for (int i = 0; i < blobs; ++i) {
          Dtype sum = check_sum(
            solver->net()->layers()[layer_id]->blobs()[i].get(),
            BlobEncoding::GRADS);
          MLOG(2)
            << "version " << version
            << " layer " << layer_id << " blob " << i
            << " accumulated checksum " << std::setprecision(30) << sum;
        }
      }
      if (solver->param().has_dump_dir())
        dump_weights(layer_id, version, "acc");

      Dtype accum_normalization = Dtype(1.);
      {
        boost::mutex::scoped_lock lock(mtx);
        accum_normalization /= total_iters;
      }
      for (int j = 0; j < param_ids.size(); ++j) {
        const vector<Blob<Dtype>*>& net_params =
          solver->net()->learnable_params();
        switch (Caffe::mode()) {
        case Caffe::CPU: {
          caffe_scal(net_params[param_ids[j]]->count(), accum_normalization,
                   net_params[param_ids[j]]->mutable_cpu_diff());
          break;
        }
        case Caffe::GPU: {
      #ifndef CPU_ONLY
          caffe_gpu_scal(net_params[param_ids[j]]->count(), accum_normalization,
                   net_params[param_ids[j]]->mutable_gpu_diff());
      #else
          NO_GPU;
      #endif
          break;
        }
        }
        solver->ApplyUpdate(param_ids[j]);
      }
      keychain->unlock(layer_id);

      // required to sync params for the layer (and all its parts)
      ready_local(up_sync, layer_id, version + 1, "up");
    }
  }

  virtual void synced_gradients(uint32_t version) {
    MLOG(1) << "net gradients are synced with version: " << version;
    if (is_root()) {
      for (int i = 0; i < const_info->layers(); ++i) {
        keychain->lock(i);
      }
      if ((solver->param().test_interval() > 0)
          && (version % solver->param().test_interval() == 0)
          && ((version > 0) || (solver->param().test_initialization()))) {
        solver->TestAll();
      }
      for (int i = 0; i < const_info->layers(); ++i) {
        keychain->unlock(i);
      }
    }
  }

  virtual void ready_local(boost::shared_ptr<BlobSyncInfo> sync,
                           int layer_id,
                           uint32_t version,
                           std::string what) {
    DLOG(INFO) << what << " ready local layer " << layer_id
               << " of version " << version;
    for (int i = 0; i < const_info->blobs(layer_id); ++i) {
      for (int j = 0; j < const_info->parts(layer_id, i); ++j) {
        sync->received(waypoint->id(), layer_id, i, j, version);
      }
    }
  }

  // triggered by sync from param or by apply update in root
  // requires (ready_params_if_root, ready_local(up_sync, ...))
  // in internal/leaf requires also (received) from parent (comms_up)
  virtual void synced_parameters(int layer_id,
                                 int blob_id,
                                 int part,
                                 uint32_t version) {
    DLOG(INFO) << "part (" << layer_id << ", " << blob_id << ", " << part << ")"
            << " is in ready with version " << version;


    // can't receive gradient update until all parameters where propagated down
    // can't calculate all parameters where received
    // we can safely clear diffs here
    // definitely the blob is not used at the moment
    // it has to be cleared before any gradients can be received from child
    if ((blob_id == 0) && (part == 0)) {  // it has to be done only once
      vector<int> param_ids =
        solver->net()->get_layer_learnable_param_ids(layer_id);
      for (int j = 0; j < param_ids.size(); ++j)
        solver->net()->ClearParamDiffs(param_ids[j]);
    }
    if (!solver->param().has_dump_dir() && !is_leaf()) {
      // pushes params down the tree
      comms_down->cancel(layer_id, version - 1);
      comms_down->push(layer_id, blob_id, part, version);
    }
  }

  // triggered by sync from param or by apply update in root
  // in root triggered by (ready_params_if_root, ready_local(up_sync, ...))
  // in internal/leaf triggered by (received, ready_local(up_sync, ...))
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
    if (!const_info->needs_syncing(layer_id)) return;
    DLOG(INFO) << "backward ready for layer " << layer_id
      << " with version: " << layers[layer_id].get_version();
    layers.at(layer_id).move_to(LayerState::updating);
    if (solver->param().has_dump_dir() && !is_leaf()) {
      // pushes params down the tree
      comms_down->cancel(layer_id, layers[layer_id].get_version() - 1);
      comms_down->push(layer_id, layers[layer_id].get_version());
    }
    ready_local(down_sync, layer_id, layers[layer_id].get_version(), "down");
  }

  // called from solver thread
  void calculate(int layer_id) {
    if (!const_info->needs_syncing(layer_id)) return;

    DLOG(INFO)  << "waiting for layer " << layer_id;
    int waited = layers.at(layer_id).wait_till(this, LayerState::calculating);

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

  vector<Dtype> partial_checksums;

  int max_blobs() {
    int ret = 0;
    for (int i = 0; i < solver->net().layers().size(); ++i) {
      ret = std::max(
        ret, static_cast<int>(solver->net().layers()[i]->blobs().size()));
    }
    return ret;
  }

 public:
  Impl(boost::shared_ptr<Solver<Dtype> > solver)
    : solver(boost::make_shared<MultiSolver<Dtype> >(
        solver, (Caffe::mode() != Caffe::CPU)))
    , waypoint(TreeWaypoint::get_instance())
    , sync(waypoint, solver)
    , partial_checksums(solver->net()->layers().size() * max_blobs()) {
  }

  void run() {
    DLOG(INFO) << "[proc " << waypoint->id() << "] solving";
    solver->add_callback(this);
    solver->Solve();
    sync.terminate();
    sync.StopInternalThread();
  }

  void on_start() {
    if (!sync.is_started()) {
      solver->net().ClearParamDiffs();
      sync.StartInternalThread();
    }
  }

  void on_start(int layer_id) {
    sync.calculate(layer_id);
  }

  void on_forward_finished(int layer_id) {
  }

  void on_gradients_ready() {
  }

  void on_backward_start(int layer_id) {
    sync.keychain->lock(layer_id);
    for (int i = 0; i < solver->net().layers()[layer_id]->blobs().size(); ++i) {
      if (solver->param().has_dump_dir())
        CHECK(check_sum(solver->net().layers()[layer_id]->blobs()[i].get(),
                        BlobEncoding::GRADS) == 0.0f);
      else if (solver->param().dump_checksums())
        partial_checksums[layer_id * 2 + i]
          = check_sum(solver->net().layers()[layer_id]->blobs()[i].get(),
                      BlobEncoding::GRADS);
    }
  }

  void on_gradients_ready(int layer_id) {
    if (solver->param().dump_checksums()) {
      int blobs = solver->net().layers()[layer_id]->blobs().size();
      for (int i = 0; i < blobs; ++i) {
        Dtype sum = check_sum(
          solver->net().layers()[layer_id]->blobs()[i].get(),
          BlobEncoding::GRADS);
        if (!solver->param().has_dump_dir())
          sum -= partial_checksums[layer_id * 2 + i];
        else
          CHECK(partial_checksums[layer_id * 2 + i] == 0.0);
        MLOG(2) <<
          "version: " << sync.layers[layer_id].get_version()
          << " layer " << layer_id << " blob "
          << i << " calculated checksum " << std::setprecision(30) << sum
            << " " << partial_checksums[layer_id * 2 + i];
      }
    }
    if (solver->param().has_dump_dir()) {
      int blobs = solver->net().layers()[layer_id]->blobs().size();
      for (int i = 0; i < blobs; ++i) {
        sync.dump_weights(layer_id, sync.layers[layer_id].get_version());
      }
    }
    sync.keychain->unlock(layer_id);
    sync.update(layer_id);
  }
};

template<typename Dtype>
SynchronousNode<Dtype>::SynchronousNode(shared_ptr<Solver<Dtype> > solver, int)
  : impl(boost::make_shared<Impl>(solver)) {
  solver->param().set_disabled_update(true);
  solver->param().set_test_interval(0);
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

