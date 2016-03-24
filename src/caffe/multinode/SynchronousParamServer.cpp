#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/guaranteed_comm.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"
#include "caffe/multinode/SynchronousParamServer.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

using internode::RemoteId;
using internode::Daemon;
using internode::Waypoint;
using internode::MultiWaypoint;

template <typename Dtype>
class SynchronousParamServer<Dtype>::Impl : public MultiWaypoint::Handler
                                          , public BlobSyncInfo::Handler {
  shared_ptr<Daemon> comm;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<MultiWaypoint> waypoint;
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<BlobInfo<Dtype> > info;
  shared_ptr<BlobKeyChain<Dtype> > keychain;
  shared_ptr<BlobComms<Dtype> > comms;

  typedef boost::unordered_map<RemoteId, shared_ptr<Waypoint> > ClientMap;
  ClientMap all_clients;
  ClientMap pending;
  uint32_t current_version;
  boost::mutex mtx;

  virtual void synced(int layer_id, int blob_id, int part, uint32_t version) {
  }

  virtual void synced(int layer_id, uint32_t version) {
    VLOG(2) << "layer " << layer_id
            << " is synced with version: " << version;

    {
      boost::mutex::scoped_lock lock(mtx);
      if (all_clients.empty()) return;
    }

    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    // when synced it should be after all parts are received
    // and before anything is being send
    // so no locking is needed here
    for (int j = 0; j < param_ids.size(); ++j) {
      solver->ApplyUpdate(param_ids[j]);
      solver->net()->ClearParamDiffs(param_ids[j]);
    }

    comms->push(layer_id, version + 1);
  }

  virtual void synced(uint32_t version) {
    VLOG(2) << "net is synced with version: " << version;
    if ((solver->param().test_interval() > 0)
        && (version % solver->param().test_interval() == 0)
        && ((version > 0) || (solver->param().test_initialization()))) {
      solver->TestAll();
    }
    if ((solver->param().snapshot()
         && version % solver->param().snapshot() == 0)) {
      solver->Snapshot();
    }
    boost::mutex::scoped_lock lock(mtx);
    current_version = version + 1;
    add_pending();

    solver->param().set_iter_size(info->get_sync_info()->get_total_iters());
    solver->set_iter(version + 1);
  }

  void add_pending() {
    if (pending.empty()) return;
    if (all_clients.empty() &&
        (pending.size() < solver->param().multinode_param().wait_for_clients()))
      return;
    typedef ClientMap::iterator It;
    for (It it = pending.begin(); it != pending.end(); ++it) {
      all_clients.insert(*it);
      info->get_sync_info()->add_remote(it->first);
    }
    pending.clear();
    for (int i = 0; i < info->get_const_info()->layers(); ++i) {
      if (!info->get_const_info()->needs_syncing(i)) continue;
      comms->push(i, current_version + 1);
    }
  }

  void accepted(shared_ptr<Waypoint> waypoint) {
    {
      boost::mutex::scoped_lock lock(mtx);
      pending[waypoint->id()] = waypoint;
      if (all_clients.empty()) add_pending();
    }
    LOG(INFO) << "accepted client " << waypoint->id();
  }

  void disconnected(internode::RemoteId id) {
    LOG(INFO) << "client disconnected " << id;
    info->get_sync_info()->remove_remote(id);

    boost::mutex::scoped_lock lock(mtx);
    all_clients.erase(id);
    pending.erase(id);
  }

 public:
  Impl(shared_ptr<Solver<Dtype> > solver,
       string bind_address,
       int num_of_threads)
    : comm(internode::create_communication_daemon())
    , codec(BlobCodec<Dtype>::create_codec(
        solver->param().multinode_param(), false))
    , waypoint(internode::configure_server(
        comm, bind_address, codec->packet_size()))
    , solver(solver)
    , info(new BlobInfo<Dtype>(solver, codec->max_elements_per_part()))
    , keychain(BlobKeyChain<Dtype>::create_empty(
        info->get_const_info()->layers()))
    , comms(
        BlobComms<Dtype>::create(
          solver, info, waypoint, codec, keychain,
          typename BlobComms<Dtype>::Settings(
            BlobEncoding::PARAMS, BlobEncoding::GRADS, 1.0, 1.0),
          num_of_threads))
    , current_version(0u) {
    waypoint->register_peer_change_handler(this);
    info->get_sync_info()->register_synced_handler(this);
    waypoint->register_receive_handler(comms.get());

    internode::create_timer(comm, 500000, boost::bind(&Impl::tick, this), true);
  }

  void tick() {
  }

  void run() {
    LOG(INFO) << "param server running";
    while (solver->GetRequestedAction() == SolverAction::NONE) {
      internode::run_one(comm);
    }
  }
};

template <typename Dtype>
SynchronousParamServer<Dtype>::SynchronousParamServer(
        shared_ptr<Solver<Dtype> > solver,
        string bind_address,
        string,
        int num_of_threads)
  : impl(boost::make_shared<Impl>(solver, bind_address, num_of_threads)) {
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::run() {
  impl->run();
}

INSTANTIATE_CLASS(SynchronousParamServer);

}  // namespace caffe

