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
class SynchronousParamServer<Dtype>::Impl
    : public MultiWaypoint::Handler
    , public BlobSyncInfo::Handler
    , public BlobComms<Dtype>::IterSizeHandler {
  shared_ptr<Daemon> comm;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<MultiWaypoint> waypoint;
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<BlobConstInfo> const_info;
  shared_ptr<BlobSyncInfo> sync_info;
  shared_ptr<BlobKeyChain<Dtype> > keychain;
  shared_ptr<BlobComms<Dtype> > comms;

  struct ClientInfo {
    shared_ptr<Waypoint> waypoint;
    int iters;
  };
  typedef boost::unordered_map<RemoteId, ClientInfo> ClientMap;
  ClientMap all_clients;
  ClientMap pending;
  uint32_t current_version;
  int total_iters;
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

  virtual void received_iter_size(RemoteId from, int iters) {
    boost::mutex::scoped_lock lock(mtx);
    pending[from].iters = iters;
    total_iters += iters;
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
    solver->param().set_iter_size(total_iters);
    current_version = version + 1;
    add_pending();

    solver->set_iter(version + 1);
  }

  void add_pending() {
    if (pending.empty()) return;
    if (all_clients.empty() &&
        (pending.size() < solver->param().multinode_param().wait_for_clients()))
      return;
    typedef typename ClientMap::iterator It;
    ClientMap left;
    for (It it = pending.begin(); it != pending.end(); ++it) {
      if (it->second.iters > 0) {
        all_clients.insert(*it);
        sync_info->add_remote(it->first);
      } else {
        left.insert(*it);
      }
    }
    pending.swap(left);
    for (int i = 0; i < const_info->layers(); ++i) {
      if (!const_info->needs_syncing(i)) continue;
      comms->push(i, current_version + 1);
    }
  }

  void accepted(shared_ptr<Waypoint> waypoint) {
    {
      boost::mutex::scoped_lock lock(mtx);
      ClientInfo info = {waypoint, 0};
      pending[waypoint->id()] = info;
      if (all_clients.empty()) add_pending();
    }
    LOG(INFO) << "accepted client " << waypoint->id();
  }

  void disconnected(internode::RemoteId id) {
    LOG(INFO) << "client disconnected " << id;
    sync_info->remove_remote(id);

    boost::mutex::scoped_lock lock(mtx);
    total_iters -= all_clients[id].iters;
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
    , const_info(BlobInfoFactory<Dtype>::create_const_info(
        solver, codec->max_elements_per_part()))
    , sync_info(BlobInfoFactory<Dtype>::create_sync_info(const_info))
    , keychain(BlobKeyChain<Dtype>::create_empty(const_info->layers()))
    , comms(
        BlobComms<Dtype>::create(
          solver, const_info, sync_info, waypoint, codec, keychain,
          typename BlobComms<Dtype>::Settings(
            BlobEncoding::PARAMS, BlobEncoding::GRADS, 1.0, 1.0),
          num_of_threads))
    , current_version(0u)
    , total_iters(0) {
    waypoint->register_peer_change_handler(this);
    sync_info->register_synced_handler(this);
    waypoint->register_receive_handler(comms.get());
    comms->register_iter_size_handler(this);

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

