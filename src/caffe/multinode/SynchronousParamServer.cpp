#include <boost/make_shared.hpp>
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
  shared_ptr<MultiWaypoint> waypoint;
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<BlobInfo<Dtype> > info;
  shared_ptr<BlobKeyChain<Dtype> > keychain;
  shared_ptr<BlobComms<Dtype> > comms;

  typedef boost::unordered_map<RemoteId, shared_ptr<Waypoint> > ClientMap;
  ClientMap all_clients;
  ClientMap pending;

  virtual void synced(int layer_id, uint32_t version) {
    VLOG(2) << "layer " << layer_id
            << " is synced with version: " << version;

    if (all_clients.empty()) return;

    vector<int> param_ids =
      solver->net()->get_layer_learnable_param_ids(layer_id);
    solver->set_iter(version);
    solver->param().set_iter_size(info->get_sync_info()->get_total_iters());
    for (int j = 0; j < param_ids.size(); ++j) {
      solver->ApplyUpdate(param_ids[j]);
      solver->net()->ClearParamDiffs(param_ids[j]);
    }

    comms->push(
      layer_id, info->get_sync_info()->min_received_version(layer_id) + 1);
  }

  virtual void synced(uint32_t version) {
    VLOG(2) << "net is synced with version: " << version;
    if ((solver->param().test_interval() > 0)
        && (version % solver->param().test_interval() == 0)
        && ((version > 0) || (solver->param().test_initialization()))) {
      solver->TestAll();
    }
    add_pending();
  }

  void add_pending() {
    if (pending.empty()) return;
    typedef ClientMap::iterator It;
    for (It it = pending.begin(); it != pending.end(); ++it) {
      all_clients.insert(*it);
      info->get_sync_info()->add_remote(it->first);
    }
    pending.clear();
    uint32_t next_version = std::max(1u, comms->currently_sending_version());
    for (int i = 0; i < info->get_const_info()->layers(); ++i) {
      if (!info->get_const_info()->needs_syncing(i)) continue;
      comms->push(i, next_version);
    }
  }

  void accepted(shared_ptr<Waypoint> waypoint) {
    pending[waypoint->id()] = waypoint;

    const int start_num_of_clients =
      solver->param().multinode_param().wait_for_clients();
    if (all_clients.empty() && (pending.size() >= start_num_of_clients)) {
      add_pending();
    }
    LOG(INFO) << "accepted client " << waypoint->id();
  }

  void disconnected(internode::RemoteId id) {
    LOG(INFO) << "client disconnected " << id;

    if (all_clients.find(id) != all_clients.end())
      info->get_sync_info()->remove_remote(id);
    all_clients.erase(id);
    pending.erase(id);
  }

 public:
  Impl(shared_ptr<Solver<Dtype> > solver, string bind_address)
    : comm(internode::create_communication_daemon())
    , waypoint(internode::configure_server(comm, bind_address))
    , solver(solver)
    , codec(BlobCodec<Dtype>::create_codec(solver->param().multinode_param()))
    , info(new BlobInfo<Dtype>(solver, codec->max_elements_per_part()))
    , keychain(BlobKeyChain<Dtype>::create(info->get_const_info()->layers()))
    , comms(
        BlobComms<Dtype>::create(
          solver, info, keychain, waypoint, codec,
          typename BlobComms<Dtype>::Settings(
            BlobEncoding::PARAMS, BlobEncoding::GRADS, 1.0, 1.0))) {
    waypoint->register_peer_change_handler(this);
    info->get_sync_info()->register_synced_handler(this);
    waypoint->register_receive_handler(comms.get());
  }

  void run() {
    LOG(INFO) << "param server running";
    while (solver->GetRequestedAction() == SolverAction::NONE) {
      internode::poll_one(comm);
    }
  }
};

template <typename Dtype>
SynchronousParamServer<Dtype>::SynchronousParamServer(
        shared_ptr<Solver<Dtype> > solver, string bind_address)
  : impl(boost::make_shared<Impl>(solver, bind_address)) {
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::run() {
  impl->run();
}

INSTANTIATE_CLASS(SynchronousParamServer);

}  // namespace caffe

