#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/guaranteed_comm.hpp"
#include "caffe/multinode/BlobComms.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"
#include "caffe/multinode/Relay.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

using internode::RemoteId;
using internode::Daemon;
using internode::Waypoint;
using internode::MultiWaypoint;

template <typename Dtype>
class ParamRelay<Dtype>::Impl : MultiWaypoint::Handler {
  shared_ptr<Daemon> comm;
  shared_ptr<BlobCodec<Dtype> > codec;
  shared_ptr<MultiWaypoint> down_waypoint;
  shared_ptr<Waypoint> up_waypoint;
  shared_ptr<Solver<Dtype> > solver;
  shared_ptr<BlobConstInfo> info;

  typedef boost::unordered_set<RemoteId> ClientSet;
  struct PartInfo {
    ClientSet to_receive_from;
    Blob<Dtype>* blob;
    uint32_t version;
    shared_ptr<std::vector<char> > last_updated;
    uint32_t last_updated_version;
    int iters;
  };
  std::vector<std::vector< std::vector<PartInfo> > > parts;
  typedef boost::unordered_map<RemoteId, uint32_t> Clients;
  Clients clients;
  Clients pending;
  uint32_t latest_version;

  struct ServerComm : Waypoint::Handler {
    Impl* impl;
    explicit ServerComm(Impl* impl) : impl(impl) {
    }
    virtual void received(char* buffer, size_t size, Waypoint* waypoint) {
      impl->received_as_server(buffer, size, waypoint->id());
    }
  } server_handler;

  struct ClientComm : Waypoint::Handler {
    Impl* impl;
    explicit ClientComm(Impl* impl) : impl(impl) {
    }
    virtual void received(char* buffer, size_t size, Waypoint*) {
      impl->received_as_client(buffer, size);
    }
  } client_handler;

  void add_pending() {
    for (int i = 0; i < info->layers(); ++i) {
      for (int j = 0; j < info->blobs(i); ++j) {
        for (int k = 0; k < info->parts(i, j); ++k) {
          PartInfo& part = parts[i][j][k];
          if (!part.last_updated) continue;
          if (part.last_updated_version < latest_version) continue;
          typedef Clients::iterator It;
          part.iters = 0;
          for (It it = pending.begin(); it != pending.end(); ++it) {
            part.to_receive_from.insert(it->first);
            part.iters++;
          }
          shared_ptr<std::vector<char> > copy = part.last_updated;
          down_waypoint->async_send(
            &copy->front(), copy->size(),
            boost::bind(&Impl::sent_vec, this, copy));
        }
      }
    }
    clients.insert(pending.begin(), pending.end());
    pending.clear();
  }

  void accepted(shared_ptr<Waypoint> waypoint) {
    pending[waypoint->id()] = latest_version;
    VLOG(1) << "accepted client " << waypoint->id()
              << " at iteration: " << latest_version;

    if (clients.empty()
        && (pending.size() >=
            solver->param().multinode_param().wait_for_clients())) {
      add_pending();
    }
  }

  void disconnected(RemoteId id) {
    clients.erase(id);
    pending.erase(id);
    for (int i = 0; i < info->layers(); ++i)
      for (int j = 0; j < info->blobs(i); ++j)
        for (int k = 0; k < info->parts(i, j); ++k)
          parts[i][j][k].to_receive_from.erase(id);
    for (int i = 0; i < info->layers(); ++i)
      for (int j = 0; j < info->blobs(i); ++j)
        for (int k = 0; k < info->parts(i, j); ++k)
          propagate_if_ready(i, j, k);
    if (clients.empty() && pending.empty()) {
      for (int i = 0; i < info->layers(); ++i) {
        for (int j = 0; j < info->blobs(i); ++j) {
          for (int k = 0; k < info->parts(i, j); ++k) {
            parts[i][j][k].to_receive_from.clear();
            parts[i][j][k].version = latest_version;
          }
        }
      }
    }
  }

  void propagate_if_ready(int layer_id, int blob_id, int part) {
    BlobUpdate update;
    update.mutable_info()->set_layer_id(layer_id);
    update.mutable_info()->set_blob_id(blob_id);
    update.mutable_info()->set_part(part);
    update.mutable_info()->set_version(
      parts[layer_id][blob_id][part].version);
    update.set_iters(parts[layer_id][blob_id][part].iters);
    propagate_if_ready(&parts[layer_id][blob_id][part], &update);
  }

  void propagate_if_ready(PartInfo* part, BlobUpdate* msg) {
    DLOG(INFO) << msg->info().layer_id() << " "
      << msg->info().blob_id() << " "
      << msg->info().part() << " "
      << " waiting for : " << part->to_receive_from.size()
      << " at version : " << part->version;
    if (part->to_receive_from.empty()) {
      msg->set_iters(clients.size());
      codec->encode(msg, part->blob, BlobEncoding::GRADS, msg->info().part());
      shared_ptr<string> serialized =
        boost::make_shared<string>(serialize(*msg));
      up_waypoint->async_send(serialized->c_str(), serialized->size(),
        boost::bind(&Impl::sent_str, this, serialized));
      typedef Clients::iterator It;
      part->iters = 0;
      for (It it = clients.begin(); it != clients.end(); ++it) {
        part->to_receive_from.insert(it->first);
        part->iters++;
      }
      for (It it = pending.begin(); it != pending.end(); ++it) {
        part->to_receive_from.insert(it->first);
        part->iters++;
      }
    }
  }

  void received_as_server(char* buffer, size_t size, RemoteId from) {
    BlobUpdate msg;
    if (!deserialize(buffer, size, &msg)) {
      LOG(ERROR) << "deserialize failed";
      return;
    }
    if (!msg.has_info()) {
      LOG(ERROR) << "msg has no info";
      return;
    }

    PartInfo& part =
      parts[msg.info().layer_id()][msg.info().blob_id()][msg.info().part()];
    Dtype old_multiplier = 1.0;
    if (part.version < msg.info().version()) {
      part.version = msg.info().version();
      old_multiplier = 0.0;
    }
    DLOG(INFO) << "Received from " << from << ": "
      << msg.info().layer_id() << " "
      << msg.info().blob_id() << " "
      << msg.info().part() << " "
      << "of version " << msg.info().version()
      << " waiting on " << part.to_receive_from.size()
      << ": " << (part.to_receive_from.empty() ? "-"
              : boost::lexical_cast<string>(*part.to_receive_from.begin()));

    typedef ClientSet::iterator It;
    It it = part.to_receive_from.find(from);
    if (it == part.to_receive_from.end()) {
      DLOG(INFO) << "already received";
      return;
    } else {
      part.to_receive_from.erase(it);
    }
    if (!codec->decode(
      msg, part.blob, BlobEncoding::GRADS, 1.0, old_multiplier)) {
      LOG(ERROR) << "decoding failed";
      return;
    }

    propagate_if_ready(&part, &msg);
  }

  void received_as_client(char* buffer, size_t size) {
    shared_ptr<std::vector<char> > copy =
      boost::make_shared<std::vector<char> >(buffer, buffer + size);
    BlobUpdate msg;
    if (!deserialize(buffer, size, &msg)) {
      LOG(ERROR) << "deserialize failed";
      return;
    }
    PartInfo& part =
      parts[msg.info().layer_id()][msg.info().blob_id()][msg.info().part()];
    part.last_updated = copy;
    latest_version = std::max(msg.info().version(), latest_version);
    part.last_updated_version = msg.info().version();

    down_waypoint->async_send(
      &copy->front(), copy->size(), boost::bind(&Impl::sent_vec, this, copy));
  }

  void sent_vec(shared_ptr<vector<char> >) const {
  }
  void sent_str(shared_ptr<string>) const {
  }

 public:
  Impl(shared_ptr<Solver<Dtype> > solver,
       string bind_address,
       string param_server_address)
    : comm(internode::create_communication_daemon())
    , codec(BlobCodec<Dtype>::create_codec(
        solver->param().multinode_param(), false))
    , down_waypoint(
        internode::configure_server(comm, bind_address, codec->packet_size()))
    , up_waypoint(
        internode::configure_client(
          comm, param_server_address, codec->packet_size()))
    , solver(solver)
    , info(BlobInfo<Dtype>(
        solver, codec->max_elements_per_part()).get_const_info())
    , latest_version(0)
    , server_handler(this)
    , client_handler(this) {
    down_waypoint->register_peer_change_handler(this);
    down_waypoint->register_receive_handler(&server_handler);
    up_waypoint->register_receive_handler(&client_handler);

    parts.resize(info->layers());
    for (int i = 0; i < info->layers(); ++i) {
      parts[i].resize(info->blobs(i));
      for (int j = 0; j < info->blobs(i); ++j) {
        parts[i][j].resize(info->parts(i, j));
        for (int k = 0; k < info->parts(i, j); ++k) {
          parts[i][j][k].version = 1;
          parts[i][j][k].blob = solver->net()->layers()[i]->blobs()[j].get();
        }
      }
    }
  }

  void run() {
    LOG(INFO) << "relay running";
    while (solver->GetRequestedAction() == SolverAction::NONE) {
      internode::poll_one(comm);
    }
  }
};

template <typename Dtype>
ParamRelay<Dtype>::ParamRelay(shared_ptr<Solver<Dtype> > solver,
                              string bind_address,
                              string param_server_address,
                              int)
  : impl(boost::make_shared<Impl>(solver, bind_address, param_server_address)) {
}

template <typename Dtype>
void ParamRelay<Dtype>::run() {
  impl->run();
}

INSTANTIATE_CLASS(ParamRelay);

}  // namespace caffe

