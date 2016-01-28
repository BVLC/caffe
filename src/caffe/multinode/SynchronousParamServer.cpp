#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include "caffe/multinode/SynchronousParamServer.hpp"
#include "caffe/serialization/BlobCodec.hpp"
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

using internode::configure_server;
using internode::create_communication_daemon;
using internode::RemoteId;

template <typename Dtype>
SynchronousParamServer<Dtype>::SynchronousParamServer(
        shared_ptr<Solver<Dtype> > solver, string bind_address)
    : daemon(create_communication_daemon())
    , waypoint(configure_server(daemon, bind_address))
    , solver(solver)
    , blob_version(solver->net()->layers().size())
    , blob_iters(solver->net()->layers().size())
    , pending_clients(solver->net()->layers().size())
    , codec(BlobCodec<Dtype>::create_codec(solver->param().multinode_param())) {
  for (int i = 0; i < blob_version.size(); ++i) {
    blob_version[i].resize(solver->net()->layers()[i]->blobs().size());
    blob_iters[i].resize(blob_version[i].size());
    pending_clients[i].resize(blob_version[i].size());
  }

  waypoint->register_receive_handler(this);
  waypoint->register_peer_change_handler(this);
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::run() {
  LOG(INFO) << "param server running";
  while (true) {
    internode::poll_one(daemon);

    SolverAction::Enum action = solver->GetRequestedAction();
    if (action == SolverAction::STOP) {
      solver->Snapshot();
      break;
    } else if (action == SolverAction::SNAPSHOT) {
      solver->Snapshot();
    }
  }
}

template <typename Dtype>
int SynchronousParamServer<Dtype>::current_iter() const {
  int ret = 0;
  for (int i = 0; i < blob_version.size(); ++i) {
    for (int j = 0; j < blob_version[i].size(); ++j) {
      ret = std::max(blob_version[i][j], ret);
    }
  }
  return ret;
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::init_client(internode::RemoteId id) {
  int curr_iter = current_iter();
  version_sent[id].resize(blob_version.size());
  for (int i = 0; i < blob_version.size(); ++i) {
    version_sent[id][i].resize(blob_version[i].size());
    for (int j = 0; j < blob_version[i].size(); ++j) {
      if ((blob_version[i][j] < curr_iter)
           & (!solver->net()->get_layer_learnable_param_ids(i).empty()))
        continue;

      Layer<Dtype>& layer = *solver->net()->layers()[i];
      Blob<Dtype>* blob = layer.blobs()[j].get();

      BlobUpdate update;
      update.set_layer_id(i);
      update.set_blob_id(j);
      update.set_version(blob_version[i][j]);
      CHECK(codec->encode(
        &update, blob, BlobCodec<Dtype>::PARAMS, 0) == blob->count());
      string str = serialize(update);
      VLOG(4)
        << "sending blob " << j << " of layer " << i;
      waypoint->send_to(str.c_str(), str.size(), id);
      VLOG(2)
        << "sent blob " << j << " of layer " << i;

      version_sent[id][i][j] = blob_version[i][j];
      if (!solver->net()->get_layer_learnable_param_ids(i).empty()) {
        pending_clients[i][j].insert(id);
      }
    }
  }
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::accepted(internode::RemoteId id) {
  all_clients.insert(id);

  if (all_clients.size()
      >= solver->param().multinode_param().wait_for_clients()) {
    typedef ClientSet::iterator It;
    for (It it = all_clients.begin(); it != all_clients.end(); ++it) {
      init_client(*it);
    }
  }

  LOG(INFO) << "accepted client " << id << " at iter: " << current_iter();
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::disconnected(internode::RemoteId id) {
  all_clients.erase(id);
  for (int i = 0; i < pending_clients.size(); ++i) {
    for (int j = 0; j < blob_version[i].size(); ++j) {
      pending_clients[i][j].erase(id);
      version_sent.erase(id);

      VLOG(2) << "layer " << i << ", blob " << j
        <<  " at iter: " << blob_version[i][j]
        << " and " << pending_clients[i][j].size() << " clients waiting";
    }
    upgrade_layer(i);
  }
  if (all_clients.empty() && !all_layers_synced()) {
    for (int i = 0; i < pending_clients.size(); ++i) {
      for (int j = 0; j < pending_clients[i].size(); ++j) {
        blob_version[i][j] = current_iter();
      }
    }
  }
  LOG(INFO) << "client disconnected " << id;
  update_clients();
}

template <typename Dtype>
bool SynchronousParamServer<Dtype>::all_layers_synced() const {
  int curr_iter = current_iter();
  for (int i = 0; i < solver->net()->layers().size(); ++i) {
    for (int j = 0; j < blob_version[i].size(); ++j) {
      if (!solver->net()->get_layer_learnable_param_ids(i).empty()
          && (blob_version[i][j] != curr_iter)) {
        return false;
      }
    }
  }
  return true;
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::upgrade_layer(int layer_id) {
  Net<Dtype>& net = *solver->net();

  vector<int> param_ids = net.get_layer_learnable_param_ids(layer_id);
  if (param_ids.empty()) return;
  if (blob_iters[layer_id].size() == 0) return;

  int current_version = blob_version[layer_id][0];
  int iter_size = blob_iters[layer_id][0];
  for (int j = 0; j < pending_clients[layer_id].size(); ++j) {
    VLOG(5) << "blob " << j << " of layer " << layer_id << " state is: "
      << "clients waiting: " << pending_clients[layer_id][j].size() << ", "
      << "version: " << blob_version[layer_id][j] << ", "
      << "iter_size: " << blob_iters[layer_id][j];
    if (!pending_clients[layer_id][j].empty()) return;
    if (blob_iters[layer_id][j] == 0) return;
  }

  solver->set_iter(current_version);
  solver->param().set_iter_size(iter_size);
  for (int j = 0; j < param_ids.size(); ++j) {
    solver->ApplyUpdate(param_ids[j]);
    net.ClearParamDiffs(param_ids[j]);
  }
  VLOG(2) << "layer " << layer_id << " version " << current_version << " ready";

  for (int j = 0; j < blob_version[layer_id].size(); ++j) {
    blob_version[layer_id][j]++;
    pending_clients[layer_id][j] = all_clients;
    blob_iters[layer_id][j] = 0;
  }

  if ((solver->param().test_interval() > 0)
      && ((blob_version[layer_id][0] - 1) % solver->param().test_interval()
          == 0)
      && all_layers_synced()
      && (((blob_version[layer_id][0] - 1) > 0)
          || (solver->param().test_initialization()))) {
    solver->TestAll();
  }
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::received(
      char* data, size_t size, RemoteId remote_id) {
  BlobUpdate msg;
  if (!deserialize(data, size, &msg))  return;

  Blob<Dtype>* blob =
    solver->net()->layers()[msg.layer_id()]->blobs()[msg.blob_id()].get();
  if (!codec->decode(msg, blob, BlobCodec<Dtype>::GRADS, 1.0, 1.0)) return;

  VLOG(2) << "received gradients from " << remote_id << ": {"
    << "version: " << msg.version()
    << ", layer: " << msg.layer_id()
    << ", blob: " << msg.blob_id() << "}";

  blob_iters[msg.layer_id()][msg.blob_id()] += msg.iters();
  pending_clients[msg.layer_id()][msg.blob_id()].erase(remote_id);
  upgrade_layer(msg.layer_id());
  update_clients();
}

template <typename Dtype>
void SynchronousParamServer<Dtype>::update_clients() {
  for (int i = 0; i < blob_version.size(); ++i) {
    for (int j = 0; j < blob_version[i].size(); ++j) {
      typedef ClientSet::iterator It;
      std::vector<RemoteId> clients_to_update;
      ClientSet& clients = pending_clients[i][j];

      for (It it = clients.begin(); it != clients.end(); ++it) {
        if (version_sent[*it][i][j] < blob_version[i][j]) {
          clients_to_update.push_back(*it);
          version_sent[*it][i][j] = blob_version[i][j];
          VLOG(3) << "client " << *it
                  << " will be updated with blob " << j << " of layer " << i
                  << " in version " << blob_version[i][j];
        }
      }

      if (clients_to_update.empty()) continue;

      Layer<Dtype>& layer = *solver->net()->layers()[i];
      Blob<Dtype>* blob = layer.blobs()[j].get();

      BlobUpdate update;
      update.set_layer_id(i);
      update.set_blob_id(j);
      update.set_version(blob_version[i][j]);
      CHECK(codec->encode(
        &update, blob, BlobCodec<Dtype>::PARAMS, 0) == blob->count());
      Dtype sum = 0.0;
      for (int k = 0; k < blob->count(); ++k) {
        sum += blob->cpu_data()[k];
      }
      string str = serialize(update);
      VLOG(4) << "sending blob " << j << " of layer " << i;
      if (clients_to_update.size() == all_clients.size()) {
        waypoint->send(str.c_str(), str.size());
      } else {
        for (int k = 0; k < clients_to_update.size(); ++k) {
          waypoint->send_to(str.c_str(), str.size(), clients_to_update[k]);
        }
      }
      VLOG(2) << "sent blob " << j << " of layer " << i;
    }
  }
}

INSTANTIATE_CLASS(SynchronousParamServer);

}  // namespace caffe

