#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <vector>
#include "caffe/multinode/DataServer.hpp"

namespace caffe {

using ::google::protobuf::Message;
using internode::RemoteId;
using internode::create_communication_daemon;
using internode::configure_server;

template <typename Dtype>
DataServer<Dtype>::DataServer(shared_ptr<Solver<Dtype> > solver,
                                string bind_address)
  : daemon(create_communication_daemon())
  , solver(solver)
  , waypoint(configure_server(daemon, bind_address)) {
  waypoint->register_receive_handler(this);
  LOG(INFO) << solver->param().DebugString();
}

template <typename Dtype>
void DataServer<Dtype>::received(char* buffer, size_t size, RemoteId id) {
  DataReq req;
  if (!req.ParseFromArray(buffer, size)) {
    LOG(ERROR) << "parsing data request failed";
  }
  VLOG(2) << "received data request for "
    << req.layer_name() << " (iters: " << req.iters() << ")";

  Net<Dtype>& net = *solver->net();
  const vector<shared_ptr<Layer<Dtype> > >& layers = net.layers();

  int layer_id = -1;
  for (int i = 0; i < layers.size(); ++i) {
    if (layers[i]->layer_param().name() == req.layer_name()) {
      layer_id = i;
    }
  }
  if (layer_id < 0) {
    LOG(ERROR) << "requested data from layer, that doesn't exist: "
               << req.layer_name();
    return;
  }

  DataMsg msg;

  net.ForwardFromTo(layer_id, layer_id);
  vector<Blob<Dtype>*> top_vecs = net.top_vecs()[layer_id];
  for (int j = 0; j < top_vecs.size(); ++j) {
    Blob<Dtype>& blob = *top_vecs[j];
    msg.add_data(blob.cpu_data(), blob.count() * sizeof(Dtype));

    const vector<int>& dims = blob.shape();
    BlobShape* shape = msg.add_shape();
    for (uint32_t i = 0; i < dims.size(); ++i) {
      shape->add_dim(dims[i]);
    }
  }

  string str;
  msg.SerializeToString(&str);
  VLOG(2) << "sending data to " << id;
  waypoint->send_to(str.c_str(), str.size(), id);
}

template <typename Dtype>
void DataServer<Dtype>::run() {
  while (true) {
    try { internode::poll_one(daemon); } catch (...) {}

    SolverAction::Enum action = solver->GetRequestedAction();
    if (action == SolverAction::STOP) {
      break;
    } else if (action == SolverAction::SNAPSHOT) {
      solver->Snapshot();
    }
  }
}

INSTANTIATE_CLASS(DataServer);
}  // namespace caffe

