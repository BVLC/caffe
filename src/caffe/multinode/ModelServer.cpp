#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <vector>
#include "caffe/multinode/ModelServer.hpp"

namespace caffe {

using ::google::protobuf::Message;
using internode::create_communication_daemon;
using internode::RemoteId;
using internode::configure_server;

template <typename Dtype>
ModelServer<Dtype>::ModelServer(shared_ptr<Solver<Dtype> > solver,
                                string bind_address)
  : daemon(create_communication_daemon())
  , solver(solver)
  , param_(prepare_model())
  , waypoint(configure_server(daemon, bind_address)) {
  waypoint->register_receive_handler(this);
  LOG(INFO) << param_.DebugString();
}


template <typename DType>
BlobShape ModelServer<DType>::blob_shape_by_name(string name) {
  const vector<int>& shape = solver->net()->blob_by_name(name)->shape();
  BlobShape ret;
  for (uint32_t i = 0; i < shape.size(); ++i) {
    ret.add_dim(shape[i]);
  }
  return ret;
}

template <typename Dtype>
SolverParameter ModelServer<Dtype>::prepare_model() {
  NetParameter net;
  solver->net()->ToProto(&net);

  for (int i = 0; i < net.layer_size(); ++i) {
    LayerParameter& layer = *net.mutable_layer(i);
    layer.clear_blobs();
    if ((layer.type().find("Data") != std::string::npos)
        && (layer.has_remote_data_param())) {
      layer.set_type("RemoteData");

      for (int j = 0; j < layer.top_size(); ++j) {
        *layer.mutable_remote_data_param()->add_shape()
          = blob_shape_by_name(layer.top(j));
      }
    }
  }

  SolverParameter ret = solver->param();
  ret.clear_net();
  ret.clear_net_param();
  ret.clear_test_net();
  ret.clear_test_net_param();
  ret.clear_train_net();
  *ret.mutable_train_net_param() = net;

  return ret;
}

template <typename DType>
void ModelServer<DType>::received(char* data, size_t size, RemoteId from) {
  ModelReq req;
  bool ret = req.ParseFromArray(data, size);
  LOG(INFO) << "received message of size " << size;
  if (!ret) {
    LOG(ERROR) << "model request parsing failed, ignoring";
    return;
  }

  if (req.name() != param_.GetTypeName()) {
    LOG(ERROR) << "model request for something else than SolverParam: "
               << req.name() << ", ignoring";
    return;
  }

  string str;
  param_.SerializeToString(&str);

  waypoint->send_to(str.c_str(), str.size(), from);
}

template <typename Dtype>
void ModelServer<Dtype>::run() {
  while (true) {
    internode::poll_one(daemon);

    SolverAction::Enum action = solver->GetRequestedAction();
    if (action == SolverAction::STOP) {
      break;
    } else if (action == SolverAction::SNAPSHOT) {
      solver->Snapshot();
    }
  }
}

INSTANTIATE_CLASS(ModelServer);
}  // namespace caffe

