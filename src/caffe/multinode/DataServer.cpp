/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <string>
#include <vector>
#include "caffe/multinode/DataServer.hpp"
#include "caffe/multinode/SendCallback.hpp"

namespace caffe {

using ::google::protobuf::Message;
using internode::RemoteId;
using internode::Waypoint;

template <typename Dtype>
DataServer<Dtype>::DataServer(shared_ptr<Solver<Dtype> > solver,
                              string bind_address,
                              string,
                              int)
  : daemon(internode::create_communication_daemon())
  , solver(solver)
  , waypoint(internode::configure_server(daemon, bind_address, UINT_MAX)) {
  waypoint->register_receive_handler(this);
  LOG(INFO) << solver->param().DebugString();
}

template <typename Dtype>
void DataServer<Dtype>::received(char* buffer,
                                 size_t size,
                                 Waypoint* remote) {
  DataReq req;
  if (!req.ParseFromArray(buffer, size)) {
    LOG(ERROR) << "parsing data request failed";
  }
  DLOG(INFO) << "received data request for "
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


  SendCallback callback;
  msg.SerializeToString(callback.buffer.get());
  DLOG(INFO) << "sending data to " << remote->id();

  remote->async_send(
    callback.buffer->c_str(), callback.buffer->size(), callback);
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

