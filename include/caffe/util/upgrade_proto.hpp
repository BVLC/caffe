// Copyright 2014 Jeff Donahue

#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "google/protobuf/message.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/deprecated/caffe_v0_to_v1_bridge.pb.h"

#include "boost/scoped_ptr.hpp"
#include "caffe/blob.hpp"

using std::string;

namespace caffe {

bool UpgradeV0Net(const V0NetParameter& v0_net_param, NetParameter* net_param);

bool UpgradeV0LayerConnection(const V0LayerConnection& v0_layer_connection,
                              LayerParameter* layer_param);

}  // namespace caffe

#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
