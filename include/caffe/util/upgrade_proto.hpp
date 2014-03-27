// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/deprecated/caffe_v0_to_v1_bridge.pb.h"

using std::string;

namespace caffe {

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const V0NetParameter& v0_net_param, NetParameter* net_param);

// Upgrade V0NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const V0NetParameter& param,
                            V0NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the new LayerParameter format.
bool UpgradeV0LayerConnection(const V0LayerConnection& v0_layer_connection,
                              LayerParameter* layer_param);

LayerParameter_LayerType UpgradeV0LayerType(const string& type);

}  // namespace caffe

#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
