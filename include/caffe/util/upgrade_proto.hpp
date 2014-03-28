// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_pretty_print.pb.h"

using std::string;

namespace caffe {

// Return true iff any layer contains parameters specified using
// deprecated V0LayerParameter.
bool NetNeedsUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a V0NetParameter into a
// NetParameter (including upgrading padding layers and LayerParameters).
bool UpgradeV0Net(const NetParameter& v0_net_param, NetParameter* net_param);

// Upgrade NetParameter with padding layers to pad-aware conv layers.
// For any padding layer, remove it and put its pad parameter in any layers
// taking its top blob as input.
// Error if any of these above layers are not-conv layers.
void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad);

// Upgrade a single V0LayerConnection to the new LayerParameter format.
bool UpgradeLayerParameter(const LayerParameter& v0_layer_connection,
                           LayerParameter* layer_param);

LayerParameter_LayerType UpgradeV0LayerType(const string& type);

// Convert a NetParameter to NetParameterPrettyPrint used for dumping to
// proto text files.
void NetParameterToPrettyPrint(const NetParameter& param,
                               NetParameterPrettyPrint* pretty_param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param);

}  // namespace caffe

#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
