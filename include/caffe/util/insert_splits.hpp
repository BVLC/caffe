// Copyright 2014 BVLC and contributors.

#ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
#define _CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

using std::pair;
using std::string;

namespace caffe {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
void InsertSplits(const NetParameter& param, NetParameter* param_split);

void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count,
    LayerParameter* split_layer_param);

string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx);

string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
