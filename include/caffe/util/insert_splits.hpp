// Copyright 2014 Jeff Donahue

#ifndef _CAFFE_UTIL_INSERT_SPLITS_HPP_
#define _CAFFE_UTIL_INSERT_SPLITS_HPP_

#include "caffe/proto/caffe.pb.h"

using std::pair;
using std::string;

namespace caffe {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
void insert_splits(const NetParameter& param, NetParameter* param_split);

void configure_split_layer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count,
    LayerConnection* split_layer_connection);

string get_split_layer_name(const string& layer_name, const string& blob_name,
    const int blob_idx);

string get_split_blob_name(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
