// Copyright 2014 BVLC and contributors.

#include <map>
#include <string>
#include <sstream>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

using std::map;
using std::ostringstream;
using std::pair;
using std::make_pair;

namespace caffe {

void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layers();
  map<string, pair<int, int> > blob_name_to_last_top_idx;
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
  map<pair<int, int>, int> top_idx_to_bottom_count;
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  map<int, string> layer_idx_to_layer_name;
  layer_idx_to_layer_name[-1] = "input";
  // Determine the number of times each blob is used as an input (bottom) blob.
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    blob_name_to_last_top_idx[blob_name] = make_pair(-1, i);
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerParameter& layer_param = param.layers(i);
    layer_idx_to_layer_name[i] = layer_param.name();
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      const string& blob_name = layer_param.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown blob input " << blob_name << " to layer " << j;
      }
      const pair<int, int>& bottom_idx = make_pair(i, j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
      ++top_idx_to_bottom_count[top_idx];
    }
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
    }
  }
  // Create split layer for any input blobs used by other layers as bottom
  // blobs more than once.
  for (int i = 0; i < param.input_size(); ++i) {
    const int split_count = top_idx_to_bottom_count[make_pair(-1, i)];
    if (split_count > 1) {
      const string& layer_name = layer_idx_to_layer_name[-1];
      const string& blob_name = param.input(i);
      LayerParameter* split_layer_param = param_split->add_layers();
      ConfigureSplitLayer(layer_name, blob_name, i, split_count,
          split_layer_param);
    }
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layers();
    layer_param->CopyFrom(param.layers(i));
    // Replace any shared bottom blobs with split layer outputs.
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      const pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string& layer_name = layer_idx_to_layer_name[top_idx.first];
        const string& blob_name = layer_param->bottom(j);
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
      }
    }
    // Create split layer for any top blobs used by other layers as bottom
    // blobs more than once.
    for (int j = 0; j < layer_param->top_size(); ++j) {
      const int split_count = top_idx_to_bottom_count[make_pair(i, j)];
      if (split_count > 1) {
        const string& layer_name = layer_idx_to_layer_name[i];
        const string& blob_name = layer_param->top(j);
        LayerParameter* split_layer_param = param_split->add_layers();
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            split_layer_param);
      }
    }
  }
}

void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type(LayerParameter_LayerType_SPLIT);
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
  }
}

string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}

string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx) {
  // 0th split top blob is given the same name as the bottom blob so that
  // computation is done 'in-place', saving a bit of time and memory.
  if (split_idx == 0) {
    return blob_name;
  }
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
