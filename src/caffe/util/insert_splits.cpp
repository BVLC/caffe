// Copyright 2014 Jeff Donahue

#include <fstream>
#include <map>
#include <string>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

using std::map;

namespace caffe {

void insert_splits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layers();
  map<string, int> blob_name_to_bottom_count;
  map<string, int> blob_name_to_bottom_split_idx;
  // Determine for each top blob (including input blobs) the number of times
  // it's used as a bottom blob.
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    blob_name_to_bottom_count[blob_name] = 0;
    blob_name_to_bottom_split_idx[blob_name] = 0;
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerConnection& layer_connection = param.layers(i);
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      blob_name_to_bottom_count[blob_name]++;
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      blob_name_to_bottom_count[blob_name] = 0;
      blob_name_to_bottom_split_idx[blob_name] = 0;
    }
  }
  // Create split layer for any input blobs user by other layers as bottom
  // blobs more than once.
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    const int split_count = blob_name_to_bottom_count[blob_name];
    if (split_count > 1) {
      LayerConnection* split_layer_connection = param_split->add_layers();
      configure_split_layer(blob_name, split_count, split_layer_connection);
    }
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    LayerConnection* layer_connection = param_split->add_layers();
    layer_connection->CopyFrom(param.layers(i));
    // Replace any shared bottom blobs with split layer outputs.
    for (int j = 0; j < layer_connection->bottom_size(); ++j) {
      const string& blob_name = layer_connection->bottom(j);
      const int split_count = blob_name_to_bottom_count[blob_name];
      if (split_count > 1) {
        string split_blob_name;
        get_split_blob_name(blob_name,
            blob_name_to_bottom_split_idx[blob_name]++, &split_blob_name);
        layer_connection->set_bottom(j, split_blob_name);
      }
    }
    // Create split layer for any top blobs used by other layers as bottom
    // blobs more than once.
    for (int j = 0; j < layer_connection->top_size(); ++j) {
      const string& blob_name = layer_connection->top(j);
      const int split_count = blob_name_to_bottom_count[blob_name];
      if (split_count > 1) {
        LayerConnection* split_layer_connection = param_split->add_layers();
        split_layer_connection->add_bottom(blob_name);
        configure_split_layer(blob_name, split_count, split_layer_connection);
      }
    }
  }
}

void configure_split_layer(const string& blob_name,
    const int split_count, LayerConnection* split_layer_connection) {
  split_layer_connection->Clear();
  split_layer_connection->add_bottom(blob_name);
  LayerParameter* split_layer_param =
      split_layer_connection->mutable_layer();
  split_layer_param->set_name(blob_name + "_split");
  split_layer_param->set_type("split");
  for (int k = 0; k < split_count; ++k) {
    string split_blob_name;
    get_split_blob_name(blob_name, k, &split_blob_name);
    split_layer_connection->add_top(split_blob_name);
  }
}

void get_split_blob_name(const string& blob_name, const int split_index,
    string* split_blob_name) {
  const int suffix_max_length = 16;
  char split_suffix[suffix_max_length];
  const int suffix_length = snprintf(split_suffix, suffix_max_length,
      "_split_%d", split_index);
  CHECK_LT(suffix_length, suffix_max_length);
  *split_blob_name = blob_name + split_suffix;
}

}  // namespace caffe
