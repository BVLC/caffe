#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

namespace caffe {

void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layer();
  map<string, pair<int, int> > blob_name_to_last_top_idx;
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
  map<pair<int, int>, int> top_idx_to_bottom_count;
  map<pair<int, int>, float> top_idx_to_loss_weight;
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  map<int, string> layer_idx_to_layer_name;
  layer_idx_to_layer_name[-1] = "input";
  // Determine the number of times each blob is used as an input (bottom) blob.
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    blob_name_to_last_top_idx[blob_name] = make_pair(-1, i);
  }
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      const string& blob_name = layer_param.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
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
    // A use of a top blob as a loss should be handled similarly to the use of
    // a top blob as an input (bottom) blob to another layer.
    const int last_loss =
        std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const string& blob_name = layer_param.top(j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      if (top_idx_to_loss_weight[top_idx]) {
        ++top_idx_to_bottom_count[top_idx];
      }
    }
  }
  // Create split layer for any input blobs used by other layer as bottom
  // blobs more than once.
  for (int i = 0; i < param.input_size(); ++i) {
    const int split_count = top_idx_to_bottom_count[make_pair(-1, i)];
    if (split_count > 1) {
      const string& layer_name = layer_idx_to_layer_name[-1];
      const string& blob_name = param.input(i);
      LayerParameter* split_layer_param = param_split->add_layer();
      const float kZeroLossWeight = 0;
      ConfigureSplitLayer(layer_name, blob_name, i, split_count,
          kZeroLossWeight, split_layer_param);
    }
  }
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layer();
    layer_param->CopyFrom(param.layer(i));
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
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.
    for (int j = 0; j < layer_param->top_size(); ++j) {
      const pair<int, int>& top_idx = make_pair(i, j);
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string& layer_name = layer_idx_to_layer_name[i];
        const string& blob_name = layer_param->top(j);
        LayerParameter* split_layer_param = param_split->add_layer();
        const float loss_weight = top_idx_to_loss_weight[top_idx];
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            loss_weight, split_layer_param);
        if (loss_weight) {
          layer_param->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }
}

void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
    if (loss_weight) {
      if (k == 0) {
        split_layer_param->add_loss_weight(loss_weight);
      } else {
        split_layer_param->add_loss_weight(0);
      }
    }
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
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
