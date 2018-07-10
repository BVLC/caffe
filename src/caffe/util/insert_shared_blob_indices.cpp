#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_shared_blob_indices.hpp"

namespace caffe {

int_tp InsertSharedBlobIndices(const NetParameter& param,
                               NetParameter* shared_memory_net_param) {
  shared_memory_net_param->CopyFrom(param);
  shared_memory_net_param->clear_layer();

  map<string, int_tp> blob_name_to_shared_idx;
  vector<bool> allocated_shared_idxs;

  for (int_tp i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    LayerParameter* new_layer_param = shared_memory_net_param->add_layer();
    new_layer_param->CopyFrom(layer_param);

    for (int_tp j = 0; j < layer_param.top_size(); ++j) {
      const string& blob_name = layer_param.top(j);
      if (blob_name_to_shared_idx.find(blob_name) !=
          blob_name_to_shared_idx.end()) {
        // Deal with in-place cases (bottom == top)
        int_tp idx = blob_name_to_shared_idx[blob_name];
        new_layer_param->add_top_shared_index(idx);
        continue;
      }
      bool found_unallocated_shared_idx = false;
      for (int_tp k = 0; k < allocated_shared_idxs.size(); ++k) {
        if (not allocated_shared_idxs[k]) {
          allocated_shared_idxs[k] = true;
          new_layer_param->add_top_shared_index(k);
          blob_name_to_shared_idx[blob_name] = k;
          // Allocate existing shared index
          found_unallocated_shared_idx = true;
          break;
        }
      }
      if (not found_unallocated_shared_idx) {
        // Create new shared index
        allocated_shared_idxs.push_back(true);
        new_layer_param->add_top_shared_index(allocated_shared_idxs.size() - 1);
        blob_name_to_shared_idx[blob_name] = allocated_shared_idxs.size() - 1;
      }
    }

    for (int_tp j = 0; j < layer_param.bottom_size(); ++j) {
      const string& blob_name = layer_param.bottom(j);
      if (blob_name_to_shared_idx.find(blob_name) ==
          blob_name_to_shared_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      int_tp idx = blob_name_to_shared_idx[blob_name];
      new_layer_param->add_bottom_shared_index(idx);
      // Deal with in-place cases (bottom == top)
      bool in_place = false;
      for (int_tp k = 0; k < layer_param.top_size(); ++k) {
        if (blob_name == layer_param.top(k)) {
          in_place = true;
        }
      }
      // Unallocate shared index
      if (!in_place) {
        allocated_shared_idxs[idx] = false;
      }
    }
  }

  // Return number of required shared buffers
  return allocated_shared_idxs.size();
}

}  // namespace caffe
