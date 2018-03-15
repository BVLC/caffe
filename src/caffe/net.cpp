#include <algorithm>
#include <chrono>
#ifndef CPU_ONLY
#include <cuda_profiler_api.h>
#endif
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype> Net<Dtype>::Net(const NetParameter &param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string &param_file, Phase phase, const int level,
                const vector<string> *stages) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

template <typename Dtype> void Net<Dtype>::Init(const NetParameter &in_param) {

  /*
  size_t free_byte;
  size_t total_byte;

  CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
  std::cout << "init free_byte=" << (free_byte) / 1024 / 1024 << std::endl;
  */

  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  top_blob_names_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  bottom_blob_names_.resize(param.layer_size());
  // param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());

  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter &layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(), layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      AppendBottom(param, layer_id, bottom_id, &available_blobs,
                   &blob_name_to_idx);
    }
    int num_top = layer_param.top_size();

    /*
  {
    size_t cur_free_byte;
    size_t cur_total_byte;
    CUDA_CHECK(cudaMemGetInfo(&cur_free_byte, &cur_total_byte));
      std::cout << "before setup cur_free_byte="
                << (cur_free_byte) / 1024 / 1024 << std::endl;
  }
  */

    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype> *layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);

    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
  }

  /*
  {
    size_t cur_free_byte;
    size_t cur_total_byte;
    CUDA_CHECK(cudaMemGetInfo(&cur_free_byte, &cur_total_byte));
      std::cout << "after setup cur_free_byte="
                << (cur_free_byte) / 1024 / 1024 << std::endl;
  }
  */

  // In the end, all remaining blobs are considered output blobs.
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  LOG(INFO) << "Network initialization done.";

  blobs_.clear();

  /*
  {
    size_t cur_free_byte;
    size_t cur_total_byte;
    CUDA_CHECK(cudaMemGetInfo(&cur_free_byte, &cur_total_byte));
      std::cout << "init cur_free_byte="
                << (cur_free_byte) / 1024 / 1024 << std::endl;
    if (cur_free_byte < free_byte) {
      std::cout << "init use more memory ="
                << (free_byte - cur_free_byte) / 1024 / 1024 << std::endl;
    } else {
      std::cout << "init use less memory ="
                << (cur_free_byte - free_byte) / 1024 / 1024 << std::endl;
    }
  }
  */
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter &param,
                           NetParameter *param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter &layer_param = param.layer(i);
    const string &layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
        << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState &state, const NetStateRule &rule,
                                const string &layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
    if (rule.phase() != state.phase()) {

      LOG(INFO) << "The NetState phase (" << state.phase()
                << ") differed from the phase (" << rule.phase()
                << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG(INFO) << "The NetState level (" << state.level()
                << ") is above the min_level (" << rule.min_level()
                << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG(INFO) << "The NetState level (" << state.level()
                << ") is above the max_level (" << rule.max_level()
                << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) {
        has_stage = true;
      }
    }
    if (!has_stage) {
      LOG(INFO) << "The NetState did not contain stage '" << rule.stage(i)
                << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) {
        has_stage = true;
      }
    }
    if (has_stage) {
      LOG(INFO) << "The NetState contained a not_stage '" << rule.not_stage(i)
                << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }

  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter &param, const int layer_id,
                           const int top_id, set<string> *available_blobs,
                           map<string, int> *blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string &blob_name = (layer_param->top_size() > top_id)
                                ? layer_param->top(top_id)
                                : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    top_blob_names_[layer_id].push_back(blob_name);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    shared_ptr<Blob<Dtype>> blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    if (blob_name_to_idx) {
      (*blob_name_to_idx)[blob_name] = blob_id;
    }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
    top_blob_names_[layer_id].push_back(blob_name);
  }
  if (available_blobs) {
    available_blobs->insert(blob_name);
  }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter &param, const int layer_id,
                             const int bottom_id, set<string> *available_blobs,
                             map<string, int> *blob_name_to_idx) {
  const LayerParameter &layer_param = param.layer(layer_id);
  const string &blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  bottom_blob_names_[layer_id].push_back(blob_name);
  available_blobs->erase(blob_name);

  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  for (int i = start; i <= end; ++i) {
    layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
  }
  return;
}

template <typename Dtype>
std::map<std::string, std::shared_ptr<Blob<Dtype>>> Net<Dtype>::ForwardConst(
    std::map<std::string, std::shared_ptr<Blob<Dtype>>> &input_blobs,
    const std::set<std::string> &output_blob_names, int gpu_no) {

  int end = -1;

  std::map<int, std::vector<std::shared_ptr<Blob<Dtype>>>> bottom_blobs;
  std::map<int, std::vector<std::shared_ptr<Blob<Dtype>>>> top_blobs;
  std::map<std::string, std::shared_ptr<Blob<Dtype>>> output_blobs;

  auto output_blob_names_tmp = output_blob_names;
  for (int i = 0; i < layers_.size(); ++i) {
    for (auto const &blob_name : bottom_blob_names_[i]) {
      auto it = input_blobs.find(blob_name);
      if (it == input_blobs.end()) {
        auto &blob_pointer = input_blobs[blob_name];
        blob_pointer.reset(new Blob<Dtype>());
        bottom_blobs[i].emplace_back(blob_pointer);
      } else {
        bottom_blobs[i].emplace_back(it->second);
      }
    }

    for (auto const &blob_name : top_blob_names_[i]) {
      auto it = input_blobs.find(blob_name);
      if (it == input_blobs.end()) {
        auto &blob_pointer = input_blobs[blob_name];
        blob_pointer.reset(new Blob<Dtype>());
        top_blobs[i].emplace_back(blob_pointer);
      } else {
        top_blobs[i].emplace_back(it->second);
      }

      {
        auto it = output_blob_names_tmp.find(blob_name);
        if (it != output_blob_names_tmp.end()) {
          end = i;
          output_blobs[blob_name] = input_blobs[blob_name];
          output_blob_names_tmp.erase(it);
        }
      }
    }
  }

  if (!output_blob_names_tmp.empty()) {
    LOG(FATAL) << "Unknown output blob " << *(output_blob_names_tmp.begin());
  }

  input_blobs.clear();

  CHECK_GE(end, 0);

  Caffe::set_device(gpu_no);
  auto mode = Caffe::mode();

  for (int i = 0; i <= end; ++i) {
    vector<Blob<Dtype> *> bottom;
    vector<Blob<Dtype> *> top;
    for (auto const &ptr : bottom_blobs[i]) {
      bottom.push_back(ptr.get());
    }

    for (auto const &ptr : top_blobs[i]) {
      top.push_back(ptr.get());
    }

    layers_[i]->Reshape_const(bottom, top);

    switch (mode) {

    case Caffe::CPU:
      layers_[i]->Forward_const_cpu(bottom, top);
      break;
    case Caffe::GPU:
      layers_[i]->Forward_const_gpu(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
    }

    bottom_blobs.erase(i);
    top_blobs.erase(i);
  }

  return output_blobs;
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter &param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter &source_layer = param.layer(i);
    const string &source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
           layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype>>> &target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL)
            << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  CopyTrainedLayersFromBinaryProto(trained_filename);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string &blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype>>
Net<Dtype>::blob_by_name(const string &blob_name) const {
  shared_ptr<Blob<Dtype>> blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype> *)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string &layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype>>
Net<Dtype>::layer_by_name(const string &layer_name) const {
  shared_ptr<Layer<Dtype>> layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype> *)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

} // namespace caffe
