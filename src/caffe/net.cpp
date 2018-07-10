#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_HDF5
#include "hdf5.h"
#endif  // USE_HDF5

#include "caffe/blob_creator.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_creator.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/insert_conversions.hpp"
#include "caffe/util/insert_shared_blob_indices.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

NetBase::NetBase(Device* device_context) :
  device_(device_context) {

}

void NetBase::set_quant_mode(QuantizerMode quant_mode) {
  quant_mode_ = quant_mode;

  // Update the quantizer mode for all quantizers
  for (int_tp i = 0; i < this->layers().size(); ++i) {
    vector<shared_ptr<QuantizerBase> > quant_base_vec =
        this->layers()[i]->get_all_quantizers();
    for (int_tp j = 0; j < quant_base_vec.size(); ++j) {
      const QuantizerParameter& quant_param = quant_base_vec[j]->quant_param();
      QuantizerParameter quant_param_copy;
      quant_param_copy.CopyFrom(quant_param);
      quant_param_copy.set_mode(quant_mode_);
      quant_base_vec[j]->update_param(quant_param_copy);
    }
  }
}

vector<shared_ptr<QuantizerBase> > NetBase::get_all_quantizers() {
  vector<shared_ptr<QuantizerBase> > all_quant_base_vec;
  for (size_t i = 0; i < this->layers().size(); ++i) {
    vector<shared_ptr<QuantizerBase> > quant_base_vec =
        this->layers()[i]->get_all_quantizers();
    for (size_t j = 0; j < quant_base_vec.size(); ++j) {
      all_quant_base_vec.push_back(quant_base_vec[j]);
    }
  }
  return all_quant_base_vec;
}

template<typename Dtype>
Net<Dtype>::Net(const NetParameter& param, Device* device_context)
    : NetBase(device_context) {
  Init(param);
}

template<typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, Device* device_context,
                const int level, const vector<string>* stages)
    : NetBase(device_context) {
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

template<typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state.
  phase_ = in_param.state().phase();

  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);

  // Create a copy of filtered_param with splits added where necessary.
  NetParameter splitted_param;
  InsertSplits(filtered_param, &splitted_param);
  // Create a copy of splitted_param with type conversions added where
  // necessary.
  NetParameter converted_param;
  InsertConversions(splitted_param, &converted_param);

  if (in_param.reduced_memory_inference() && phase_ == caffe::TEST) {
    NetParameter shared_memory_net_param;
    int_tp num_shared_blobs = InsertSharedBlobIndices(converted_param,
                                                      &shared_memory_net_param);
    for (int_tp i = 0; i < num_shared_blobs; ++i) {
      shared_blobs_.push_back(make_shared<Blob<uint8_t> >(this->device_));
    }
    converted_param = shared_memory_net_param;
  }

  if (Caffe::root_solver()) {
    LOG(INFO) << "Initializing net from parameters: " << std::endl
              << converted_param.DebugString();
  }

  param_ = converted_param;
  param_.set_data_type(proto_data_type<Dtype>());

  // To debug InsertSplits and InsertConversions
  // std::cout << param.DebugString() << std::endl;

  // Basically, build all the layers and set up its connections.
  name_ = param_.name();
  map<string, int_tp> blob_name_to_idx;
  std::set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param_.layer_size());
  top_vecs_.resize(param_.layer_size());
  bottom_id_vecs_.resize(param_.layer_size());
  param_id_vecs_.resize(param_.layer_size());
  top_id_vecs_.resize(param_.layer_size());
  bottom_need_backward_.resize(param_.layer_size());
  for (int_tp layer_id = 0; layer_id < param_.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    if (!param_.layer(layer_id).has_phase()) {
      param_.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& c_layer_param = param_.layer(layer_id);

    LayerParameter layer_param = c_layer_param;

    // Set device
    layer_param.set_device(Caffe::GetDefaultDevice()->id());

    // Set autotuning settings
    if (param_.has_autotune() && !layer_param.has_autotune()) {
      layer_param.set_autotune(param_.autotune());
    }

    // Set data types
    if (!layer_param.has_bottom_data_type()) {
      layer_param.set_bottom_data_type(proto_data_type<Dtype>());
    }

    if (!layer_param.has_compute_data_type()) {
      layer_param.set_compute_data_type(proto_data_type<Dtype>());
    }

    if (!layer_param.has_top_data_type()) {
      layer_param.set_top_data_type(proto_data_type<Dtype>());
    }

    // Set quantizer settings
    if (!layer_param.has_net_quantizer()) {
      QuantizerParameter quant_param;
      quant_param.set_device(layer_param.device());
      quant_param.set_input_data_type(proto_data_type<Dtype>());
      quant_param.set_output_data_type(layer_param.top_data_type());
      layer_param.mutable_net_quantizer()->CopyFrom(quant_param);
    }

    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())<< "propagate_down param must be specified "
      << "either 0 or bottom_size times ";
    }
    layers_.push_back(CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating Layer " << layer_param.name();
    }
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int_tp bottom_id = 0; bottom_id < layer_param.bottom_size();
        ++bottom_id) {
      const int_tp blob_id = AppendBottom(param_, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int_tp num_top = layer_param.top_size();
    for (int_tp top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param_, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int_tp blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    LayerBase* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int_tp needed_num_top = std::max(layer->MinTopBlobs(),
                                          layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param_, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int_tp top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      Dtype layer_loss;
      layer->loss(top_id, &layer_loss);
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer_loss;
      if (Caffe::root_solver()) {
        LOG(INFO) << "Top shape: "
                  << top_vecs_[layer_id][top_id]->shape_string();
      }
      if (layer_loss) {
        if (Caffe::root_solver()) {
          LOG(INFO) << "    with loss weight " << layer_loss;
        }
      }
      memory_used_ += top_vecs_[layer_id][top_id]->byte_count();
    }
    if (Caffe::root_solver()) {
      DLOG(INFO) << "Memory required for data: " << memory_used_ << " B";
    }
    const int_tp param_size = layer_param.param_size();
    const int_tp num_param_blobs = layers_[layer_id]->blobs_size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int_tp param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int_tp param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param_, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int_tp top_id = 0; top_id < top_id_vecs_[layer_id].size();
          ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }

  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire layer
  std::set<string> blobs_under_loss;
  std::set<string> blobs_skip_backp;
  for (int_tp layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int_tp top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      Dtype layer_loss;
      layers_[layer_id]->loss(top_id, &layer_loss);
      if (layer_loss
          || (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int_tp bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
          ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) {
      layer_need_backward_[layer_id] = false;
    }
    for (int_tp bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
        ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param_.force_backward()) {
    for (int_tp layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int_tp bottom_id = 0;
          bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id]
                || layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]]
                || bottom_need_backward_[layer_id][bottom_id];
      }
      for (int_tp param_id = 0; param_id < layers_[layer_id]->blobs_size();
          ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  for (int_tp layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    if (layer_need_backward_[layer_id]) {
      if (Caffe::root_solver()) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      }
    } else {
      if (Caffe::root_solver()) {
        LOG(INFO) << layer_names_[layer_id]
                  << " does not need backward computation.";
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (std::set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "This network produces output " << *it;
    }
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (uint_tp blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (uint_tp layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param_.debug_info();
  if (Caffe::root_solver()) {
    LOG(INFO) << "Network initialization done.";
    LOG(INFO) << "Memory required for data: " << memory_used_ << " B";
  }

  // Set up shared blobs
  if (param_.reduced_memory_inference() && phase_ == caffe::TEST) {
    this->SetUpSharedBlobs();
  }

  // Set quantizer mode
  set_quant_mode(param_.state().quantizer_mode());
}

template<typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
                           NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int_tp i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
        << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int_tp j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int_tp j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template<typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state, const NetStateRule& rule,
                                const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
    if (rule.phase() != state.phase()) {
      if (Caffe::root_solver()) {
        LOG(INFO)<< "The NetState phase (" << state.phase()
        << ") differed from the phase (" << rule.phase()
        << ") specified by a rule in layer " << layer_name;
      }
      return false;
    }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      if (Caffe::root_solver()) {
        LOG(INFO) << "The NetState level (" << state.level()
                  << ") is above the min_level (" << rule.min_level()
                  << ") specified by a rule in layer " << layer_name;
      }
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      if (Caffe::root_solver()) {
        LOG(INFO) << "The NetState level (" << state.level()
                  << ") is above the max_level (" << rule.max_level()
                  << ") specified by a rule in layer " << layer_name;
      }
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int_tp i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int_tp j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) {has_stage = true;}
    }
    if (!has_stage) {
      if (Caffe::root_solver()) {
        LOG(INFO) << "The NetState did not contain stage '" << rule.stage(i)
                  << "' specified by a rule in layer " << layer_name;
      }
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int_tp i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int_tp j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) {has_stage = true;}
    }
    if (has_stage) {
      if (Caffe::root_solver()) {
        LOG(INFO) << "The NetState contained a not_stage '" << rule.not_stage(i)
                  << "' specified by a rule in layer " << layer_name;
      }
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int_tp layer_id,
                           const int_tp top_id, std::set<string>* available_blobs,
                           map<string, int_tp>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name << " (in-place)";
    }
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
      blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<BlobBase> blob_pointer = CreateBlob(device_,
        layer_param->top_data_type());
    const int_tp blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) {
      (*blob_name_to_idx)[blob_name] = blob_id;
    }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) {
    available_blobs->insert(blob_name);
  }
}

// Helper for Net::Init: add a new bottom blob to the net.

template<typename Dtype>
int_tp Net<Dtype>::AppendBottom(const NetParameter& param,
                                const int_tp layer_id, const int_tp bottom_id,
                                std::set<string>* available_blobs,
                                map<string, int_tp>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int_tp blob_id = (*blob_name_to_idx)[blob_name];
  if (Caffe::root_solver()) {
    LOG(INFO) << layer_names_[layer_id] << " <- " << blob_name;
  }
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}

template<typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int_tp layer_id,
                             const int_tp param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int_tp param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int_tp net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blob_bases()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int_tp learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int_tp owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int_tp, int_tp>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int_tp owner_layer_id = owner_index.first;
    const int_tp owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    BlobBase* this_blob = layers_[layer_id]->blob_bases()[param_id]
        .get();
    BlobBase* owner_blob = layers_[owner_layer_id]->blob_bases()[owner_param_id]
        .get();
    const int_tp param_size = layer_param.param_size();
    if (param_size > param_id
        && (layer_param.param(param_id).share_mode()
            == ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }

    const int_tp learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template<typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int_tp start, int_tp end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int_tp i = start; i <= end; ++i) {
    for (int_tp c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss;
    // std::cout << layer_names()[i] << std::endl;
    layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i],
                        static_cast<void*>(&layer_loss));
#ifndef NDEBUG
    this->device_->FinishQueues();
#endif  // NDEBUG
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
    for (int_tp c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

template<typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int_tp start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template<typename Dtype>
Dtype Net<Dtype>::ForwardTo(int_tp end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<BlobBase*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template<typename Dtype>
void Net<Dtype>::BackwardFromTo(int_tp start, int_tp end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int_tp i = start; i >= end; --i) {
    for (int_tp c = 0; c < before_backward_.size(); ++c) {
      before_backward_[c]->run(i);
    }
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(top_vecs_[i], bottom_need_backward_[i],
                           bottom_vecs_[i]);
      if (debug_info_) {
        BackwardDebugInfo(i);
      }
    }
    for (int c = 0; c < after_backward_.size(); ++c) {
      after_backward_[c]->run(i);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int_tp layer_id) {
  for (int_tp top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const BlobBase& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    Dtype data_abs_val_mean;
    blob.asum_data(static_cast<void*>(&data_abs_val_mean));
    data_abs_val_mean /= blob.count();
    if (Caffe::root_solver()) {
      LOG(INFO) << "    [Forward] "
                << "Layer " << layer_names_[layer_id]
                << ", top blob " << blob_name
                << " data: " << data_abs_val_mean;
    }
  }
  for (int_tp param_id = 0; param_id < layers_[layer_id]->blobs_size();
      ++param_id) {
    const BlobBase& blob = *layers_[layer_id]->blob_bases()[param_id];
    const int_tp net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    Dtype data_abs_val_mean;
    blob.asum_data(static_cast<void*>(&data_abs_val_mean));
    data_abs_val_mean /= blob.count();
    if (Caffe::root_solver()) {
      LOG(INFO) << "    [Forward] "
                << "Layer " << layer_names_[layer_id]
                << ", param blob " << blob_name
                << " data: " << data_abs_val_mean;
    }
  }
}

template<typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int_tp layer_id) {
  const vector<BlobBase*>& bottom_vec = bottom_vecs_[layer_id];
  for (int_tp bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) {
      continue;
    }
    const BlobBase& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    Dtype diff_abs_val_mean;
    blob.asum_diff(static_cast<void*>(&diff_abs_val_mean));
    diff_abs_val_mean /= blob.count();
    if (Caffe::root_solver()) {
      LOG(INFO) << "    [Backward] "
                << "Layer " << layer_names_[layer_id]
                << ", bottom blob " << blob_name
                << " diff: " << diff_abs_val_mean;
    }
  }
  for (int_tp param_id = 0; param_id < layers_[layer_id]->blobs_size();
      ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) {
      continue;
    }
    const BlobBase& blob = *layers_[layer_id]->blob_bases()[param_id];
    Dtype diff_abs_val_mean;
    blob.asum_diff(static_cast<void*>(&diff_abs_val_mean));
    diff_abs_val_mean /= blob.count();
    if (Caffe::root_solver()) {
      LOG(INFO) << "    [Backward] "
                << "Layer " << layer_names_[layer_id]
                << ", param blob " << param_id
                << " diff: " << diff_abs_val_mean;
    }
  }
}

template<typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int_tp param_id) {
  const BlobBase& blob = *params_[param_id];
  const int_tp param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  Dtype diff_abs_val_mean;
  blob.asum_diff(static_cast<void*>(&diff_abs_val_mean));
  diff_abs_val_mean /= blob.count();
  if (param_owner < 0) {
    Dtype data_abs_val_mean;
    blob.asum_data(static_cast<void*>(&data_abs_val_mean));
    diff_abs_val_mean /= blob.count();
    if (Caffe::root_solver()) {
      LOG(INFO) << "    [Update] Layer " << layer_name
                << ", param " << param_display_name
                << " data: " << data_abs_val_mean
                << "; diff: " << diff_abs_val_mean;
    }
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    if (Caffe::root_solver()) {
      LOG(INFO) << "    [Update] Layer " << layer_name
                << ", param blob " << param_display_name
                << " (owned by layer " << owner_layer_name << ", " << "param "
                << param_display_names_[param_owners_[param_id]] << ")"
                << " diff: " << diff_abs_val_mean;
    }
  }
}

template<typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const NetBase* other) {
  int_tp num_source_layers = other->layers().size();
  for (int_tp i = 0; i < num_source_layers; ++i) {
    LayerBase* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int_tp target_layer_id = 0;
    while (target_layer_id != layer_names_.size()
        && layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO)<< "Copying source layer " << source_layer_name;
    vector<shared_ptr<BlobBase> > target_blobs = layers_[target_layer_id]
        ->blob_bases();
    CHECK_EQ(target_blobs.size(), source_layer->blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int_tp j = 0; j < target_blobs.size(); ++j) {
      BlobBase* source_blob = source_layer->blob_bases()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareDataBase(source_blob);
    }
  }
}

template<typename Dtype>
void Net<Dtype>::BackwardFrom(int_tp start) {
  BackwardFromTo(start, 0);
}

template<typename Dtype>
void Net<Dtype>::BackwardTo(int_tp end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template<typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int_tp i = 0; i < learnable_params_.size(); ++i) {
      Dtype temp;
      learnable_params_[i]->asum_data(static_cast<void*>(&temp));
      asum_data += temp;
      learnable_params_[i]->asum_diff(static_cast<void*>(&temp));
      asum_diff += temp;
      learnable_params_[i]->sumsq_data(static_cast<void*>(&temp));
      sumsq_data += temp;
      learnable_params_[i]->sumsq_diff(static_cast<void*>(&temp));
      sumsq_diff += temp;
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template<typename Dtype>
void Net<Dtype>::Reshape() {
  for (int_tp i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
  // Set up shared blobs
  if (param_.reduced_memory_inference() && phase_ == caffe::TEST) {
    this->SetUpSharedBlobs();
  }
}

template<typename Dtype>
void Net<Dtype>::SetUpSharedBlobs() {
  vector<uint_tp> shared_blob_byte_sizes(shared_blobs_.size(), 0);
  for (int_tp i = 0; i < layers_.size(); ++i) {
    for (int_tp j = 0; j < top_vecs_[i].size(); ++j) {
      int_tp idx = param_.layer(i).top_shared_index(j);
      shared_blob_byte_sizes[idx] = std::max(shared_blob_byte_sizes[idx],
                                             top_vecs_[i][j]->byte_count());
    }
  }
  for (int_tp i = 0; i < shared_blobs_.size(); ++i) {
    vector<int_tp> buffer_shape(1, shared_blob_byte_sizes[i]);
    shared_blobs_[i]->Reshape(buffer_shape);
  }
  for (int_tp i = 0; i < layers_.size(); ++i) {
    for (int_tp j = 0; j < top_vecs_[i].size(); ++j) {
      int_tp idx = param_.layer(i).top_shared_index(j);
      top_vecs_[i][j]->ShareDataBase(shared_blobs_[idx].get());
      top_vecs_[i][j]->ShareDiffBase(shared_blobs_[idx].get());
    }
  }
}


template<typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {

  // Load quantizer statistics
  // Map by zones
  std::map<int_tp, std::pair<double, double> > quantizer_zone_map;
  // Map by names
  std::map<string, std::pair<double, double> > quantizer_name_map;
  for (int_tp i = 0; i < param.quantizer_size(); ++i) {
    QuantizerParameter quant_param = param.quantizer(i);
    if (quant_param.has_zone()) {
      quantizer_zone_map[quant_param.zone()] = std::make_pair<double, double>(
          quant_param.observed_min(), quant_param.observed_max());
    }
    if (quant_param.has_name()) {
      quantizer_name_map[quant_param.name()] = std::make_pair<double, double>(
          quant_param.observed_min(), quant_param.observed_max());
    }
  }

  for (int_tp i = 0; i < this->layers().size(); ++i) {
    vector<shared_ptr<QuantizerBase> > quantizers =
        this->layers()[i]->get_all_quantizers();
    vector<shared_ptr<QuantizerBase> > quant_base_vec =
        layers_[i]->get_all_quantizers();
    for (int_tp j = 0; j < quant_base_vec.size(); ++j) {
      const QuantizerParameter& quant_param = quant_base_vec[j]->quant_param();
      QuantizerParameter quant_param_copy;
      quant_param_copy.CopyFrom(quant_param);
      if (quant_param.has_zone()) {
        std::map<int_tp, std::pair<double, double> >::iterator iter =
            quantizer_zone_map.find(quant_param.zone());
        if (iter != quantizer_zone_map.end()) {
          quant_param_copy.set_observed_min(std::get<0>(iter->second));
          quant_param_copy.set_observed_max(std::get<1>(iter->second));
        }
      } else if (quant_param.has_name()) {
        std::map<string, std::pair<double, double> >::iterator iter =
            quantizer_name_map.find(quant_param.name());
        if (iter != quantizer_name_map.end()) {
          quant_param_copy.set_observed_min(std::get<0>(iter->second));
          quant_param_copy.set_observed_max(std::get<1>(iter->second));
        } else {
          LOG(WARNING) << "No quantizer parameters found for name: "
                       << quant_param.name();
        }
      }
      quant_base_vec[j]->update_param(quant_param_copy);
    }
  }

  int_tp num_source_layers = param.layer_size();
  for (int_tp i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int_tp target_layer_id = 0;
    while (target_layer_id != layer_names_.size()
        && layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO)<< "Copying source layer " << source_layer_name;
    vector<shared_ptr<BlobBase> > target_blobs = layers_[target_layer_id]
        ->blob_bases();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int_tp j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        shared_ptr<BlobBase> source_blob = CreateBlob(device_,
            source_layer.blobs(j).data_type());
        const bool kReshape = true;
        source_blob->FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob->shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template<typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
#ifdef USE_HDF5
  if (H5Fis_hdf5(trained_filename.c_str())) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
#else
  CopyTrainedLayersFromBinaryProto(trained_filename);
#endif
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
#ifdef USE_HDF5
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int_tp num_layers = hdf5_get_num_links(data_hid);
  for (int_tp i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int_tp target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<BlobBase> > target_blobs =
        layers_[target_layer_id]->blob_bases();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int_tp num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int_tp j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int_tp target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      // FIXME
      // hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
      //     target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
#else
  LOG(FATAL) << "CopyTrainedLayersFromHDF5 requires hdf5;"
             << " compile with USE_HDF5.";
#endif  // USE_HDF5
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers.";
  for (int_tp i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
  QuantizerToProto(param);
}

template<typename Dtype>
void Net<Dtype>::QuantizerToProto(NetParameter* param) const {

  param->clear_quantizer();

  // Map by zones
  std::map<int_tp, std::pair<double, double> > quantizer_zone_map;
  // Map by names
  std::map<string, std::pair<double, double> > quantizer_name_map;

  for (size_t i = 0; i < layers_.size(); ++i) {
    vector<shared_ptr<QuantizerBase> > quant_base_vec =
        layers_[i]->get_all_quantizers();
    for (size_t j = 0; j < quant_base_vec.size(); ++j) {
      if (quant_base_vec[j]->quant_param().has_zone()) {
        int_tp idx = quant_base_vec[j]->quant_param().zone();
        double l_min = quant_base_vec[j]->observed_min();
        double l_max = quant_base_vec[j]->observed_max();
        std::map<int_tp, std::pair<double, double> >::iterator iter =
            quantizer_zone_map.find(idx);
        if (iter != quantizer_zone_map.end()) {
          l_min = std::min(std::get<0>(iter->second), l_min);
          l_max = std::max(std::get<1>(iter->second), l_max);
        }
        quantizer_zone_map[idx] = std::make_pair(l_min, l_max);
      }
      if (quant_base_vec[j]->quant_param().has_name()) {
        string name = quant_base_vec[j]->quant_param().name();
        double l_min = quant_base_vec[j]->observed_min();
        double l_max = quant_base_vec[j]->observed_max();
        std::map<string, std::pair<double, double> >::iterator iter =
            quantizer_name_map.find(name);
        if (iter != quantizer_name_map.end()) {
          l_min = std::min(std::get<0>(iter->second), l_min);
          l_max = std::max(std::get<1>(iter->second), l_max);
        }
        quantizer_name_map[name] = std::make_pair(l_min, l_max);
      }
    }
  }


  DLOG(INFO) << "Serializing " << quantizer_zone_map.size()
             << " quantizer zones.";
  DLOG(INFO) << "Serializing " << quantizer_name_map.size()
             << " named quantizers.";
  for (std::map<int_tp, std::pair<double, double> >::iterator iter =
       quantizer_zone_map.begin(); iter != quantizer_zone_map.end(); ++iter) {
    QuantizerParameter* quant_param = param->add_quantizer();
    quant_param->set_zone(iter->first);
    quant_param->set_observed_min(std::get<0>(iter->second));
    quant_param->set_observed_max(std::get<1>(iter->second));
    DLOG(INFO) << "Quantizer zone " << iter->first << ": ["
               << std::get<0>(iter->second) << ","
               << std::get<1>(iter->second) << "].";
  }
  for (std::map<string, std::pair<double, double> >::iterator iter =
       quantizer_name_map.begin(); iter != quantizer_name_map.end(); ++iter) {
    QuantizerParameter* quant_param = param->add_quantizer();
    quant_param->set_name(iter->first);
    quant_param->set_observed_min(std::get<0>(iter->second));
    quant_param->set_observed_max(std::get<1>(iter->second));
    DLOG(INFO) << "Quantizer name " << iter->first << ": ["
               << std::get<0>(iter->second) << ","
               << std::get<1>(iter->second) << "].";
  }
}

template<typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
#ifdef USE_HDF5
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int_tp layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int_tp num_params = layers_[layer_id]->blobs_size();
    for (int_tp param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int_tp net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        // FIXME
        // hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
        //     *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        // FIXME
        // hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
        //     *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }

  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }

  std::map<int_tp, std::pair<double, double> > quantizer_map;
  for (size_t i = 0; i < layers_.size(); ++i) {
    vector<shared_ptr<QuantizerBase> > quant_base_vec =
        layers_[i]->get_all_quantizers();
    for (size_t j = 0; j < quant_base_vec.size(); ++j) {
      int_tp idx = quant_base_vec[j]->quant_param().zone();
      double l_min = quant_base_vec[j]->observed_min();
      double l_max = quant_base_vec[j]->observed_max();
      std::map<int_tp, std::pair<double, double> >::iterator iter =
          quantizer_map.find(idx);
      if (iter != quantizer_map.end()) {
        l_min = std::min(std::get<0>(iter->second), l_min);
        l_max = std::max(std::get<1>(iter->second), l_max);
      }
      quantizer_map[idx] = std::make_pair(l_min, l_max);
    }
  }

  hid_t quant_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  for (std::map<int_tp, std::pair<double, double> >::iterator iter =
      quantizer_map.begin(); iter != quantizer_map.end(); ++iter) {
    hid_t quant_idx_hid = H5Gcreate2(file_hid,
                               std::to_string(iter->first).c_str(), H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
    // FIXME: Implement HDF5 quantizer storage
    H5Gclose(quant_idx_hid);
  }
  H5Gclose(quant_hid);

  H5Fclose(file_hid);
// This code is taken from https://github.com/sh1r0/caffe-android-lib
#else
  LOG(FATAL) << "ToHDF5 requires hdf5; compile with USE_HDF5.";
#endif  // USE_HDF5
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int_tp i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int_tp i = 0; i < learnable_params_.size(); ++i) {
    BlobBase* blob = learnable_params_[i];
    blob->Clear();
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int_tp i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareDataBase(params_[param_owners_[i]].get());
    params_[i]->ShareDiffBase(params_[param_owners_[i]].get());
  }
}

bool NetBase::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

const shared_ptr<BlobBase> NetBase::blob_by_name(
    const string& blob_name) const {
  shared_ptr<BlobBase> blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((BlobBase*) (NULL));
    LOG(WARNING)<< "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

bool NetBase::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

const shared_ptr<LayerBase> NetBase::layer_by_name(
    const string& layer_name) const {
  shared_ptr<LayerBase> layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((LayerBase*) (NULL));
    LOG(WARNING)<< "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

class NetBase;
INSTANTIATE_CLASS_1T_GUARDED(Net, (half_fp)(float)(double));

}  // namespace caffe
