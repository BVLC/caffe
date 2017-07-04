/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "boost/algorithm/string.hpp"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/cpu_info.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/performance.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/multinode/mlsl.hpp"
#include "caffe/multinode/apply_mn_param.hpp"
#include "caffe/util/remove_batch_norm.hpp"

PERFORMANCE_CREATE_MONITOR();

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages,
    const Net* root_net, std::string engine)
    : root_net_(root_net) {
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
  if (engine != "")
    param.set_engine(engine);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  CHECK(Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";

#ifdef _OPENMP
  static bool executed = false;
  if (!executed) {
    if (Caffe::mode() == Caffe::GPU) {
      caffe::cpu::OpenMpManager::setGpuEnabled();
    } else {
      caffe::cpu::OpenMpManager::setGpuDisabled();
    }

    caffe::cpu::OpenMpManager::bindOpenMpThreads();
    caffe::cpu::OpenMpManager::printVerboseInformation();
  }
#endif

  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);

  // Backward compatibility for obsolete compile-time flags
#ifdef USE_MKL2017_AS_DEFAULT_ENGINE
  if (filtered_param.engine() == "")
    filtered_param.set_engine("MKL2017");
#endif
#ifdef USE_MKLDNN_AS_DEFAULT_ENGINE
  if (filtered_param.engine() == "")
    filtered_param.set_engine("MKLDNN");
#endif
  engine_name_ = filtered_param.engine();

  NetParameter& param = filtered_param;
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param_with_splits;
  InsertSplits(param, &param_with_splits);
  param = param_with_splits;

  NetParameter compiled_param;
  // Transform Net (merge layers etc.) improve computational performance
  CompileNet(param, &compiled_param);
  param = compiled_param;
  this->bn_scale_remove_ = param.compile_net_state().bn_scale_remove();
  this->bn_scale_merge_ = param.compile_net_state().bn_scale_merge();
  int kept_bn_layers_num = param.compile_net_state().kept_bn_layers_size();
  for (int idx = 0; idx < kept_bn_layers_num; ++idx) {
    this->kept_bn_layers_.push_back(param.compile_net_state().kept_bn_layers(idx));
  }

#ifdef USE_MLSL
  NetParameter param_with_mn;
  if (mn::is_multinode()) {
    ApplyMultinodeParams<Dtype>(param, &param_with_mn);
    param = param_with_mn;
  }
#endif

  // Printing processed model
  if (Caffe::root_solver()) {
    LOG(INFO) << "Initializing net from parameters: " << std::endl;
    LOG(INFO).flush();
    fflush(0);
    param.PrintDebugString();
    fflush(0);
  }

#ifdef USE_MLSL
  int global_batch_size = -1;
#endif
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (param.engine() != "" && param.layer(layer_id).engine() == "")
      param.mutable_layer(layer_id)->set_engine(param.engine());
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
    } else {
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    }
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
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

#ifdef USE_MLSL
    if (!layer_param.type().compare("Data")       ||
        !layer_param.type().compare("DummyData")  ||
        !layer_param.type().compare("ImageData")  ||
        !layer_param.type().compare("HDF5Data")   ||
        !layer_param.type().compare("MemoryData") ||
        !layer_param.type().compare("Input") ||
        !layer_param.type().compare("WindowData")) {

        // FIXME: retrieve batch_size from top[0]->shape[0] when MLSL stuff will be moved from LayerSetUp
        //int batch_size = top_vecs_[layer_id][0]->shape(0);

        int batch_size = 0;
        if (!layer_param.type().compare("Data"))
            batch_size = layer_param.data_param().batch_size();
        else if (!layer_param.type().compare("DummyData"))
            batch_size = layer_param.dummy_data_param().shape(0).dim(0);
        else if (!layer_param.type().compare("ImageData"))
            batch_size = layer_param.image_data_param().batch_size();
        else if (!layer_param.type().compare("HDF5Data"))
            batch_size = layer_param.hdf5_data_param().batch_size();
        else if (!layer_param.type().compare("MemoryData"))
            batch_size = layer_param.memory_data_param().batch_size();
        else if (!layer_param.type().compare("WindowData"))
            batch_size = layer_param.window_data_param().batch_size();

        if (caffe::TRAIN == param.state().phase()) {
            LOG(WARNING) << "SetMinibatchSize " << batch_size;
            if (global_batch_size < 0) {
              global_batch_size = batch_size * mn::get_nodes_count();
              mn::train::set_global_minibatch_size(global_batch_size);
            } else {
              CHECK_EQ(global_batch_size, batch_size * mn::get_nodes_count());
            }
        }
    }
#endif /* USE_MLSL */

    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }

  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
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
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
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
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();

#ifdef USE_MLSL
  if (this->phase_ == TRAIN) {
      for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
        boost::shared_ptr<Layer<Dtype>> layer{ layers_[layer_id] };
        if ((layer->layerOp != nullptr) && layer->layerOp->HasParameterSets()) {
              vector<int> param_ids = get_layer_learnable_param_ids(layer_id);
              for (int i = 0; i < param_ids.size(); i++) {
                  int mlsl_weight_size = layer->layerOp->GetParameterSet(i)->GetLocalKernelCount()
                                        * layer->layerOp->GetParameterSet(i)->GetKernelSize()
                                        * sizeof(Dtype);
                  int caffe_weight_size = learnable_params_[param_ids[i]]->count() * sizeof(Dtype);
                  if (mlsl_weight_size < caffe_weight_size)
                      LOG(FATAL) << "InitNet: ERROR: check weight sizes for layer " << layer->type() << ", layer_id " << layer_id
                                 << ", param_id " << param_ids[i]
                                 << ", MLSL weight size in bytes " << mlsl_weight_size
                                 << ", CAFFE weight size in bytes " << caffe_weight_size;
              }
          }
      }
  }
#endif /* USE_MLSL */

  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
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
void Net<Dtype>::CompileNet(const NetParameter& param,
    NetParameter* param_compiled) {



  NetParameter param_temp0;
  param_temp0.CopyFrom(param);
  param_temp0.clear_layer();
  RemoveBNScale(param, &param_temp0);

  NetParameter param_temp;  // temporary compiled param
  param_temp.CopyFrom(param_temp0);
  param_temp.clear_layer();    // Remove layers
  CompilationRuleOne(param_temp0, &param_temp);

  NetParameter param_temp2;  // temporary compiled param
  param_temp2.CopyFrom(param_temp);
  param_temp2.clear_layer();   // Remove layers

  CompilationRuleTwo(param_temp, &param_temp2);

  param_compiled->CopyFrom(param_temp2);
  param_compiled->clear_layer();    // Remove layers
  CompilationRuleThree(param_temp2, param_compiled);
}

template <typename Dtype>
void Net<Dtype>::CompilationRuleOne(const NetParameter& param,
                                    NetParameter* param_compiled) {

  bool merge_bn_scale = false;
  std::set<std::string> layers_to_drop;
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(i);
    bool layer_included = true;

    // Optimization rule 1:
    // - If we are having engine MKL2017 and Scale layer within a model
    // and input bottom comes from  BatchNorm of engine MKL2017
    // then we can remove Scale layer
    // and rename BatchNorm top blob after deleted Scale's top

    // Extension of optimization rule 1:
    // - If we are having engine MKLDNN and Scale layer within a model
    // and input bottom comes from  BatchNorm of engine MKLDNN
    // then we can remove Scale layer
    // and rename BatchNorm top blob after deleted Scale's top

        // If current layer is BatchNorm of MKL2017 engine..
    if (((layer_param->type().compare("BatchNorm") == 0) &&
       ((layer_param->batch_norm_param().engine() ==
         BatchNormParameter_Engine_MKL2017)
       || ((layer_param->batch_norm_param().engine() ==
           BatchNormParameter_Engine_DEFAULT) &&
            param.engine().compare("MKL2017") == 0))) ||
        // If current layer is BatchNorm of MKLDNN engine..
        ((layer_param->type().compare("BatchNorm") == 0) &&
         ((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_MKLDNN)
          || (((layer_param->batch_norm_param().engine() == BatchNormParameter_Engine_DEFAULT) &&
               (param.engine().compare(0, 6, "MKLDNN") == 0)) ||
              (param.engine() == "" &&
               layer_param->engine().compare(0, 6, "MKLDNN") == 0))))) {
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params,
                       layer_param->top(0),
                       param,
                       i+1 < param.layer_size() ? i+1 : i);
      const LayerParameter& consumer_layer_param =
                                    consumer_layer_params.size() > 0 ?
                                    *(consumer_layer_params[0]) : *layer_param;
      // Consumer layer of blob produced by BN
      // has to be Scale layer with one Input Blob
      if ((consumer_layer_param.type().compare("Scale") == 0) &&
           (consumer_layer_param.bottom_size() == 1)) {
        string& batchnorm_top_blob_name =
            const_cast<string&>(layer_param->top(0));
        const string& scale_top_blob_name = consumer_layer_param.top(0);
        // Mark Consumer layer (its name) as the one marked for dropping
        layers_to_drop.insert(consumer_layer_param.name());
        if (!merge_bn_scale) merge_bn_scale = true;

        // Replace BatchNorm top name with Scale top name
        batchnorm_top_blob_name.resize(scale_top_blob_name.size());
        batchnorm_top_blob_name.replace(0,
                                        scale_top_blob_name.size(),
                                        scale_top_blob_name);
        // Read the bias_term param of Scale Layer and set bias_term param
        // of MKLBatchNorm accordingly
        bool scale_bias_term = consumer_layer_param.
                               scale_param().bias_term();
        layer_param->mutable_batch_norm_param()->
        set_bias_term(scale_bias_term);
        if (consumer_layer_param.blobs_size() == 2) {
          layer_param->add_blobs()->CopyFrom(consumer_layer_param.blobs(0));
          layer_param->add_blobs()->CopyFrom(consumer_layer_param.blobs(1));
        }
      }
    }

    if (layers_to_drop.find(layer_param->name()) != layers_to_drop.end()) {
      LOG_IF(INFO, Caffe::root_solver()) << "Dropped layer: "
             << layer_param->name() << std::endl;
      layer_included = false;
      // Remove dropped layer from the list of layers to be dropped
      layers_to_drop.erase(layers_to_drop.find(layer_param->name()));
    }

    if (layer_included) {
      param_compiled->add_layer()->CopyFrom(*layer_param);
    }
  }
  param_compiled->mutable_compile_net_state()->set_bn_scale_merge(merge_bn_scale);
}


template <typename Dtype>
void Net<Dtype>::CompilationRuleTwo(const NetParameter& param,
                                    NetParameter* param_compiled) {
  std::set<std::string> layers_to_drop;
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(i);
    bool layer_included = true;

    // Optimization rule 2:
    // - If we are having engine MKLDNN and ReLU layer within a model
    // and input bottom comes from  Convolution of engine MKLDNN
    // then we can remove ReLU layer
    // and rename Convolution top blob after deleted ReLU's top
    // Note: Currently merging of convolution and relu layers is feasible
    // only for caffe::TEST phase, as there is no Backward primitive of conv Relu

    // If current layer is Convolution of MKLDNN engine..
    /*
    //Old Structure:        if ((A == TEST) && (B == 0) && ((C == ConvolutionParameter_Engine_MKLDNN) || ((D == ConvolutionParameter_Engine_DEFAULT) && ((E == 0 && F == string::npos)) || ((G == "" && H == 0 && I == string::npos)))))
    //New tmp Structure:    if ((A == TEST) && (B == 0) && ((C == ConvolutionParameter_Engine_MKLDNN) || (((D == ConvolutionParameter_Engine_DEFAULT) && ((E == 0 && F == string::npos))) || ((G == "" && H == 0 && I == string::npos)))))
    //New Structure:        if ((A == TEST) && (B == 0) && ((C == ConvolutionParameter_Engine_MKLDNN) || (((D == ConvolutionParameter_Engine_DEFAULT) && (E == 0 && F == string::npos)) || (G == "" && H == 0 && I == string::npos))))
    //Old Structure:
    //if ((A == TEST) &&
    //    (B == 0) &&
    //   ((C == ConvolutionParameter_Engine_MKLDNN)
    //   || ((D == ConvolutionParameter_Engine_DEFAULT) &&
    //        ((E == 0
    //        && F == string::npos)) ||
    //        ((G == "" &&
    //          H == 0 &&
    //          I == string::npos)))))
    */
    if ((param.state().phase() == TEST) &&
        (layer_param->type().compare("Convolution") == 0) &&
       ((layer_param->convolution_param().engine() == ConvolutionParameter_Engine_MKLDNN)
       || (((layer_param->convolution_param().engine() == ConvolutionParameter_Engine_DEFAULT) &&
            (param.engine().compare(0, 6, "MKLDNN") == 0
            && param.engine().find(":DLA", 6) == string::npos)) ||
            (param.engine() == "" &&
              layer_param->engine().compare(0, 6, "MKLDNN") == 0 &&
              layer_param->engine().find(":DLA", 6) == string::npos)))) {
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params, layer_param->top(0),
                       param, i+1 < param.layer_size() ? i+1 : i);
      const LayerParameter& consumer_layer_param =
                                    consumer_layer_params.size() > 0 ?
                                    *(consumer_layer_params[0]) : *layer_param;

      // Consumer layer of blob produced by Conv
      // has to be ReLU layer with one Input Blob
      /*
      //Old Structure:      if ((A == 0) && ((B == ReLUParameter_Engine_MKLDNN) || ((C == ReLUParameter_Engine_DEFAULT) && ((D == 0 && E == string::npos)) || ((F == "" && G == 0 && H == string::npos)))))
      //New tmp Structure:  if ((A == 0) && ((B == ReLUParameter_Engine_MKLDNN) || (((C == ReLUParameter_Engine_DEFAULT) && ((D == 0 && E == string::npos))) || ((F == "" && G == 0 && H == string::npos)))))
      //New Structure:      if ((A == 0) && ((B == ReLUParameter_Engine_MKLDNN) || (((C == ReLUParameter_Engine_DEFAULT) && (D == 0 && E == string::npos)) || (F == "" && G == 0 && H == string::npos))))
      //Old Structure:
      //if ((A == 0) &&
      //  ((B == ReLUParameter_Engine_MKLDNN)
      //  || ((C == ReLUParameter_Engine_DEFAULT) &&
      //      ((D == 0
      //      && E == string::npos)) ||
      //      ((F == "" &&
      //        G == 0 &&
      //        H == string::npos)))))
      */
      if ((consumer_layer_param.type().compare("ReLU") == 0) &&
        ((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_MKLDNN)
        || (((consumer_layer_param.relu_param().engine() == ReLUParameter_Engine_DEFAULT) &&
            (param.engine().compare(0, 6, "MKLDNN") == 0
            && param.engine().find(":DLA", 6) == string::npos)) ||
            (param.engine() == "" &&
              layer_param->engine().compare(0, 6, "MKLDNN") == 0 &&
              layer_param->engine().find(":DLA", 6) == string::npos)))) {
        string& convolution_top_blob_name =
            const_cast<string&>(layer_param->top(0));
        const string& scale_top_blob_name = consumer_layer_param.top(0);
        // Mark Consumer layer (its name) as the one marked for dropping
        layers_to_drop.insert(consumer_layer_param.name());

        // Replace Convolution top name with ReLU top name
        convolution_top_blob_name.resize(scale_top_blob_name.size());
        convolution_top_blob_name.replace(0,
                                        scale_top_blob_name.size(),
                                        scale_top_blob_name);
        // set relu flag in convolution
        layer_param->mutable_convolution_param()->set_relu(true);
        float negative_slope1 =
                  consumer_layer_param.relu_param().negative_slope();
        layer_param->mutable_convolution_param()->
                    set_negative_slope(negative_slope1);
      }
    }

    if (layers_to_drop.find(layer_param->name()) != layers_to_drop.end()) {
      LOG_IF(INFO, Caffe::root_solver()) << "Dropped layer: "
             << layer_param->name() << std::endl;
      layer_included = false;
      // Remove dropped layer from the list of layers to be dropped
      layers_to_drop.erase(layers_to_drop.find(layer_param->name()));
    }

    if (layer_included) {
      param_compiled->add_layer()->CopyFrom(*layer_param);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CompilationRuleThree(const NetParameter& param,
                             NetParameter* param_compiled) {
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param =
          (const_cast<NetParameter&>(param)).mutable_layer(i);

    // Optimization rule 3:
    // - If we are having engine MKL2017 and Batch Normalization
    // doing inplace computation then
    // to improve performance we create another top buffer
    // and make other layers consuming BatchNorm top to use new buffer

    // If current layer is BatchNorm of MKL2017 engine..
    if (((layer_param->type().compare("BatchNorm") == 0) &&
        ((layer_param->batch_norm_param().engine() ==
         BatchNormParameter_Engine_MKL2017)
        || ((layer_param->batch_norm_param().engine() ==
           BatchNormParameter_Engine_DEFAULT) &&
            param.engine().compare("MKL2017") == 0))) &&
        (layer_param->top(0) == layer_param->bottom(0) )) {
      std::string& batch_norm_top = const_cast<string&>(layer_param->top(0));
      std::vector<const LayerParameter*> consumer_layer_params;
      GetBlobConsumers(consumer_layer_params,
                       batch_norm_top,
                       param,
                       i+1 < param.layer_size() ? i+1 : i);

      for (std::vector<const LayerParameter*>::iterator it =
        consumer_layer_params.begin();
        it != consumer_layer_params.end(); ++it) {
        // If consumer is computing inplace then modify top as well
        if (((*it)->top_size() > 0 ) &&
            ((*it)->bottom(0).compare((*it)->top(0)) == 0)) {
          // Modify consumer top
          const_cast<string&>((*it)->top(0)).append("_x");
        }

        // Modify consumer bottom. Sometimes searched
        // buffer is under higher bottom index than 0 eg.
        // In case of Eltwise
        for (unsigned int i = 0; i < (*it)->bottom_size(); ++i) {
          if ((*it)->bottom(i).compare(batch_norm_top) == 0) {
            const_cast<string&>((*it)->bottom(i)).append("_x");
          }
        }
      }
      // Modify top so it is diffrent from bottom
      batch_norm_top.append("_x");
    }
    param_compiled->add_layer()->CopyFrom(*layer_param);
  }
  return;
}


template <typename Dtype>
void Net<Dtype>::RemoveBNScale(const NetParameter& param,
                             NetParameter* param_compiled) {
    // - In TEST Phase, if we detect sequential layers conv->batch norm ->scale,
    // We will merge batch norm and scale layer into conv layer.
  if(param.state().phase() != TEST) {
    param_compiled->CopyFrom(param);
    param_compiled->mutable_compile_net_state()->set_bn_scale_remove(false);
    return ;
  }

  bool bn_scale_remove = false;
  bool is_net_init = param.compile_net_state().is_init();
  std::set<std::string> layers_to_drop;
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter *layer_param = (const_cast<NetParameter&>(param)).mutable_layer(i);
    bool layer_included = true;
    bool bn_use_global_stats_set = true;
    if (layer_param->type().compare("Convolution") == 0) {
      std::vector<const LayerParameter*> child_layers_params;
      GetBlobConsumers(child_layers_params, layer_param->top(0), param, i + 1 < param.layer_size() ? i + 1 : i);
      const LayerParameter &child_layer_param = child_layers_params.size() > 0 ? *(child_layers_params[0]) : *layer_param;
      // check whether child layer is BatchNorm
      if (child_layer_param.type().compare("BatchNorm") == 0) {
        BatchNormParameter bn_param = child_layer_param.batch_norm_param();
        if (is_net_init) {
          //Testing Network init process
          bool bn_use_global_stats = true;
          if (bn_param.has_use_global_stats()) {
            bn_use_global_stats = bn_param.use_global_stats();
          }
          if (!bn_use_global_stats) {
            //This bn layer's use_global_stats is set manually! Don't remove it.
            //remained_bn_layer_names.push_back(child_layer_param.name());
            param_compiled->mutable_compile_net_state()->add_kept_bn_layers(child_layer_param.name());
            bn_use_global_stats_set = false;
          }
        } else {
          int kept_bn_layers_num = param.compile_net_state().kept_bn_layers_size();
          bool in_kept_list = false;
          for (int idx = 0; idx < kept_bn_layers_num; ++idx) {
            if (child_layer_param.name().compare(param.compile_net_state().kept_bn_layers(idx)) == 0) {
              in_kept_list = true;
              break;
            }
          }
          if (in_kept_list) {
            bn_use_global_stats_set = false;
          }
        }

        if (!bn_use_global_stats_set) {
          //Even in caffe TEST phase, current batch norm layer has set use_global_stats = false in protxt file, so we won't
          //merge this layer into convolution layer.
         param_compiled->add_layer()->CopyFrom(*layer_param);
          continue;
        }
        std::vector<const LayerParameter*> grandchild_layers_params;
        GetBlobConsumers(grandchild_layers_params, child_layer_param.top(0), param, i + 2 < param.layer_size() ? i + 2 : i);
        const LayerParameter &grandchild_layer_param = (grandchild_layers_params.size() > 0) ? *(grandchild_layers_params[0]) : child_layer_param;
        if (grandchild_layer_param.type().compare("Scale") == 0) {
          MergeLayer(*layer_param, grandchild_layer_param);
          AdjustConvLayer<Dtype>(*layer_param, child_layer_param, grandchild_layer_param, is_net_init);
          if (bn_scale_remove == false) bn_scale_remove = true;
          layers_to_drop.insert(child_layer_param.name());
          layers_to_drop.insert(grandchild_layer_param.name());
        } else if (&child_layer_param != &grandchild_layer_param) {
          //In fact, conv-->batchnorm can also be optimized. In such case, we check the blob size of batch norm layer
          //if is 3, it means current net hasn't used scale layer, this is equivalent to scale layer with all 1 weights and 0 bias
          //if is 4 or 5, it means intel caffe compilation rule 1 works here, we can recover the scale layer from batch norm layer
          MergeLayer(*layer_param, child_layer_param);
          if (!is_net_init) {
            shared_ptr<LayerParameter> scale_layer_param(new LayerParameter());
            RecoverScaleFromBN(child_layer_param, *scale_layer_param, (Dtype)1, (Dtype)0);
            AdjustConvLayer<Dtype>(*layer_param, child_layer_param, *scale_layer_param, is_net_init);
          } else {
            AdjustConvLayer<Dtype>(*layer_param, child_layer_param, grandchild_layer_param, true);
		  }
          if (bn_scale_remove == false) bn_scale_remove = true;
          layers_to_drop.insert(child_layer_param.name());
        }
      }
    }
    if (layers_to_drop.find(layer_param->name()) != layers_to_drop.end()) {
      LOG_IF(INFO, Caffe::root_solver()) << "Dropped Layer: "<< layer_param->name() << std::endl;
      layer_included = false;
      // Remove dropped layer from the list of layers to be dropped
      layers_to_drop.erase(layers_to_drop.find(layer_param->name()));
    }
    if (layer_included) {
            if (layer_param->type().compare("BatchNorm") == 0) {
              param_compiled->mutable_compile_net_state()->add_kept_bn_layers(layer_param->name());
            }
            param_compiled->add_layer()->CopyFrom(*layer_param);
    }
  }

  param_compiled->mutable_compile_net_state()->set_bn_scale_remove(bn_scale_remove);
 }

template <typename Dtype>
void Net<Dtype>::GetBlobConsumers(
                  std::vector<const LayerParameter*>& consumer_blobs,
                  const string& blob_name_to_find,
                  const NetParameter& param,
                  int layer_id_to_start_traversing_from) {
  consumer_blobs.clear();
  // Validate values of ids of layers are <1..num_layers-1>
  CHECK_GE(layer_id_to_start_traversing_from, 1);
  CHECK_LT(layer_id_to_start_traversing_from, param.layer_size());

  // Traverse through layers to search the layer that consumes blob_name_to_find
  for (int i = layer_id_to_start_traversing_from; i < param.layer_size(); ++i) {
    // check bottom blobs if any of them is consuming given blob
    for (int j = 0; j < param.layer(i).bottom_size(); ++j) {
      if (param.layer(i).bottom(j).compare(blob_name_to_find) == 0) {
        consumer_blobs.push_back(&param.layer(i));
      }
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
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
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
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
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
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
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
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

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
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
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
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
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
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



template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    PERFORMANCE_MEASUREMENT_BEGIN();

    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);

    PERFORMANCE_MEASUREMENT_END((std::string("FW_") + layer_names_[i]).c_str());

    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    if (layer_need_backward_[i]) {
      PERFORMANCE_MEASUREMENT_BEGIN();

      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);

      PERFORMANCE_MEASUREMENT_END((std::string("BW_")+layer_names_[i]).c_str());

      if (debug_info_) { BackwardDebugInfo(i); }
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {


    if (this->bn_scale_remove_) {
    //This path shows testing network's blobs(weight & bias) has been adjusted
    //We can't share weights & blobs with training net! We will save current
    //training net to a temp model file and load to memory later
    NetParameter temp_net_param;
    NetParameter complete_net_param;
    other->ToProto(&temp_net_param, false);
    //Copy this->remained_bn_layer_names to temp_net_param
    for (vector<string>::iterator it = kept_bn_layers_.begin(); it != kept_bn_layers_.end(); it++) {
      temp_net_param.mutable_compile_net_state()->add_kept_bn_layers(*it);
    }
    //temp_net_param.mutable_compile_net_state()->set_bn_top_rename(other->bn_top_rename_);
    complete_net_param.CopyFrom(temp_net_param);
    if (other->bn_scale_merge_) {
      complete_net_param.clear_layer();
      RecoverBNScaleMergedNet<Dtype>(&temp_net_param, &complete_net_param);
    }
    CopyTrainedLayersFrom(complete_net_param);
    return ;
  }
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
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
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param_inp) {
  NetParameter param_tmp = param_inp;
  NetParameter &param = param_tmp;
  param.set_engine(engine_name_);
  param_tmp.mutable_state()->set_phase(phase_);
  param_tmp.mutable_compile_net_state()->set_is_init(false);
  for (vector<string>::iterator it = this->kept_bn_layers_.begin(); it != this->kept_bn_layers_.end(); it++) {
    param_tmp.mutable_compile_net_state()->add_kept_bn_layers(*it);
  }
  NetParameter param_compiled;
  CompileNet(param, &param_compiled);
  param = param_compiled;
#ifdef USE_MLSL
  NetParameter param_mn;
  if (mn::is_multinode()) {
    CopyMultinodeParamsFromNet<Dtype>(this, &param);
    ApplyMultinodeParams<Dtype>(param, &param_mn);
    param = param_mn;
  }
#endif

  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
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
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
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
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
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
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
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
#ifdef USE_MLSL
      const MultinodeLayerParameter &mn_layer_param =
        layers_[target_layer_id]->layer_param().multinode();
      int num_nodes = mn_layer_param.num_nodes();
      int model_parts = mn_layer_param.model_parts();
      mn::GetCanonicalMnParam(num_nodes, model_parts);
      Blob<Dtype> orig_blob;
      vector<int> shape = target_blobs[j]->shape();
      CHECK_GT(shape.size(), 0);
      int offset = 0;
      if (model_parts > 1) {
        shape[0] *= model_parts;
        offset = target_blobs[j]->count() * (mn::get_node_id() % model_parts);
      }
      orig_blob.Reshape(shape);
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          &orig_blob);
      caffe_copy(target_blobs[j]->count(), orig_blob.cpu_data() + offset,
                 target_blobs[j]->mutable_cpu_data());
#else
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
#endif
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
  // TODO: Should implement the param adjustment for ToHDF5 as well
  // TODO: Decompile net to BVLC compatibility
  // DecompileNet(param);
#ifdef USE_MLSL
  if (mn::is_multinode()) {
    RevertMultinodeParams<Dtype>(param, write_diff);
  }
#endif
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
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
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
#ifdef USE_MLSL
    if (layer_param.type() == "MnActivation") continue;
#endif
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
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
#ifdef USE_MLSL
      const MultinodeLayerParameter &mn_layer_param = layer_param.multinode();
      int num_nodes = mn_layer_param.num_nodes();
      int model_parts = mn_layer_param.model_parts();
      mn::GetCanonicalMnParam(num_nodes, model_parts);
      Blob<Dtype> new_blob;
      vector<int> shape = params_[net_param_id]->shape();
      CHECK_GT(shape.size(), 0);
      if (model_parts > 1) {
        mn::Distribution *distrib = mn::get_distrib(num_nodes/model_parts, model_parts);
        shape[0] *= model_parts;
        new_blob.Reshape(shape);
        distrib->allgather<Dtype,MLSL::GT_MODEL>(
          params_[net_param_id]->mutable_cpu_data(),
          params_[net_param_id]->count(),
          new_blob.mutable_cpu_data());
        if (write_diff) {
          distrib->allgather<Dtype,MLSL::GT_MODEL>(
            params_[net_param_id]->mutable_cpu_diff(),
            params_[net_param_id]->count(),
            new_blob.mutable_cpu_diff());
        }
      } else {
        new_blob.Reshape(shape);
        caffe_copy(new_blob.count(), params_[net_param_id]->cpu_data(),
                   new_blob.mutable_cpu_data());
        if (write_diff) {
          caffe_copy(new_blob.count(), params_[net_param_id]->cpu_diff(),
                     new_blob.mutable_cpu_diff());
        }
      }
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            new_blob);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            new_blob, true);
      }
#else
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
#endif
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
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs(int learnable_param_id) {
  Blob<Dtype>* blob = learnable_params_[learnable_param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU:
      if (blob->prv_diff())
        caffe_set(blob->prv_diff_count(), static_cast<Dtype>(0),
                  blob->mutable_prv_diff());
      else
        caffe_set(blob->count(), static_cast<Dtype>(0),
                  blob->mutable_cpu_diff());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                  blob->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    ClearParamDiffs(i);
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
vector<int> Net<Dtype>::get_layer_learnable_param_ids(int layer_id) const {
  CHECK_GE(layer_id, 0);
  CHECK(layer_id < param_id_vecs_.size());
  const vector<int>& layer_param_ids = param_id_vecs_[layer_id];
  vector<int> ret;
  for (int i = 0; i < layer_param_ids.size(); ++i) {
    ret.push_back(learnable_param_ids_[layer_param_ids[i]]);
    CHECK(params_[layer_param_ids[i]].get() == learnable_params_[ret.back()]);
  }
  return ret;
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
