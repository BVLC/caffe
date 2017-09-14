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

#ifdef USE_MLSL

#include <string>
#include <map>
#include <set>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/multinode/mlsl.hpp"
#include "caffe/multinode/apply_mn_param.hpp"

namespace caffe {

template <typename Dtype>
void ApplyMultinodeParams(const NetParameter& param,
    NetParameter* param_with_mn) {
  // save per-layer global parameter mapping being applied later
  map<string, MnModelParallelParameter> net_layer_params;
  // aux map for inserting MnActivationLayer
  map<string, MnActivationParameter> blob_param_map;
  MultinodeParameter mn_param = param.multinode();

  // Step 1: Identify all the layers having global net params
  for (int param_id = 0; param_id < mn_param.model_parallel_size(); param_id++) {
    MnModelParallelParameter model_parallel_param = mn_param.model_parallel(param_id);
    string layer_from = model_parallel_param.layer_from();
    string layer_to = model_parallel_param.layer_to();
    set<string> marked_blobs;
    for (int i = 0; i < param.layer_size(); i++) {
      const LayerParameter& layer_param = param.layer(i);
      bool layer_covered_by_global = false;
      if (layer_param.name() == layer_from ||
        layer_param.name() == layer_to) {
        layer_covered_by_global = true;
      } else {
        for (int j = 0; j < layer_param.bottom_size(); j++) {
          if (marked_blobs.find(layer_param.bottom(j)) !=
            marked_blobs.end()) {
            layer_covered_by_global = true;
            break;
          }
        }
      }
      if (layer_covered_by_global) {
        for (int j = 0; j < layer_param.top_size(); j++) {
          marked_blobs.insert(layer_param.top(j));
        }
        net_layer_params[layer_param.name()] = model_parallel_param;
        // For cross-channel LRN, we assume there is always one model part
        // for simple implementation.
        if (layer_param.type() == "LRN" &&
            layer_param.lrn_param().norm_region() ==
            LRNParameter_NormRegion_ACROSS_CHANNELS) {
          net_layer_params[layer_param.name()].set_model_parts(1);
        }
      }
      if (layer_param.name() == layer_to ||
          layer_param.top_size() == 0) {
        break;
      }
    }
  }

  // Step 2: Identify the places to insert activation layers
  map<string, MnActivationParameter> blob_mdg_map;
  for (int i = 0; i < param.layer_size(); i++) {
    const LayerParameter& layer_param = param.layer(i);
    string layer_name = layer_param.name();
    string layer_type = layer_param.type();
    const MultinodeLayerParameter& mn_layer_param = layer_param.multinode();
    int num_nodes = mn_layer_param.num_nodes();
    int model_parts = mn_layer_param.model_parts();
    if (net_layer_params.find(layer_name) != net_layer_params.end()) {
      MnModelParallelParameter model_parallel_param =
        net_layer_params[layer_name];
      num_nodes = model_parallel_param.num_nodes();
      model_parts = model_parallel_param.model_parts();
    }
    for (int j = 0; j < layer_param.bottom_size(); j++) {
      string bottom_name = layer_param.bottom(j);
      if (blob_mdg_map.find(bottom_name) != blob_mdg_map.end()) {
        MnActivationParameter mdg = blob_mdg_map[bottom_name];
        mdg.set_num_nodes_out(num_nodes);
        mdg.set_model_parts_out(model_parts);
        int num_nodes_in = mdg.num_nodes_in();
        int num_nodes_out = mdg.num_nodes_out();
        int model_parts_in = mdg.model_parts_in();
        int model_parts_out = mdg.model_parts_out();
        mn::GetCanonicalMnParam(num_nodes_in, model_parts_in);
        mn::GetCanonicalMnParam(num_nodes_out, model_parts_out);
        if ((model_parts_out > 1 &&
             (layer_type == "Convolution" || layer_type == "InnerProduct" ||
              layer_type == "Accuracy" || layer_type == "SoftmaxWithLoss")) ||
            num_nodes_in != num_nodes_out ||
            model_parts_in != model_parts_out) {
          string layer_blob_name = layer_name + "/" + layer_param.bottom(j);
          if (layer_type == "Accuracy" || layer_type == "SoftmaxWithLoss") {
            mdg.set_need_reduce(false);
          }
          blob_param_map[layer_blob_name] = mdg;
        }
        blob_mdg_map.erase(bottom_name);
      }
    }
    for (int j = 0;  j < layer_param.top_size(); j++) {
      MnActivationParameter mdg;
      mdg.set_num_nodes_in(num_nodes);
      mdg.set_model_parts_in(model_parts);
      blob_mdg_map[layer_param.top(j)] = mdg;
    }
  }

  // Step 3: Create the new net, apply global mn setting to each layer,
  //         insert activation layers if needed
  param_with_mn->CopyFrom(param);
  param_with_mn->clear_layer();
  if (mn::is_param_server()) {
    // do not insert activation layers when loaded on param servers
    blob_param_map.clear();
  }
  for (int i = 0; i < param.layer_size(); i++) {
    const LayerParameter& orig_layer_param = param.layer(i);
    map<int, string> updated_blob_idx_to_name;
    for (int j = 0; j < orig_layer_param.bottom_size(); j++) {
      const string& bottom_blob_name = orig_layer_param.bottom(j);
      string layer_blob_name = orig_layer_param.name() + "/" + bottom_blob_name;
      if (blob_param_map.find(layer_blob_name) != blob_param_map.end()) {
        LayerParameter* mn_activation_layer_param =
          param_with_mn->add_layer();
        string new_name = "mn_activation/" + layer_blob_name;
        mn_activation_layer_param->Clear();
        mn_activation_layer_param->set_name(new_name);
        mn_activation_layer_param->set_type("MnActivation");
        mn_activation_layer_param->add_bottom(bottom_blob_name);
        mn_activation_layer_param->add_top(new_name);
        MnActivationParameter *mn_activation_param =
          mn_activation_layer_param->mutable_mn_activation_param();
        *mn_activation_param = blob_param_map[layer_blob_name];
        updated_blob_idx_to_name[j] = new_name;
      }
    }
    LayerParameter* layer_param = param_with_mn->add_layer();
    layer_param->CopyFrom(orig_layer_param);
    // Apply global mn setting
    if (net_layer_params.find(layer_param->name()) != net_layer_params.end()) {
      MultinodeLayerParameter *mn_layer_param = layer_param->mutable_multinode();
      const MnModelParallelParameter &mn_param = net_layer_params[layer_param->name()];
      mn_layer_param->set_num_nodes(mn_param.num_nodes());
      mn_layer_param->set_model_parts(mn_param.model_parts());
    }
    const MultinodeLayerParameter &mn_layer_param = layer_param->multinode();
    int num_nodes = mn_layer_param.num_nodes();
    int model_parts = mn_layer_param.model_parts();
    mn::GetCanonicalMnParam(num_nodes, model_parts);
    if (model_parts > 1 && !mn::is_param_server()) {
      // TODO: support transpose
      // TODO: support undividible num_output
      if (layer_param->type() == "Convolution") {
        ConvolutionParameter *conv_param = layer_param->mutable_convolution_param();
        int new_num_output = conv_param->num_output() / model_parts;
        CHECK_EQ(conv_param->num_output(), model_parts * new_num_output)
          << "Convolution layer " << layer_param->name()
          << ": Undividible num_output " << conv_param->num_output()
          << " by model_parts " << model_parts;
        conv_param->set_num_output(new_num_output);
      } else if (layer_param->type() == "InnerProduct") {
        InnerProductParameter *ip_param = layer_param->mutable_inner_product_param();
        int new_num_output = ip_param->num_output() / model_parts;
        CHECK_EQ(ip_param->num_output(), model_parts * new_num_output)
          << "InnerProduct layer " << layer_param->name()
          << ": Undividible num_output " << ip_param->num_output()
          << " by model_parts " << model_parts;
        ip_param->set_num_output(ip_param->num_output() / model_parts);
        CHECK(!ip_param->transpose()) << "Model parallelism does not support transpose!";
      }
      for (int j = 0; j < layer_param->blobs_size(); j++) {
        Blob<Dtype> blob;
        Blob<Dtype> new_blob;
        const BlobProto &proto = layer_param->blobs(j);
        blob.FromProto(proto);
        vector<int> shape = blob.shape();
        new_blob.Reshape(shape);
        if (shape.size() > 0) {
          if (proto.has_num() || proto.has_channels() ||
              proto.has_height() || proto.has_width()) {
            // deprecated 4D blob
            if (layer_param->type() == "InnerProduct") {
              CHECK_EQ(shape.size(), 4);
              CHECK_EQ(shape[0], 1);
              CHECK_EQ(shape[1], 1);
              if (shape[2] == 1) {
                shape.resize(1);
                shape[0] = blob.shape(3);
              } else {
                shape.resize(2);
                shape[0] = blob.shape(2);
                shape[1] = blob.shape(3);
              }
              new_blob.Reshape(shape);
            }
          }
          int count = blob.count() / model_parts;
          int offset = count * (mn::get_node_id() % model_parts);
          shape[0] /= model_parts;
          new_blob.Reshape(shape);
          caffe_copy(count, blob.cpu_data() + offset, new_blob.mutable_cpu_data());
          caffe_copy(count, blob.cpu_diff() + offset, new_blob.mutable_cpu_diff());
          BlobProto *updated_blob_proto = layer_param->mutable_blobs(j);
          updated_blob_proto->Clear();
          new_blob.ToProto(updated_blob_proto, true);
        }
      }
    }
    for (int j = 0; j < orig_layer_param.bottom_size(); j++) {
      if (updated_blob_idx_to_name.find(j) != updated_blob_idx_to_name.end()) {
        layer_param->set_bottom(j, updated_blob_idx_to_name[j]);
      }
    }
  }
}

template <typename Dtype>
void CopyMultinodeParamsFromNet(const Net<Dtype> *net, NetParameter *param) {
  // set per-layer multi-node parameters before adjusting net proto
  for (int i = 0; i < param->layer_size(); i++) {
    LayerParameter* source_layer = param->mutable_layer(i);
    const string& source_layer_name = source_layer->name();
    int target_layer_id = 0;
    while (target_layer_id != net->layer_names().size() &&
           net->layer_names()[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == net->layer_names().size()) continue;
    *source_layer->mutable_multinode() =
      net->layers()[target_layer_id]->layer_param().multinode();
  }
}

template <typename Dtype>
void RevertMultinodeParams(NetParameter* param, bool write_diff) {
  NetParameter orig_param;
  orig_param.CopyFrom(*param);
  param->clear_layer();
  for (int i = 0; i < orig_param.layer_size(); i++) {
    const LayerParameter& orig_layer_param = orig_param.layer(i);
    if (orig_layer_param.type() == "MnActivation") continue;
    LayerParameter* layer_param = param->add_layer();
    layer_param->CopyFrom(orig_layer_param);
    layer_param->clear_bottom();
    for (int j = 0; j < orig_layer_param.bottom_size(); j++) {
      string bottom_name = orig_layer_param.bottom(j);
      string prefix = "mn_activation/" + orig_layer_param.name() + "/";
      if (bottom_name.find(prefix) == 0) {
        bottom_name = bottom_name.substr(prefix.size());
      }
      layer_param->add_bottom(bottom_name);
    }
    const MultinodeLayerParameter &mn_layer_param = orig_layer_param.multinode();
    int num_nodes = mn_layer_param.num_nodes();
    int model_parts = mn_layer_param.model_parts();
    mn::GetCanonicalMnParam(num_nodes, model_parts);
    if (model_parts > 1) {
      if (layer_param->type() == "Convolution") {
        ConvolutionParameter *conv_param = layer_param->mutable_convolution_param();
        conv_param->set_num_output(conv_param->num_output() * model_parts);
      } else if (layer_param->type() == "InnerProduct") {
        InnerProductParameter *ip_param = layer_param->mutable_inner_product_param();
        ip_param->set_num_output(ip_param->num_output() * model_parts);
        CHECK(!ip_param->transpose()) << "Model parallelism does not support transpose!";
      }
      layer_param->clear_blobs();
      for (int j = 0; j < orig_layer_param.blobs_size(); j++) {
        BlobProto *blob_proto = layer_param->add_blobs();
        Blob<Dtype> orig_blob;
        orig_blob.FromProto(orig_layer_param.blobs(j));
        vector<int> shape = orig_blob.shape();
        Blob<Dtype> new_blob;
        if (shape.size() > 0) {
          mn::Distribution *distrib = mn::get_distrib(num_nodes/model_parts, model_parts);
          int count = orig_blob.count();
          shape[0] *= model_parts;
          new_blob.Reshape(shape);
          distrib->allgather<Dtype,MLSL::GT_MODEL>(
            orig_blob.mutable_cpu_data(), count, new_blob.mutable_cpu_data());
          if (write_diff) {
            distrib->allgather<Dtype,MLSL::GT_MODEL>(
              orig_blob.mutable_cpu_diff(), count, new_blob.mutable_cpu_diff());
          }
        }
        new_blob.ToProto(blob_proto, write_diff);
      }
    }
    layer_param->mutable_multinode()->Clear();
  }
}

template void ApplyMultinodeParams<float>(const NetParameter& param,
    NetParameter* param_with_mn);
template void ApplyMultinodeParams<double>(const NetParameter& param,
    NetParameter* param_with_mn);
template void CopyMultinodeParamsFromNet<float>(const Net<float> *net, NetParameter *param);
template void CopyMultinodeParamsFromNet<double>(const Net<double> *net, NetParameter *param);
template void RevertMultinodeParams<float>(NetParameter* param, bool write_diff);
template void RevertMultinodeParams<double>(NetParameter* param, bool write_diff);
} // namespace caffe

#endif // USE_MLSL
