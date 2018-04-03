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

#include <vector>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/util/remove_batch_norm.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/net.hpp"
namespace caffe {

template <typename Dtype>
void RecoverScaleFromBN(const LayerParameter& bn_layer_param, LayerParameter& scale_layer_param, Dtype default_scale_weights, Dtype default_scale_bias) {
  CHECK(bn_layer_param.blobs_size() >= 3) << "BatchNorm Layer's blob size must be 3 at least!" << std::endl;
  CHECK(bn_layer_param.type().compare("BatchNorm") == 0) << "Scale layer can only be recovered from batch norm layer!" << std::endl;
  scale_layer_param.set_name("scale_" + bn_layer_param.name());
  scale_layer_param.set_type("Scale");
  scale_layer_param.set_phase(TEST);
  //Assume the scale layer commonly use in-place top/bottom
  scale_layer_param.add_top(const_cast<string &>(bn_layer_param.top(0)));
  scale_layer_param.add_bottom(const_cast<string &>(bn_layer_param.top(0)));
  int bn_layer_blob_size = bn_layer_param.blobs_size();
  //Pre-assumption: scale layer weight and bias blob have same shape
  if (bn_layer_blob_size == 5) {
    scale_layer_param.add_blobs()->CopyFrom(bn_layer_param.blobs(3));
    scale_layer_param.add_blobs()->CopyFrom(bn_layer_param.blobs(4));
  } else if (bn_layer_blob_size == 4) {
    scale_layer_param.add_blobs()->CopyFrom(bn_layer_param.blobs(3));
    Blob<Dtype> scale_bias_blob, scale_weight_blob;
    scale_weight_blob.FromProto(scale_layer_param.blobs(0));
    scale_bias_blob.ReshapeLike(scale_weight_blob);
    caffe_set(scale_bias_blob.count(), default_scale_bias, scale_bias_blob.mutable_cpu_data());
    BlobProto scale_bias_blob_proto;
    scale_bias_blob.ToProto(&scale_bias_blob_proto, false);
    scale_layer_param.add_blobs()->CopyFrom(scale_bias_blob_proto);
  } else {
    Blob<Dtype> scale_weight_blob, scale_bias_blob, bn_mean_blob;
    BlobProto scale_weight_blob_proto, scale_bias_blob_proto;
    bn_mean_blob.FromProto(bn_layer_param.blobs(0));
    vector<int> scale_shape_vec;
    scale_shape_vec.resize(1);
    scale_shape_vec[0] = bn_mean_blob.shape(0);
    scale_weight_blob.Reshape(scale_shape_vec);
    scale_bias_blob.Reshape(scale_shape_vec);
    caffe_set(scale_weight_blob.count(), default_scale_weights, scale_weight_blob.mutable_cpu_data());
    caffe_set(scale_bias_blob.count(), default_scale_bias, scale_bias_blob.mutable_cpu_data());
    scale_weight_blob.ToProto(&scale_weight_blob_proto, false);
    scale_bias_blob.ToProto(&scale_bias_blob_proto, false);
    scale_layer_param.add_blobs()->CopyFrom(scale_weight_blob_proto);
    scale_layer_param.add_blobs()->CopyFrom(scale_bias_blob_proto);
  }
}
void MergeLayer(LayerParameter &layer1,
                const LayerParameter &layer2)
{
  string &layer1_top_blob_name = const_cast<string &>(layer1.top(0));
  const string &layer2_top_blob_name = layer2.top(0);

  // Replace Conv top name with Scale top name
  layer1_top_blob_name.resize(layer2_top_blob_name.size());
  layer1_top_blob_name.replace(0, layer2_top_blob_name.size(), layer2_top_blob_name);
  return;
}


template <typename Dtype>
void AdjustConvLayer(LayerParameter &conv_layer,
                     const LayerParameter &batch_norm_layer,
                     const LayerParameter &scale_layer, bool is_net_init) {
  if (is_net_init) {
    if (!conv_layer.convolution_param().bias_term()) {
      //We will merge batch norm and scale layer to con layer, if conv layer doesn't use bias, adjust it!
      conv_layer.mutable_convolution_param()->set_bias_term(true);
    }
  } else {
    Blob<Dtype> conv_weight_blob, conv_bias_blob;
    Blob<Dtype> scale_weight_blob, scale_bias_blob;
    Blob<Dtype> bn_mean_blob, bn_variance_blob, bn_scale_factor_blob;
    Dtype bn_scale_factor;
    Dtype bn_eps = batch_norm_layer.batch_norm_param().eps();

    conv_weight_blob.FromProto(conv_layer.blobs(0), true);
    if (!conv_layer.convolution_param().bias_term()) {
      conv_layer.mutable_convolution_param()->set_bias_term(true);
      vector<int> conv_bias_shape_vec;
      conv_bias_shape_vec.resize(1);
      conv_bias_shape_vec[0] = conv_weight_blob.shape(0);
      conv_bias_blob.Reshape(conv_bias_shape_vec);
      caffe_set(conv_bias_blob.count(), (Dtype)0, conv_bias_blob.mutable_cpu_data());
      BlobProto conv_bias_blob_proto;
      conv_bias_blob.ToProto(&conv_bias_blob_proto, false);
      conv_layer.add_blobs()->CopyFrom(conv_bias_blob_proto);
    } else {
      conv_bias_blob.FromProto(conv_layer.blobs(1), true);
    }

    //We assume scale layer use weight & bias, but is bias necessary? Need confirm!
    scale_weight_blob.FromProto(scale_layer.blobs(0), true);
    scale_bias_blob.FromProto(scale_layer.blobs(1), true);
    bn_mean_blob.FromProto(batch_norm_layer.blobs(0), true);
    bn_variance_blob.FromProto(batch_norm_layer.blobs(1), true);
    bn_scale_factor_blob.FromProto(batch_norm_layer.blobs(2), true);
    bn_scale_factor = bn_scale_factor_blob.cpu_data()[0] == 0 ? 1 : (1 / bn_scale_factor_blob.cpu_data()[0]);
    CHECK_EQ(bn_variance_blob.shape(0), scale_weight_blob.shape(0));
    CHECK_EQ(conv_weight_blob.shape(0), scale_weight_blob.shape(0));
    CHECK_EQ(scale_weight_blob.count(), bn_variance_blob.count());
    int alpha_count = scale_weight_blob.count();
    Dtype alpha, scale_weight_val, bn_variance_val;
    Dtype * conv_weight_buf = conv_weight_blob.mutable_cpu_data();
    Dtype * conv_bias_buf = conv_bias_blob.mutable_cpu_data();
    const Dtype * scale_bias_buf = scale_bias_blob.cpu_data();
    const Dtype * bn_mean_buf = bn_mean_blob.cpu_data();
    int weight_count = conv_weight_blob.count() / conv_weight_blob.shape(0);
    for (int i = 0; i < alpha_count; i++) {
      scale_weight_val = scale_weight_blob.cpu_data()[i];
      bn_variance_val = bn_variance_blob.cpu_data()[i];
      alpha = scale_weight_val / (std::sqrt(bn_variance_val * bn_scale_factor + bn_eps));
      conv_bias_buf[i] = conv_bias_buf[i] * alpha + (scale_bias_buf[i] -(bn_mean_buf[i] * bn_scale_factor * alpha));
      Dtype * weight_area = conv_weight_buf + i * weight_count;
      caffe_scal(weight_count, alpha, weight_area);

    }
    BlobProto *updated_weight_blob_proto = conv_layer.mutable_blobs(0);
    BlobProto *updated_bias_blob_proto = conv_layer.mutable_blobs(1);
    conv_weight_blob.ToProto(updated_weight_blob_proto);
    conv_bias_blob.ToProto(updated_bias_blob_proto);
  }

}


template <typename Dtype>
void RecoverBNScaleMergedNet(NetParameter * net_param, NetParameter* recovered_net_param) {
  CHECK(net_param != NULL && recovered_net_param != NULL) << "Can NOT recover a NULL network!" << std::endl;
  int kept_bn_layers_num = net_param->compile_net_state().kept_bn_layers_size();
  int idx;
  bool in_kept_list = false;
  for (int i = 0; i < net_param->layer_size(); ++i) {
    const LayerParameter layer_param = net_param->layer(i);
    recovered_net_param->add_layer()->CopyFrom(layer_param);

    if (layer_param.type().compare("BatchNorm") == 0 && layer_param.blobs_size() >= 3) {
      for (idx = 0; idx < kept_bn_layers_num; ++idx) {
        if (layer_param.name().compare(net_param->compile_net_state().kept_bn_layers(idx)) == 0) {
          in_kept_list = true;
          break;
        }
      }

      if (in_kept_list) continue;
      shared_ptr<LayerParameter> scale_layer_param(new LayerParameter());
      RecoverScaleFromBN(layer_param, *scale_layer_param, (Dtype)1, (Dtype)0);
      recovered_net_param->add_layer()->CopyFrom(*scale_layer_param);
    }
  }
}

template <typename Dtype>
void RemoveBNScale(const NetParameter& param, NetParameter* param_compiled) {

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
      Net<Dtype>::GetBlobConsumers(child_layers_params, layer_param->top(0), param, i + 1 < param.layer_size() ? i + 1 : i);
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
        Net<Dtype>::GetBlobConsumers(grandchild_layers_params, child_layer_param.top(0), param, i + 2 < param.layer_size() ? i + 2 : i);
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

template void RecoverScaleFromBN<float>(const LayerParameter& bn_layer_param, LayerParameter& scale_layer_param, float default_scale_weights, float default_scale_bias);
template void RecoverScaleFromBN<double>(const LayerParameter& bn_layer_param, LayerParameter& scale_layer_param, double default_scale_weights, double default_scale_bias);
template void AdjustConvLayer<float>(LayerParameter &conv_layer,
                     const LayerParameter &batch_norm_layer,
                     const LayerParameter &scale_layer, bool is_net_init);

template void AdjustConvLayer<double>(LayerParameter &conv_layer,
                     const LayerParameter &batch_norm_layer,
                     const LayerParameter &scale_layer, bool is_net_init);

template void RecoverBNScaleMergedNet<float>(NetParameter * net_param, NetParameter* recovered_net_param);
template void RecoverBNScaleMergedNet<double>(NetParameter * net_param, NetParameter* recovered_net_param);
template void RemoveBNScale<float>(const NetParameter& param, NetParameter* param_compiled);
template void RemoveBNScale<double>(const NetParameter& param, NetParameter* param_compiled);
}
