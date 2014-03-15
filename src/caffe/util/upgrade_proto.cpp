// Copyright 2014 Jeff Donahue

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

#include <algorithm>
#include <string>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/common.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/deprecated/caffe_v0_to_v1_bridge.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;

namespace caffe {

bool UpgradeV0LayerConnection(const V0LayerConnection& v0_layer_connection,
                              LayerParameter* layer_param) {
  bool full_compatibility = true;
  layer_param->Clear();
  for (int i = 0; i < v0_layer_connection.bottom_size(); ++i) {
    layer_param->add_bottom(v0_layer_connection.bottom(i));
  }
  for (int i = 0; i < v0_layer_connection.top_size(); ++i) {
    layer_param->add_top(v0_layer_connection.top(i));
  }
  if (v0_layer_connection.has_layer()) {
    const V0LayerParameter& v0_layer_param = v0_layer_connection.layer();
    if (v0_layer_param.has_name()) {
      layer_param->set_name(v0_layer_param.name());
    }
    const string& type = v0_layer_param.type();
    if (v0_layer_param.has_num_output()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_num_output(
            v0_layer_param.num_output());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_num_output(
            v0_layer_param.num_output());
      } else {
        LOG(ERROR) << "Unknown parameter num_output for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_biasterm()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_bias_term(
            v0_layer_param.biasterm());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_bias_term(
            v0_layer_param.biasterm());
      } else {
        LOG(ERROR) << "Unknown parameter biasterm for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_weight_filler()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->
            mutable_weight_filler()->CopyFrom(v0_layer_param.weight_filler());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->
            mutable_weight_filler()->CopyFrom(v0_layer_param.weight_filler());
      } else {
        LOG(ERROR) << "Unknown parameter weight_filler for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_bias_filler()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->
            mutable_bias_filler()->CopyFrom(v0_layer_param.bias_filler());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->
            mutable_bias_filler()->CopyFrom(v0_layer_param.bias_filler());
      } else {
        LOG(ERROR) << "Unknown parameter bias_filler for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_pad()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_pad(v0_layer_param.pad());
      } else {
        LOG(ERROR) << "Unknown parameter pad for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_kernelsize()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_kernel_size(
            v0_layer_param.kernelsize());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_kernel_size(
            v0_layer_param.kernelsize());
      } else {
        LOG(ERROR) << "Unknown parameter kernelsize for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_stride()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_stride(
            v0_layer_param.stride());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_stride(
            v0_layer_param.stride());
      } else {
        LOG(ERROR) << "Unknown parameter stride for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_pool()) {
      if (type == "pool") {
        V0LayerParameter_PoolMethod pool = v0_layer_param.pool();
        switch (pool) {
        V0LayerParameter_PoolMethod_MAX:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_MAX);
          break;
        V0LayerParameter_PoolMethod_AVE:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_AVE);
          break;
        V0LayerParameter_PoolMethod_STOCHASTIC:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_STOCHASTIC);
          break;
        default:
          LOG(ERROR) << "Unknown pool method " << pool;
          full_compatibility = false;
        }
      } else {
        LOG(ERROR) << "Unknown parameter pool for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_dropout_ratio()) {
      if (type == "dropout_ratio") {
        layer_param->mutable_dropout_param()->set_dropout_ratio(
            v0_layer_param.dropout_ratio());
      } else {
        LOG(ERROR) << "Unknown parameter dropout_ratio for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_local_size()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_local_size(
            v0_layer_param.local_size());
      } else {
        LOG(ERROR) << "Unknown parameter local_size for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_alpha()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_alpha(v0_layer_param.alpha());
      } else {
        LOG(ERROR) << "Unknown parameter alpha for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_beta()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_beta(v0_layer_param.beta());
      } else {
        LOG(ERROR) << "Unknown parameter beta for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_source()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_source(v0_layer_param.source());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "infogain_loss") {
        layer_param->mutable_infogain_loss_param()->set_source(
            v0_layer_param.source());
      } else {
        LOG(ERROR) << "Unknown parameter source for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_scale()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_scale(v0_layer_param.scale());
      } else {
        LOG(ERROR) << "Unknown parameter scale for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_meanfile()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_mean_file(v0_layer_param.meanfile());
      } else {
        LOG(ERROR) << "Unknown parameter meanfile for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_batchsize()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else {
        LOG(ERROR) << "Unknown parameter batchsize for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_cropsize()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_crop_size(
            v0_layer_param.cropsize());
      } else {
        LOG(ERROR) << "Unknown parameter cropsize for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_mirror()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_mirror(v0_layer_param.mirror());
      } else {
        LOG(ERROR) << "Unknown parameter mirror for layer type " << type;
        full_compatibility = false;
      }
    }
    for (int i = 0; i < v0_layer_param.blobs_size(); ++i) {
      layer_param->add_blobs()->CopyFrom(v0_layer_param.blobs(i));
    }
    for (int i = 0; i < v0_layer_param.blobs_lr_size(); ++i) {
      layer_param->add_blobs_lr(v0_layer_param.blobs_lr(i));
    }
    for (int i = 0; i < v0_layer_param.weight_decay_size(); ++i) {
      layer_param->add_weight_decay(v0_layer_param.weight_decay(i));
    }
    if (v0_layer_param.has_rand_skip()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else {
        LOG(ERROR) << "Unknown parameter rand_skip for layer type " << type;
        full_compatibility = false;
      }
    }
    if (v0_layer_param.has_concat_dim()) {
      if (type == "concat") {
        layer_param->mutable_concat_param()->set_concat_dim(
            v0_layer_param.concat_dim());
      } else {
        LOG(ERROR) << "Unknown parameter concat_dim for layer type " << type;
        full_compatibility = false;
      }
    }
  }
  return full_compatibility;
}

}  // namespace caffe
