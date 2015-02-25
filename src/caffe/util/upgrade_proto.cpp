#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <map>
#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

bool NetNeedsUpgrade(const NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).has_layer()) {
      return true;
    }
  }
  return false;
}

bool UpgradeV0Net(const NetParameter& v0_net_param_padding_layers,
                  NetParameter* net_param) {
  // First upgrade padding layers to padded conv layers.
  NetParameter v0_net_param;
  UpgradeV0PaddingLayers(v0_net_param_padding_layers, &v0_net_param);
  // Now upgrade layer parameters.
  bool is_fully_compatible = true;
  net_param->Clear();
  if (v0_net_param.has_name()) {
    net_param->set_name(v0_net_param.name());
  }
  for (int i = 0; i < v0_net_param.layers_size(); ++i) {
    is_fully_compatible &= UpgradeLayerParameter(v0_net_param.layers(i),
                                                 net_param->add_layers());
  }
  for (int i = 0; i < v0_net_param.input_size(); ++i) {
    net_param->add_input(v0_net_param.input(i));
  }
  for (int i = 0; i < v0_net_param.input_dim_size(); ++i) {
    net_param->add_input_dim(v0_net_param.input_dim(i));
  }
  if (v0_net_param.has_force_backward()) {
    net_param->set_force_backward(v0_net_param.force_backward());
  }
  return is_fully_compatible;
}

void UpgradeV0PaddingLayers(const NetParameter& param,
                            NetParameter* param_upgraded_pad) {
  // Copy everything other than the layers from the original param.
  param_upgraded_pad->Clear();
  param_upgraded_pad->CopyFrom(param);
  param_upgraded_pad->clear_layers();
  // Figure out which layer each bottom blob comes from.
  map<string, int> blob_name_to_last_top_idx;
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    blob_name_to_last_top_idx[blob_name] = -1;
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerParameter& layer_connection = param.layers(i);
    const V0LayerParameter& layer_param = layer_connection.layer();
    // Add the layer to the new net, unless it's a padding layer.
    if (layer_param.type() != "padding") {
      param_upgraded_pad->add_layers()->CopyFrom(layer_connection);
    }
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown blob input " << blob_name << " to layer " << j;
      }
      const int top_idx = blob_name_to_last_top_idx[blob_name];
      if (top_idx == -1) {
        continue;
      }
      LayerParameter source_layer = param.layers(top_idx);
      if (source_layer.layer().type() == "padding") {
        // This layer has a padding layer as input -- check that it is a conv
        // layer or a pooling layer and takes only one input.  Also check that
        // the padding layer input has only one input and one output.  Other
        // cases have undefined behavior in Caffe.
        CHECK((layer_param.type() == "conv") || (layer_param.type() == "pool"))
            << "Padding layer input to "
            "non-convolutional / non-pooling layer type "
            << layer_param.type();
        CHECK_EQ(layer_connection.bottom_size(), 1)
            << "Conv Layer takes a single blob as input.";
        CHECK_EQ(source_layer.bottom_size(), 1)
            << "Padding Layer takes a single blob as input.";
        CHECK_EQ(source_layer.top_size(), 1)
            << "Padding Layer produces a single blob as output.";
        int layer_index = param_upgraded_pad->layers_size() - 1;
        param_upgraded_pad->mutable_layers(layer_index)->mutable_layer()
            ->set_pad(source_layer.layer().pad());
        param_upgraded_pad->mutable_layers(layer_index)
            ->set_bottom(j, source_layer.bottom(0));
      }
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      blob_name_to_last_top_idx[blob_name] = i;
    }
  }
}

bool UpgradeLayerParameter(const LayerParameter& v0_layer_connection,
                           LayerParameter* layer_param) {
  bool is_fully_compatible = true;
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
    if (v0_layer_param.has_type()) {
      layer_param->set_type(UpgradeV0LayerType(type));
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
    if (v0_layer_param.has_num_output()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_num_output(
            v0_layer_param.num_output());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_num_output(
            v0_layer_param.num_output());
      } else {
        LOG(ERROR) << "Unknown parameter num_output for layer type " << type;
        is_fully_compatible = false;
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
        is_fully_compatible = false;
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
        is_fully_compatible = false;
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
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_pad()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_pad(v0_layer_param.pad());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_pad(v0_layer_param.pad());
      } else {
        LOG(ERROR) << "Unknown parameter pad for layer type " << type;
        is_fully_compatible = false;
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
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_group()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_group(
            v0_layer_param.group());
      } else {
        LOG(ERROR) << "Unknown parameter group for layer type " << type;
        is_fully_compatible = false;
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
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_pool()) {
      if (type == "pool") {
        V0LayerParameter_PoolMethod pool = v0_layer_param.pool();
        switch (pool) {
        case V0LayerParameter_PoolMethod_MAX:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_MAX);
          break;
        case V0LayerParameter_PoolMethod_AVE:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_AVE);
          break;
        case V0LayerParameter_PoolMethod_STOCHASTIC:
          layer_param->mutable_pooling_param()->set_pool(
              PoolingParameter_PoolMethod_STOCHASTIC);
          break;
        default:
          LOG(ERROR) << "Unknown pool method " << pool;
          is_fully_compatible = false;
        }
      } else {
        LOG(ERROR) << "Unknown parameter pool for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_dropout_ratio()) {
      if (type == "dropout") {
        layer_param->mutable_dropout_param()->set_dropout_ratio(
            v0_layer_param.dropout_ratio());
      } else {
        LOG(ERROR) << "Unknown parameter dropout_ratio for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_local_size()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_local_size(
            v0_layer_param.local_size());
      } else {
        LOG(ERROR) << "Unknown parameter local_size for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_alpha()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_alpha(v0_layer_param.alpha());
      } else {
        LOG(ERROR) << "Unknown parameter alpha for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_beta()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_beta(v0_layer_param.beta());
      } else {
        LOG(ERROR) << "Unknown parameter beta for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_source()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_source(v0_layer_param.source());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "infogain_loss") {
        layer_param->mutable_infogain_loss_param()->set_source(
            v0_layer_param.source());
      } else {
        LOG(ERROR) << "Unknown parameter source for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_scale()) {
      layer_param->mutable_transform_param()->
          set_scale(v0_layer_param.scale());
    }
    if (v0_layer_param.has_meanfile()) {
      layer_param->mutable_transform_param()->
          set_mean_file(v0_layer_param.meanfile());
    }
    if (v0_layer_param.has_batchsize()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else {
        LOG(ERROR) << "Unknown parameter batchsize for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_cropsize()) {
      layer_param->mutable_transform_param()->
          set_crop_size(v0_layer_param.cropsize());
    }
    if (v0_layer_param.has_mirror()) {
      layer_param->mutable_transform_param()->
          set_mirror(v0_layer_param.mirror());
    }
    if (v0_layer_param.has_rand_skip()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else {
        LOG(ERROR) << "Unknown parameter rand_skip for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_shuffle_images()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_shuffle(
            v0_layer_param.shuffle_images());
      } else {
        LOG(ERROR) << "Unknown parameter shuffle for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_new_height()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_new_height(
            v0_layer_param.new_height());
      } else {
        LOG(ERROR) << "Unknown parameter new_height for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_new_width()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_new_width(
            v0_layer_param.new_width());
      } else {
        LOG(ERROR) << "Unknown parameter new_width for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_concat_dim()) {
      if (type == "concat") {
        layer_param->mutable_concat_param()->set_concat_dim(
            v0_layer_param.concat_dim());
      } else {
        LOG(ERROR) << "Unknown parameter concat_dim for layer type " << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_fg_threshold()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_fg_threshold(
            v0_layer_param.det_fg_threshold());
      } else {
        LOG(ERROR) << "Unknown parameter det_fg_threshold for layer type "
                   << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_bg_threshold()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_bg_threshold(
            v0_layer_param.det_bg_threshold());
      } else {
        LOG(ERROR) << "Unknown parameter det_bg_threshold for layer type "
                   << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_fg_fraction()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_fg_fraction(
            v0_layer_param.det_fg_fraction());
      } else {
        LOG(ERROR) << "Unknown parameter det_fg_fraction for layer type "
                   << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_context_pad()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_context_pad(
            v0_layer_param.det_context_pad());
      } else {
        LOG(ERROR) << "Unknown parameter det_context_pad for layer type "
                   << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_crop_mode()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_crop_mode(
            v0_layer_param.det_crop_mode());
      } else {
        LOG(ERROR) << "Unknown parameter det_crop_mode for layer type "
                   << type;
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_hdf5_output_param()) {
      if (type == "hdf5_output") {
        layer_param->mutable_hdf5_output_param()->CopyFrom(
            v0_layer_param.hdf5_output_param());
      } else {
        LOG(ERROR) << "Unknown parameter hdf5_output_param for layer type "
                   << type;
        is_fully_compatible = false;
      }
    }
  }
  return is_fully_compatible;
}

LayerParameter_LayerType UpgradeV0LayerType(const string& type) {
  if (type == "accuracy") {
    return LayerParameter_LayerType_ACCURACY;
  } else if (type == "bnll") {
    return LayerParameter_LayerType_BNLL;
  } else if (type == "concat") {
    return LayerParameter_LayerType_CONCAT;
  } else if (type == "conv") {
    return LayerParameter_LayerType_CONVOLUTION;
  } else if (type == "data") {
    return LayerParameter_LayerType_DATA;
  } else if (type == "dropout") {
    return LayerParameter_LayerType_DROPOUT;
  } else if (type == "euclidean_loss") {
    return LayerParameter_LayerType_EUCLIDEAN_LOSS;
  } else if (type == "flatten") {
    return LayerParameter_LayerType_FLATTEN;
  } else if (type == "hdf5_data") {
    return LayerParameter_LayerType_HDF5_DATA;
  } else if (type == "hdf5_output") {
    return LayerParameter_LayerType_HDF5_OUTPUT;
  } else if (type == "im2col") {
    return LayerParameter_LayerType_IM2COL;
  } else if (type == "images") {
    return LayerParameter_LayerType_IMAGE_DATA;
  } else if (type == "infogain_loss") {
    return LayerParameter_LayerType_INFOGAIN_LOSS;
  } else if (type == "innerproduct") {
    return LayerParameter_LayerType_INNER_PRODUCT;
  } else if (type == "lrn") {
    return LayerParameter_LayerType_LRN;
  } else if (type == "multinomial_logistic_loss") {
    return LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS;
  } else if (type == "pool") {
    return LayerParameter_LayerType_POOLING;
  } else if (type == "relu") {
    return LayerParameter_LayerType_RELU;
  } else if (type == "sigmoid") {
    return LayerParameter_LayerType_SIGMOID;
  } else if (type == "softmax") {
    return LayerParameter_LayerType_SOFTMAX;
  } else if (type == "softmax_loss") {
    return LayerParameter_LayerType_SOFTMAX_LOSS;
  } else if (type == "split") {
    return LayerParameter_LayerType_SPLIT;
  } else if (type == "tanh") {
    return LayerParameter_LayerType_TANH;
  } else if (type == "window_data") {
    return LayerParameter_LayerType_WINDOW_DATA;
  } else {
    LOG(FATAL) << "Unknown layer name: " << type;
    return LayerParameter_LayerType_NONE;
  }
}

bool NetNeedsDataUpgrade(const NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).type() == LayerParameter_LayerType_DATA) {
      DataParameter layer_param = net_param.layers(i).data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
    if (net_param.layers(i).type() == LayerParameter_LayerType_IMAGE_DATA) {
      ImageDataParameter layer_param = net_param.layers(i).image_data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
    if (net_param.layers(i).type() == LayerParameter_LayerType_WINDOW_DATA) {
      WindowDataParameter layer_param = net_param.layers(i).window_data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
  }
  return false;
}

#define CONVERT_LAYER_TRANSFORM_PARAM(TYPE, Name, param_name) \
  do { \
    if (net_param->layers(i).type() == LayerParameter_LayerType_##TYPE) { \
      Name##Parameter* layer_param = \
          net_param->mutable_layers(i)->mutable_##param_name##_param(); \
      TransformationParameter* transform_param = \
          net_param->mutable_layers(i)->mutable_transform_param(); \
      if (layer_param->has_scale()) { \
        transform_param->set_scale(layer_param->scale()); \
        layer_param->clear_scale(); \
      } \
      if (layer_param->has_mean_file()) { \
        transform_param->set_mean_file(layer_param->mean_file()); \
        layer_param->clear_mean_file(); \
      } \
      if (layer_param->has_crop_size()) { \
        transform_param->set_crop_size(layer_param->crop_size()); \
        layer_param->clear_crop_size(); \
      } \
      if (layer_param->has_mirror()) { \
        transform_param->set_mirror(layer_param->mirror()); \
        layer_param->clear_mirror(); \
      } \
    } \
  } while (0)

void UpgradeNetDataTransformation(NetParameter* net_param) {
  for (int i = 0; i < net_param->layers_size(); ++i) {
    CONVERT_LAYER_TRANSFORM_PARAM(DATA, Data, data);
    CONVERT_LAYER_TRANSFORM_PARAM(IMAGE_DATA, ImageData, image_data);
    CONVERT_LAYER_TRANSFORM_PARAM(WINDOW_DATA, WindowData, window_data);
  }
}

void NetParameterToPrettyPrint(const NetParameter& param,
                               NetParameterPrettyPrint* pretty_param) {
  pretty_param->Clear();
  if (param.has_name()) {
    pretty_param->set_name(param.name());
  }
  if (param.has_force_backward()) {
    pretty_param->set_force_backward(param.force_backward());
  }
  for (int i = 0; i < param.input_size(); ++i) {
    pretty_param->add_input(param.input(i));
  }
  for (int i = 0; i < param.input_dim_size(); ++i) {
    pretty_param->add_input_dim(param.input_dim(i));
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    pretty_param->add_layers()->CopyFrom(param.layers(i));
  }
}

void UpgradeNetAsNeeded(const string& param_file, NetParameter* param) {
  if (NetNeedsUpgrade(*param)) {
    // NetParameter was specified using the old style (V0LayerParameter); try to
    // upgrade it.
    LOG(ERROR) << "Attempting to upgrade input file specified using deprecated "
               << "V0LayerParameter: " << param_file;
    NetParameter original_param(*param);
    if (!UpgradeV0Net(original_param, param)) {
      LOG(ERROR) << "Warning: had one or more problems upgrading "
          << "V0NetParameter to NetParameter (see above); continuing anyway.";
    } else {
      LOG(INFO) << "Successfully upgraded file specified using deprecated "
                << "V0LayerParameter";
    }
    LOG(ERROR) << "Note that future Caffe releases will not support "
        << "V0NetParameter; use ./build/tools/upgrade_net_proto_text for "
        << "prototxt and ./build/tools/upgrade_net_proto_binary for model "
        << "weights upgrade this and any other net protos to the new format.";
  }
  // NetParameter uses old style data transformation fields; try to upgrade it.
  if (NetNeedsDataUpgrade(*param)) {
    LOG(ERROR) << "Attempting to upgrade input file specified using deprecated "
               << "transformation parameters: " << param_file;
    UpgradeNetDataTransformation(param);
    LOG(INFO) << "Successfully upgraded file specified using deprecated "
              << "data transformation parameters.";
    LOG(ERROR) << "Note that future Caffe releases will only support "
               << "transform_param messages for transformation fields.";
  }
}

void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param) {
  CHECK(ReadProtoFromTextFile(param_file, param))
      << "Failed to parse NetParameter file: " << param_file;
  UpgradeNetAsNeeded(param_file, param);
}

void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param) {
  CHECK(ReadProtoFromBinaryFile(param_file, param))
      << "Failed to parse NetParameter file: " << param_file;
  UpgradeNetAsNeeded(param_file, param);
}

}  // namespace caffe
