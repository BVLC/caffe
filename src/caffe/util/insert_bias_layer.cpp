#ifdef CAFFE_MSL

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_bias_layer.hpp"

namespace caffe {

void SeparateBias(const NetParameter& param, NetParameter* param_split) {
  param_split->CopyFrom(param);
  param_split->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
      LayerParameter* layer_param = param_split->add_layer();
      layer_param->CopyFrom(param.layer(i));
      if(layer_param->type() == "InnerProduct" || layer_param->type() == "Convolution") {
          bool add_bias_layer = false;
          FillerParameter* bias_filler = NULL;

          if(layer_param->has_convolution_param() && layer_param->convolution_param().bias_term() == true) {
              add_bias_layer = true;
              layer_param->mutable_convolution_param()->set_bias_term(false);
              if(layer_param->convolution_param().has_bias_filler())
                  bias_filler = layer_param->mutable_convolution_param()->mutable_bias_filler();
          }
          if(layer_param->has_inner_product_param() && layer_param->inner_product_param().bias_term() == true) {
              add_bias_layer = true;
              layer_param->mutable_inner_product_param()->set_bias_term(false);
              if(layer_param->inner_product_param().has_bias_filler())
                  bias_filler = layer_param->mutable_inner_product_param()->mutable_bias_filler();
          }
          if(add_bias_layer) {
              LayerParameter* bias_param = param_split->add_layer();
              bias_param->Clear();
              bias_param->add_bottom(layer_param->top(0));
              bias_param->add_top(layer_param->top(0));
              bias_param->set_name(layer_param->top(0) + "_bias");
              bias_param->set_type("Bias");
              bias_param->mutable_bias_param()->set_axis(1);
              if(bias_filler != NULL) {
                  bias_param->mutable_bias_param()->mutable_filler()->CopyFrom(*bias_filler);
                  if(layer_param->has_convolution_param() && layer_param->convolution_param().has_bias_filler()) layer_param->mutable_convolution_param()->clear_bias_filler();
                  if(layer_param->has_inner_product_param() && layer_param->inner_product_param().has_bias_filler()) layer_param->mutable_inner_product_param()->clear_bias_filler();
              }
              if(layer_param->param_size() > 1) {
                  layer_param->clear_param();
                  ParamSpec* param_spec = layer_param->add_param();
                  param_spec->CopyFrom(param.layer(i).param(0));
                  bias_param->add_param()->CopyFrom(param.layer(i).param(1));
              }
          }
      }
  }
}  
}

#endif /* CAFFE_MSL */