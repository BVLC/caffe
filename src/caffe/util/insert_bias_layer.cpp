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

#endif /* USE_MLSL */
