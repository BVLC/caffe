/*
All modification made by Intel Corporation: © 2018 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md

A part of the code referenced BVLC CAFFE ristretto branch
For the original code go to https://github.com/pmgysel/caffe 

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

#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "caffe/net.hpp"
#include "caffe/quant/quantization.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;

Quantization::Quantization(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, int score_number, string scaling, int detection, int power) {
  this->model_ = model;
  this->weights_ = weights;
  this->model_quantized_ = model_quantized;
  this->iterations_ = iterations;
  this->trimming_mode_ = trimming_mode;
  this->error_margin_ = error_margin;
  this->score_number = score_number;
  this->scaling = scaling;
  this->detection = detection;
  this->power = power;
}

void Quantization::QuantizeNet() {
  CheckWritePermissions(model_quantized_);

  float accuracy;
  Net<float>* net_test = new Net<float>(model_, caffe::TEST);
  net_test->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(this->iterations_, net_test, &accuracy, true, this->score_number); // RangeInLayer during sampling
  delete net_test;

  // Do network quantization and scoring.
  if (trimming_mode_ == "dynamic_fixed_point") {
    Quantize2DynamicFixedPoint();
  } else {
    LOG(FATAL) << "Unknown trimming mode: " << trimming_mode_;
  }
}

void Quantization::CheckWritePermissions(const string path) {
  std::ofstream probe_ofs(path.c_str());
  if (probe_ofs.good()) {
    probe_ofs.close();
    std::remove(path.c_str());
  } else {
    LOG(FATAL) << "Missing write permissions";
  }
}

void Quantization::RunForwardBatches(const int iterations,
      Net<float>* caffe_net, float* accuracy, const bool do_stats,
      const int score_number) {
  LOG(INFO) << "Running for " << iterations << " iterations.";
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    // Do forward propagation.
    const vector<Blob<float>*>& result =
        caffe_net->Forward(bottom_vec, &iter_loss);
    // Find maximal values in network.
    if(do_stats) {
      caffe_net->RangeInLayers(&layer_names_, &max_in_, &max_out_, &max_params_, this->scaling);
    }
    // Keep track of network score over multiple batches.
    loss += iter_loss;
    if (this->detection) continue;

    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
      }
    }
    LOG(INFO) << "Iteration: " << i;
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  if (this->detection) return;

  for (int i = 0; i < test_score.size(); ++i) {
    const float loss_weight = caffe_net->blob_loss_weights()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
  }
  *accuracy = test_score[score_number] / iterations;
}

void Quantization::Quantize2DynamicFixedPoint() {
  // Find the integer length for dynamic fixed point numbers.
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part.
  // For layer activations, we reduce the integer length by one bit.
  vector<int> lens;
  vector<float> scales;
  for (int i = 0; i < layer_names_.size(); ++i) {
    if (this->power) {
      il_in_.push_back((int)ceil(log2(max_in_[i])));
      il_out_.push_back((int)ceil(log2(max_out_[i])));
    } else {
      scale_in_.push_back(max_in_[i]);
      scale_out_.push_back(max_out_[i]);
    }
    if (this->scaling == "single") {
      if (this->power)
        lens.push_back((int)ceil(log2(max_params_[i][0])+1));
      else
        scales.push_back(max_params_[i][0]);
    } else {
      for (int j = 0; j < max_params_[i].size(); j++) {
        if (this->power)
          lens.push_back((int)ceil(log2(max_params_[i][j])+1));
        else
          scales.push_back(max_params_[i][j]+0.0);
      }
    }
    if (this->power) {
      il_params_.push_back(lens);
      lens.clear();
    } else {
      scale_params_.push_back(scales);
      scales.clear();
    }
  }
  // Debug
  for (int k = 0; k < layer_names_.size(); ++k) {
    if (this->scaling != "single") {
      if (this->power)
        LOG(INFO) << "Layer " << layer_names_[k] << ", parameters channel=" << il_params_[k].size();
      else
        LOG(INFO) << "Layer " << layer_names_[k] << ", parameters channel=" << scale_params_[k].size();
    }

    if (this->power) {
      LOG(INFO) << "Integer length input=" << il_in_[k];
      LOG(INFO) << "Integer length output=" << il_out_[k];
    } else {
      LOG(INFO) << "Scale input=" << scale_in_[k];
      LOG(INFO) << "Scale output=" << scale_out_[k];
    }
   
    if (this->scaling == "single") {
      if (this->power)
        LOG(INFO) << "Integer length param=" << il_params_[k][0];
      else
        LOG(INFO) << "Scale param=" << scale_params_[k][0];
    } else {
      if (this->power){
        for (int j = 0; j < il_params_[k].size(); j++) {
          LOG(INFO) << "Integer length params[" << j << "]=" << il_params_[k][j];
        }
      } else{
        for (int j = 0; j < scale_params_[k].size(); j++) {
          LOG(INFO) << "Scale params[" << j << "]=" << scale_params_[k][j];
        }
      }
    }
  }

  // Choose bit-width for different network parts
  bw_conv_params_ = 8; 
  bw_out_ = 8;
  bw_in_ = bw_out_;

  NetParameter param;
  // Score dynamic fixed point network.
  // This network combines dynamic fixed point parameters in convolutional and
  // inner product layers, as well as dynamic fixed point activations.
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  EditNetDescriptionDynamicFixedPoint(&param, "Convolution",
      "Parameters_and_Activations", bw_conv_params_, bw_in_,
      bw_out_);
  WriteProtoToTextFile(param, model_quantized_);
}

void Quantization::EditNetDescriptionDynamicFixedPoint(NetParameter* param,
      const string layer_quantize, const string net_part, const int bw_conv,
      const int bw_in, const int bw_out) {
  int index = 0;
  bool first_convolution = false;
  for (int i = 0; i < param->layer_size(); ++i) {
    // TODO: move first convolution check to transform script
    if (layer_quantize.find("Convolution") != string::npos &&
        param->layer(i).type().find("Convolution") != string::npos) {
      if (!first_convolution) {
          first_convolution = true;
          continue;
      }

      // quantize parameters
      if (net_part.find("Parameters") != string::npos) {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("Convolution");
        if (trimming_mode_ == "dynamic_fixed_point") {
          param_layer->mutable_quantization_param()->set_bw_params(bw_conv);
          if (this->power) {
            vector<int> vals = GetIntegerLengthParams(param->layer(i).name());
            for (int j = 0; j < vals.size(); j++) {
              vals[j] = bw_conv - vals[j];
              param_layer->mutable_quantization_param()->add_fl_params(vals[j]);
            }
          } else {
            vector<float> vals = GetScaleParams(param->layer(i).name());
            for (int j = 0; j < vals.size(); j++) {
              param_layer->mutable_quantization_param()->add_scale_params(vals[j]);
            }
          }
        }
      }
      // quantize activations
      if (net_part.find("Activations") != string::npos) {
        LayerParameter* param_layer = param->mutable_layer(i);
        param_layer->set_type("Convolution");
        if (trimming_mode_ == "dynamic_fixed_point") {
          param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
          param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
          if (this->power) {
            int val = GetIntegerLengthIn(param->layer(i).name());
            param_layer->mutable_quantization_param()->add_fl_layer_in(bw_in - val);
            val = GetIntegerLengthOut(param->layer(i).name());
            param_layer->mutable_quantization_param()->add_fl_layer_out(bw_out - val);
          } else {
            float val = GetScaleIn(param->layer(i).name());
            param_layer->mutable_quantization_param()->add_scale_in(val);
            val = GetScaleOut(param->layer(i).name());
            param_layer->mutable_quantization_param()->add_scale_out(val);
          }
        }
      }
      LayerParameter* param_layer = param->mutable_layer(i);
      if (trimming_mode_ == "dynamic_fixed_point") {
        param_layer->mutable_quantization_param()->set_precision(caffe::QuantizationParameter_Precision(0));
      } else {
        LOG(FATAL) << "Unknown trimming mode: " << trimming_mode_;
      }
      index++;
    }
  }
}

vector<int> Quantization::GetIntegerLengthParams(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_params_[pos];
}

int Quantization::GetIntegerLengthIn(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_in_[pos];
}

int Quantization::GetIntegerLengthOut(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_out_[pos];
}

vector<float> Quantization::GetScaleParams(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return scale_params_[pos];
}

float Quantization::GetScaleIn(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return scale_in_[pos];
}

float Quantization::GetScaleOut(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return scale_out_[pos];
}
