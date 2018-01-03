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

#ifndef QUANTIZATION_HPP_
#define QUANTIZATION_HPP_

#include "caffe/caffe.hpp"

using caffe::string;
using caffe::vector;
using caffe::Net;

/**
 * @brief Approximate 32-bit floating point networks.
 *
 * This is the Ristretto tool. Use it to generate file descriptions of networks
 * which use reduced word width arithmetic.
 */
class Quantization {
public:
  explicit Quantization(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, int score_number, string scaling="single", int detection=0, int power=1);
  void QuantizeNet();
private:
  void CheckWritePermissions(const string path);
  /**
   * @brief Score network.
   * @param accuracy Reports the network's accuracy according to
   * accuracy_number.
   * @param do_stats: Find the maximal values in each layer.
   * @param score_number The accuracy layer that matters.
   *
   * For networks with multiple accuracy layers, set score_number to the
   * appropriate value. For example, for BVLC GoogLeNet, use score_number=7.
   */
  void RunForwardBatches(const int iterations, Net<float>* caffe_net,
      float* accuracy, const bool do_stats = false, const int score_number = 0); // 7 for GoogleNet-V1
  /**
   * @brief Quantize convolutional and fully connected layers to dynamic fixed
   * point.
   * The parameters and layer activations get quantized and the resulting
   * network will be tested.
   * Find the required number of bits required for parameters and layer
   * activations (which might differ from each other).
   */
  void Quantize2DynamicFixedPoint();

  /**
   * @brief Change network to dynamic fixed point.
   */
  void EditNetDescriptionDynamicFixedPoint(caffe::NetParameter* param,
      const string layer_quantize, const string network_part,
      const int bw_conv, const int bw_in, const int bw_out);
  
  vector<int> GetIntegerLengthParams(const string layer_name);
  vector<float> GetScaleParams(const string layer_name);
  /**
   * @brief Find the integer length for dynamic fixed point inputs of a certain
   * layer.
   */
  int GetIntegerLengthIn(const string layer_name);
  float GetScaleIn(const string layer_name);
  /**
   * @brief Find the integer length for dynamic fixed point outputs of a certain
   * layer.
   */
  int GetIntegerLengthOut(const string layer_name);
  float GetScaleOut(const string layer_name);

  string model_;
  string weights_;
  string model_quantized_;
  int iterations_;
  string trimming_mode_;
  double error_margin_;
  int score_number;
  string scaling;
  int detection;
  int power;
  float test_score_baseline_;
  // The maximal absolute values of layer inputs, parameters and
  // layer outputs.
  vector<float> max_in_, max_out_;
  vector<vector<float>> max_params_;

  // The integer bits for dynamic fixed point layer inputs, parameters and
  // layer outputs.
  vector<int> il_in_, il_out_;
  vector<vector<int>> il_params_;

  vector<float> scale_in_, scale_out_;
  vector<vector<float>> scale_params_;
  // The name of the layers that need to be quantized to dynamic fixed point.
  vector<string> layer_names_;
  // The number of bits used for dynamic fixed point layer inputs, parameters
  // and layer outputs.
  int bw_in_, bw_conv_params_, bw_fc_params_, bw_out_;
};

#endif // QUANTIZATION_HPP_
