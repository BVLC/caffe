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

#ifndef CAFFE_PRELU_LAYER_HPP_
#define CAFFE_PRELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Parameterized Rectified Linear Unit non-linearity @f$
 *        y_i = \max(0, x_i) + a_i \min(0, x_i)
 *        @f$. The differences from ReLULayer are 1) negative slopes are
 *        learnable though backprop and 2) negative slopes can vary across
 *        channels. The number of axes of input blob should be greater than or
 *        equal to 2. The 1st axis (0-based) is seen as channels.
 */
template <typename Dtype>
class PReLULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides PReLUParameter prelu_param,
   *     with PReLULayer options:
   *   - filler (\b optional, FillerParameter,
   *     default {'type': constant 'value':0.25}).
   *   - channel_shared (\b optional, default false).
   *     negative slopes are shared across channels.
   */
  explicit PReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PReLU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times ...) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times ...) @f$
   *      the computed outputs for each channel @f$i@f$ @f$
   *        y_i = \max(0, x_i) + a_i \min(0, x_i)
   *      @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the PReLU inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times ...) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times ...) @f$
   *      the inputs @f$ x @f$; For each channel @f$i@f$, backward fills their
   *      diff with gradients @f$
   *        \frac{\partial E}{\partial x_i} = \left\{
   *        \begin{array}{lr}
   *            a_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
   *            \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i > 0
   *        \end{array} \right.
   *      @f$.
   *      If param_propagate_down_[0] is true, it fills the diff with gradients
   *      @f$
   *        \frac{\partial E}{\partial a_i} = \left\{
   *        \begin{array}{lr}
   *            \sum_{x_i} x_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
   *            0 & \mathrm{if} \; x_i > 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool channel_shared_;
  Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
  Blob<Dtype> bottom_memory_;  // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_PRELU_LAYER_HPP_
