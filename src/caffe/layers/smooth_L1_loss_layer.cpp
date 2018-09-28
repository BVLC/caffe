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

// ------------------------------------------------------------------
// Fast R-CNN
// copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// Modified by Wei Liu
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  has_weights_ = (bottom.size() >= 3);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both inside and outside weights";
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
    CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[3]->height());
    CHECK_EQ(bottom[0]->width(), bottom[3]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  // vector of ones used to sum
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  for (int i = 0; i < bottom[0]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
    int count = bottom[0]->count();
    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());    // d := b0 - b1
    if (has_weights_) {
        // apply "inside" weights
        caffe_mul(count, bottom[2]->cpu_data(), diff_.cpu_data(), diff_.mutable_cpu_data());  // d := w_in * (b0 - b1)
    }

    for(int i = 0; i < count; i++) {
        Dtype val = diff_.cpu_data()[i];
        Dtype abs_val = fabs(val);
        if (abs_val < 1.0 / sigma2_) {
           errors_.mutable_cpu_data()[i] = 0.5 * val * val * sigma2_;
        } 
        else {
           errors_.mutable_cpu_data()[i] = abs_val - 0.5 / sigma2_;
        }
    }
    if (has_weights_) {
        // apply "outside" weights
        caffe_mul(count, bottom[3]->cpu_data(), errors_.cpu_data(), errors_.mutable_cpu_data());  // d := w_out * SmoothL1(w_in * (b0 - b1))
    }

    Dtype loss = caffe_cpu_dot(count, ones_.cpu_data(), errors_.cpu_data());
    Dtype normalizer = Dtype(bottom[0]->num());
    if (this->layer_param_.loss_param().normalization() == LossParameter_NormalizationMode_NONE)
      normalizer = 1.f;
#ifdef USE_MLSL
    else {
      // We assume local bs is same across all nodes
      normalizer *= mn::get_group_size();
    }
#endif
    top[0]->mutable_cpu_data()[0] = loss / normalizer;
}


template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    // after forwards, diff_ holds w_in * (b0 - b1)
    int count = diff_.count();
    
    Dtype normalizer = Dtype(bottom[0]->num());
    if (this->layer_param_.loss_param().normalization() == LossParameter_NormalizationMode_NONE)
      normalizer = 1.f;
#ifdef USE_MLSL
    else {
      // We assume local bs is same across all nodes
      normalizer *= mn::get_group_size();
    }
#endif
    for (int i = 0; i < count; i++) {
        // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
        //       = sign(x)                   otherwise
        Dtype val = diff_.cpu_data()[i];
        Dtype abs_val = fabs(val);
        if (abs_val < 1.0 / sigma2_) {
          diff_.mutable_cpu_data()[i] = sigma2_ * val;
        } 
        else {
          diff_.mutable_cpu_data()[i] = (Dtype(0) < val) - (val < Dtype(0));
        }
    }

    for (int i = 0; i < 2; ++i) {
        if (propagate_down[i])  {
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / normalizer;
            caffe_cpu_axpby(
              count,                           // count
              alpha,                           // alpha
              diff_.cpu_data(),                // x
              Dtype(0),                        // beta
              bottom[i]->mutable_cpu_diff());  // y
            if (has_weights_)  {
                // Scale by "inside" weight
                caffe_mul(
                    count,
                    bottom[2]->cpu_data(),
                    bottom[i]->cpu_diff(),
                    bottom[i]->mutable_cpu_diff());
                // Scale by "outside" weight
                caffe_mul(
                    count,
                    bottom[3]->cpu_data(),
                    bottom[i]->cpu_diff(),
                    bottom[i]->mutable_cpu_diff());
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
