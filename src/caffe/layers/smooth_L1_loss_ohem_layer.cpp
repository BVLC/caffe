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


#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <cfloat>


#include "caffe/layers/smooth_L1_loss_ohem_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossOHEMLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3);

  if (!this->layer_param_.loss_param().has_normalization() &&
    this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
    LossParameter_NormalizationMode_VALID :
    LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SmoothL1LossOHEMLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }

  outer_num_ = bottom[0]->num();
  inner_num_ = bottom[0]->height() * bottom[0]->width();

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  // top[2] stores per-instance loss, which takes the shape of N*1*H*W
  if (top.size() >= 2) {
    top[1]->Reshape(
      bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
Dtype SmoothL1LossOHEMLayer<Dtype>::get_normalizer(
  LossParameter_NormalizationMode normalization_mode,
  Dtype pre_fixed_normalizer) {
  Dtype normalizer;
  switch (normalization_mode) {
  case LossParameter_NormalizationMode_FULL:
    normalizer = Dtype(outer_num_ * inner_num_);
    break;
  case LossParameter_NormalizationMode_VALID:
    normalizer = Dtype(outer_num_ * inner_num_);
    break;
  case LossParameter_NormalizationMode_BATCH_SIZE:
    normalizer = Dtype(outer_num_);
    break;
  case LossParameter_NormalizationMode_PRE_FIXED:
    normalizer = pre_fixed_normalizer;
    break;
  case LossParameter_NormalizationMode_NONE:
    normalizer = Dtype(1);
    break;
  default:
    LOG(FATAL) << "Unknown normalization mode: "
      << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SmoothL1LossOHEMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
  
    caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());    // d := b0 - b1
    if (has_weights_) {
      caffe_mul(
        count,
        bottom[2]->cpu_data(),
        diff_.cpu_data(),
        diff_.mutable_cpu_data());  // d := w * (b0 - b1)
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int index = 0; index < count; index++) {
      Dtype val = diff_.cpu_data()[index];
      Dtype abs_val = abs(val);
      if (abs_val < 1) {
        errors_.mutable_cpu_data()[index] = 0.5 * val * val;
      } else {
        errors_.mutable_cpu_data()[index] = abs_val - 0.5;
      }
    }

    Dtype loss = caffe_cpu_asum(count, errors_.cpu_data());

    Dtype pre_fixed_normalizer =
      this->layer_param_.loss_param().pre_fixed_normalizer();
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
      pre_fixed_normalizer);

    // Output per-instance loss
    if (top.size() >= 2) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int i = 0; i < outer_num_; ++i) {
            for (int j = 0; j < inner_num_; j++) {
                Dtype sum = 0;
                for (int c = 0; c < bottom[0]->channels(); ++c) {
                    sum += errors_.cpu_data()[(i * bottom[0]->channels() + c) * inner_num_ + j];
                }
                top[1]->mutable_cpu_data()[i * inner_num_ + j] = sum;
            }
        }
    }
}

#if 0
template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
  const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
	int n = index / spatial_dim;
	int s = index % spatial_dim;
	Dtype sum = 0;
	for (int c = 0; c < channels; ++c) {
	  sum += data[(n * channels + c) * spatial_dim + s];
	}
	channel_sum[index] = sum;
  }
}
#endif

template <typename Dtype>
void SmoothL1LossOHEMLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
    int count = diff_.count();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int index = 0; index < count; index++) {
      Dtype val = diff_.cpu_data()[index];
      Dtype abs_val = abs(val);
      if (abs_val < 1) {
        diff_.mutable_cpu_data()[index] = val;
      } else {
        diff_.mutable_cpu_data()[index] = (Dtype(0) < val) - (val < Dtype(0));
      }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;

        Dtype pre_fixed_normalizer =
          this->layer_param_.loss_param().pre_fixed_normalizer();
        Dtype normalizer = get_normalizer(normalization_, pre_fixed_normalizer);
        Dtype alpha = sign * top[0]->cpu_diff()[0] / normalizer;

        caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.cpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_cpu_diff());  // y
      	}
    }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossOHEMLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossOHEMLayer);
REGISTER_LAYER_CLASS(SmoothL1LossOHEM);

}  // namespace caffe
