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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  // Initialize with the first blob.
  vector<int> top_shape = bottom[0]->shape();
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  int bottom_count_sum = bottom[0]->count();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += bottom[i]->count();
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom_count_sum, top[0]->count());
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int offset_value = offset_concat_axis;
    offset_concat_axis += bottom_concat_axis;
#ifdef _OPENMP
  #pragma omp parallel for if(num_concats_ > 1)
#endif
    for (int n = 0; n < num_concats_; ++n) {
      caffe_copy(bottom_concat_axis * concat_input_size_,
          bottom_data + n * bottom_concat_axis * concat_input_size_,
          top_data + (n * top_concat_axis + offset_value)
              * concat_input_size_);
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int offset_value = offset_concat_axis;
    offset_concat_axis += bottom_concat_axis;
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
#ifdef _OPENMP
  #pragma omp parallel for if(num_concats_ > 1)
#endif
      for (int n = 0; n < num_concats_; ++n) {
        caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
            (n * top_concat_axis + offset_value) * concat_input_size_,
            bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
#endif

INSTANTIATE_CLASS(ConcatLayer);

}  // namespace caffe
