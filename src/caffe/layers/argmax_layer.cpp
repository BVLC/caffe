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
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/argmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ArgMaxParameter& argmax_param = this->layer_param_.argmax_param();
  out_max_val_ = argmax_param.out_max_val();
  top_k_ = argmax_param.top_k();
  has_axis_ = argmax_param.has_axis();
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  if (has_axis_) {
    axis_ = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
    CHECK_LE(top_k_, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    CHECK_LE(top_k_, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  if ( num_top_axes < 3 ) num_top_axes = 3;
  std::vector<int> shape(num_top_axes, 1);
  if (has_axis_) {
    // Produces max_ind or max_val per axis
    shape = bottom[0]->shape();
    shape[axis_] = top_k_;
  } else {
    shape[0] = bottom[0]->shape(0);
    // Produces max_ind
    shape[2] = top_k_;
    if (out_max_val_) {
      // Produces max_ind and max_val
      shape[1] = 2;
    }
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int dim, axis_dist;
  if (has_axis_) {
    dim = bottom[0]->shape(axis_);
    // Distance between values of axis in blob
    axis_dist = bottom[0]->count(axis_) / dim;
  } else {
    dim = bottom[0]->count(1);
    axis_dist = 1;
  }
  int num = bottom[0]->count() / dim;
  std::vector<std::pair<Dtype, int> > bottom_data_vector(dim);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
        bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    for (int j = 0; j < top_k_; ++j) {
      if (out_max_val_) {
        if (has_axis_) {
          // Produces max_val per axis
          top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
            = bottom_data_vector[j].first;
        } else {
          // Produces max_ind and max_val
          top_data[2 * i * top_k_ + j] = bottom_data_vector[j].second;
          top_data[2 * i * top_k_ + top_k_ + j] = bottom_data_vector[j].first;
        }
      } else {
        // Produces max_ind per axis
        top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
          = bottom_data_vector[j].second;
      }
    }
  }
}

INSTANTIATE_CLASS(ArgMaxLayer);
REGISTER_LAYER_CLASS(ArgMax);

}  // namespace caffe
