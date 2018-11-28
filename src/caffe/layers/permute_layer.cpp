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

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Permute(const int count, Dtype* bottom_data, const bool forward,
    const int* permute_order, const int* old_steps, const int* new_steps,
    const int num_axes, Dtype* top_data) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < count; ++i) {
      int old_idx = 0;
      int idx = i;
      for (int j = 0; j < num_axes; ++j) {
        int order = permute_order[j];
        old_idx += (idx / new_steps[j]) * old_steps[order];
        idx %= new_steps[j];
      }
      if (forward) {
        top_data[i] = bottom_data[old_idx];
      } else {
        bottom_data[old_idx] = top_data[i];
      }
    }
}

template <typename Dtype>
void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PermuteParameter permute_param = this->layer_param_.permute_param();
  CHECK_EQ(bottom.size(), 1);
  num_axes_ = bottom[0]->num_axes();
  vector<int> orders;
  // Push the specified new orders.
  for (int i = 0; i < permute_param.order_size(); ++i) {
    int order = permute_param.order(i);
    CHECK_LT(order, num_axes_)
        << "order should be less than the input dimension.";
    if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
      LOG(FATAL) << "there are duplicate orders";
    }
    orders.push_back(order);
  }
  // Push the rest orders. And save original step sizes for each axis.
  for (int i = 0; i < num_axes_; ++i) {
    if (std::find(orders.begin(), orders.end(), i) == orders.end()) {
      orders.push_back(i);
    }
  }
  CHECK_EQ(num_axes_, orders.size());
  // Check if we need to reorder the data or keep it.
  need_permute_ = false;
  for (int i = 0; i < num_axes_; ++i) {
    if (orders[i] != i) {
      // As long as there is one order which is different from the natural order
      // of the data, we need to permute. Otherwise, we share the data and diff.
      need_permute_ = true;
      break;
    }
  }

  vector<int> top_shape(num_axes_, 1);
  permute_order_.Reshape(num_axes_, 1, 1, 1);
  old_steps_.Reshape(num_axes_, 1, 1, 1);
  new_steps_.Reshape(num_axes_, 1, 1, 1);
  for (int i = 0; i < num_axes_; ++i) {
    permute_order_.mutable_cpu_data()[i] = orders[i];
    top_shape[i] = bottom[0]->shape(orders[i]);
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps_.mutable_cpu_data()[i] = 1;
    } else {
      old_steps_.mutable_cpu_data()[i] = bottom[0]->count(i + 1);
    }
    top_shape.push_back(bottom[0]->shape(permute_order_.cpu_data()[i]));
  }
  top[0]->Reshape(top_shape);

  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      new_steps_.mutable_cpu_data()[i] = 1;
    } else {
      new_steps_.mutable_cpu_data()[i] = top[0]->count(i + 1);
    }
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (need_permute_) {
    Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int top_count = top[0]->count();
    const int* permute_order = permute_order_.cpu_data();
    const int* old_steps = old_steps_.cpu_data();
    const int* new_steps = new_steps_.cpu_data();
    bool forward = true;
    Permute(top_count, bottom_data, forward, permute_order, old_steps,
            new_steps, num_axes_, top_data);
  } else {
    // If there is no need to permute, we share data to save memory.
    top[0]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (need_permute_) {
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int top_count = top[0]->count();
    const int* permute_order = permute_order_.cpu_data();
    const int* old_steps = old_steps_.cpu_data();
    const int* new_steps = new_steps_.cpu_data();
    bool forward = false;
    Permute(top_count, bottom_diff, forward, permute_order, old_steps,
            new_steps, num_axes_, top_diff);
  } else {
    // If there is no need to permute, we share diff to save memory.
    bottom[0]->ShareDiff(*top[0]);
  }
}

#ifdef CPU_ONLY
STUB_GPU(PermuteLayer);
#endif

INSTANTIATE_CLASS(PermuteLayer);
REGISTER_LAYER_CLASS(Permute);

}  // namespace caffe
