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
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void PReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PReLUParameter prelu_param = this->layer_param().prelu_param();
  int channels = bottom[0]->shape(1);
  channel_shared_ = prelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (prelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(prelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void PReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const long count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->shape(1);
  const Dtype* slope_data = this->blobs_[0]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (long i = 0; i < count; ++i) {
    long c = (i / dim) % channels / div_factor;
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + slope_data[c] * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const long count = bottom[0]->count();
  const long dim = bottom[0]->count(2);
  const int channels = bottom[0]->shape(1);

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    for (long i = 0; i < count; ++i) {
      long c = (i / dim) % channels / div_factor;
      slope_diff[c] += top_diff[i] * bottom_data[i] * (bottom_data[i] <= 0);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (long i = 0; i < count; ++i) {
      long c = (i / dim) % channels / div_factor;
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PReLULayer);
#endif

INSTANTIATE_CLASS(PReLULayer);
REGISTER_LAYER_CLASS(PReLU);

}  // namespace caffe
