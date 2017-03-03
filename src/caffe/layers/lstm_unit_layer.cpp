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
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num_instances = bottom[0]->shape(1);
  for (int i = 0; i < bottom.size(); ++i) {
    if (i == 2) {
      CHECK_EQ(2, bottom[i]->num_axes());
    } else {
      CHECK_EQ(3, bottom[i]->num_axes());
    }
    CHECK_EQ(1, bottom[i]->shape(0));
    CHECK_EQ(num_instances, bottom[i]->shape(1));
  }
  hidden_dim_ = bottom[0]->shape(2);
  CHECK_EQ(num_instances, bottom[1]->shape(1));
  CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
  X_acts_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* X = bottom[1]->cpu_data();
  const Dtype* cont = bottom[2]->cpu_data();
  Dtype* C = top[0]->mutable_cpu_data();
  Dtype* H = top[1]->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Dtype i = sigmoid(X[d]);
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      const Dtype c_prev = C_prev[d];
      const Dtype c = f * c_prev + i * g;
      C[d] = c;
      const Dtype tanh_c = tanh(c);
      H[d] = o * tanh_c;
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    ++cont;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }

  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* X = bottom[1]->cpu_data();
  const Dtype* cont = bottom[2]->cpu_data();
  const Dtype* C = top[0]->cpu_data();
  const Dtype* H = top[1]->cpu_data();
  const Dtype* C_diff = top[0]->cpu_diff();
  const Dtype* H_diff = top[1]->cpu_diff();
  Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
  Dtype* X_diff = bottom[1]->mutable_cpu_diff();
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      const Dtype i = sigmoid(X[d]);
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      const Dtype c_prev = C_prev[d];
      const Dtype c = C[d];
      const Dtype tanh_c = tanh(c);
      Dtype* c_prev_diff = C_prev_diff + d;
      Dtype* i_diff = X_diff + d;
      Dtype* f_diff = X_diff + 1 * hidden_dim_ + d;
      Dtype* o_diff = X_diff + 2 * hidden_dim_ + d;
      Dtype* g_diff = X_diff + 3 * hidden_dim_ + d;
      const Dtype c_term_diff =
          C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
      *c_prev_diff = c_term_diff * f;
      *i_diff = c_term_diff * g * i * (1 - i);
      *f_diff = c_term_diff * c_prev * f * (1 - f);
      *o_diff = H_diff[d] * tanh_c * o * (1 - o);
      *g_diff = c_term_diff * i * (1 - g * g);
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    C_diff += hidden_dim_;
    H_diff += hidden_dim_;
    X_diff += x_dim;
    C_prev_diff += hidden_dim_;
    ++cont;
  }
}

#ifdef CPU_ONLY
STUB_GPU(LSTMUnitLayer);
#endif

INSTANTIATE_CLASS(LSTMUnitLayer);
REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe
