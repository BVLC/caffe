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

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void AdamSolver<Dtype>::AdamPreSolve() {
  // Add the extra history entries for Adam after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate);
#endif

template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();

  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();

  const int t = this->iter_ + 1;
  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
      (Dtype(1.) - pow(beta1, t));
  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
    val_t->mutable_cpu_data());
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    // set update
    caffe_powx(N,
        val_v->cpu_data(), Dtype(0.5),
        val_t->mutable_cpu_data());
    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
    caffe_div(N,
        val_m->cpu_data(),
        val_t->cpu_data(),
        val_t->mutable_cpu_data());

    caffe_cpu_scale(N, local_rate*correction,
        val_t->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    adam_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta2,
        eps_hat, local_rate*correction);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(AdamSolver);
REGISTER_SOLVER_CLASS(Adam);

}  // namespace caffe
