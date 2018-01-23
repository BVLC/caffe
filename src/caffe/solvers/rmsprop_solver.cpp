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

#ifndef CPU_ONLY
template <typename Dtype>
void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay,
    Dtype delta, Dtype local_rate);
#endif

template <typename Dtype>
void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype delta = this->param_.delta();
  Dtype rms_decay = this->param_.rms_decay();
  Dtype local_rate = rate * net_params_lr[param_id];
  Dtype momentum = this->param_.momentum();
  const Dtype* diff_ptr = NULL;
  Dtype* mutable_diff_ptr = NULL;

  switch (Caffe::mode()) {
  case Caffe::CPU:
    if (net_params[param_id]->prv_diff()
        && (net_params[param_id]->prv_diff_count()
            == net_params[param_id]->count())) {
      diff_ptr = net_params[param_id]->prv_diff();
      mutable_diff_ptr = net_params[param_id]->mutable_prv_diff();
    } else {
      diff_ptr = net_params[param_id]->cpu_diff();
      mutable_diff_ptr = net_params[param_id]->mutable_cpu_diff();
    }

    // compute square of gradient in update
    caffe_powx(net_params[param_id]->count(),
      diff_ptr, Dtype(2),
      this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id] -> count(),
      Dtype(1-rms_decay), this->update_[param_id]->cpu_data(),
      rms_decay, this->history_[param_id]-> mutable_cpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
      this->history_[param_id]->cpu_data(),
      this->update_[param_id]->mutable_cpu_data());

    // add delta
    caffe_add_scalar(net_params[param_id]->count(),
      delta, this->update_[param_id]->mutable_cpu_data());

    // prepare update
    caffe_powx(net_params[param_id]->count(),
      this->update_[param_id]->cpu_data(), Dtype(0.5),
      this->update_[param_id]->mutable_cpu_data());

    caffe_div(net_params[param_id]->count(),
      diff_ptr, this->update_[param_id]->cpu_data(),
      this->update_[param_id]->mutable_cpu_data());

    // scale and copy
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
      this->update_[param_id]->cpu_data(), momentum,
      this->temp_[param_id]->mutable_cpu_data());

    caffe_copy(net_params[param_id]->count(),
      this->temp_[param_id]->cpu_data(),
      mutable_diff_ptr);

    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    rmsprop_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        this->history_[param_id]->mutable_gpu_data(),
        rms_decay, delta, local_rate);
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(RMSPropSolver);
REGISTER_SOLVER_CLASS(RMSProp);

}  // namespace caffe
