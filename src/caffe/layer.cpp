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

#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

#ifdef USE_MLSL
template <typename Dtype>
mn::Distribution & Layer<Dtype>::GetDistribution() {
  const MultinodeLayerParameter &mn_layer_param = layer_param_.multinode();
  int num_nodes = mn_layer_param.num_nodes();
  int model_parts = mn_layer_param.model_parts();
  mn::GetCanonicalMnParam(num_nodes, model_parts);
  return *mn::get_distrib(num_nodes/model_parts, model_parts);
}

template <typename Dtype>
bool Layer<Dtype>::Bypass(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  return GetDistribution().get_group_id() > 0;
}

template <typename Dtype>
void Layer<Dtype>::MultinodeSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->layerOp != NULL || this->phase_ != TRAIN || Bypass(bottom, top)) {
    return;
  }

  int num_nodes = layer_param_.multinode().num_nodes();
  int model_parts = layer_param_.multinode().model_parts();
  mn::GetCanonicalMnParam(num_nodes, model_parts);
  int data_parts = num_nodes / model_parts;

  if (data_parts <= 1 || this->blobs_.size() == 0) return;

  // We only initialize data parallelism here so operation type is
  // irrelevant here, hard-code to OT_CC
  mn::OpRegInfo reg_info(mn::train::get_session(), MLSL::OT_CC);
  reg_info.set_name(this->layer_param().name());
  for (int i = 0; i < this->blobs_.size(); i++) {
    int hw = 1, ic = 1, oc = 1;
    const vector<int> &shape = this->blobs_[i]->shape();
    CHECK_GT(shape.size(), 0);
    oc = shape[0];
    if (shape.size() > 1) ic = shape[1];
    if (shape.size() >= 4) hw = shape[2] * shape[3];
    // Note that MLSL expects the entire weights from a model group.
    // So we should multiply by model_parts here.
    reg_info.add_parameter_set<Dtype>(ic * oc * model_parts, hw);
  }
  this->layerOp = mn::train::add_operation(reg_info, this->GetDistribution());
}
#endif

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
