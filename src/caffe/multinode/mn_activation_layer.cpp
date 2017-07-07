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

#ifdef USE_MLSL

#include "caffe/multinode/mn_activation_layer.hpp"
#include "caffe/multinode/mlsl.hpp"

namespace caffe {

template <typename Dtype>
void MnActivationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MnActivationParameter param = this->layer_param_.mn_activation_param();
  num_nodes_in_ = param.num_nodes_in();
  num_nodes_out_ = param.num_nodes_out();
  model_parts_in_ = param.model_parts_in();
  model_parts_out_ = param.model_parts_out();
  mn::GetCanonicalMnParam(num_nodes_in_, model_parts_in_);
  mn::GetCanonicalMnParam(num_nodes_out_, model_parts_out_);
  data_parts_in_ = num_nodes_in_ / model_parts_in_;
  data_parts_out_ = num_nodes_out_ / model_parts_out_;
  
  CHECK_EQ(num_nodes_in_, data_parts_in_ * model_parts_in_);
  CHECK_EQ(num_nodes_out_, data_parts_out_ * model_parts_out_);
  CHECK(data_parts_in_  != data_parts_out_  ||
        model_parts_in_ != model_parts_out_ ||
        model_parts_in_ > 1);
  
  distrib_in_ = mn::get_distrib(data_parts_in_, model_parts_in_);
  distrib_out_ = mn::get_distrib(data_parts_out_, model_parts_out_);

  if (data_parts_in_ != data_parts_out_) {
    int num_nodes = mn::get_nodes_count();
    int node_id = mn::get_node_id();
    int data_parts_max = std::max(data_parts_in_, data_parts_out_);
    int data_parts_min = std::min(data_parts_in_, data_parts_out_);
    int num_data_groups = num_nodes / data_parts_min;
    // make sure data_color in-use starts from 0 and ends at data_parts_min-1
    int data_color = node_id / num_data_groups +
      (node_id % (num_nodes / data_parts_max)) * data_parts_min;
    LOG(INFO) << "Create data_in_out distribution: "
              << data_parts_in_ << " ==> " << data_parts_out_
              << ", (" << data_parts_max / data_parts_min
              << ",1), data color: " << data_color
              << ", data color max: " << data_parts_min-1;
    distrib_data_in_out_ = mn::create_distrib(
      data_parts_max / data_parts_min, 1, data_color, MLSL_DEFAULT_COLOR,
      data_parts_min-1, MLSL_DEFAULT_COLOR);
  }
}

template <typename Dtype>
void MnActivationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const vector<int> &bottom_shape = bottom[0]->shape();
  vector<int> top_shape = bottom[0]->shape();
  // re-group and distribute the data parts
  top_shape[0] = bottom_shape[0] * data_parts_in_ / data_parts_out_;
  if (top_shape.size() > 1) {
    // gather all the model parts split from previous output
    top_shape[1] = bottom_shape[1] * model_parts_in_;
  }
  top[0]->Reshape(top_shape);
  top_reduce_buf_.ReshapeLike(*top[0]);
  vector<int> bottom_gather_shape = bottom[0]->shape();
  if (bottom_shape.size() > 1) {
    bottom_gather_shape[1] = bottom_shape[1] * model_parts_in_;
  }
  bottom_gather_buf_.Reshape(bottom_gather_shape);
  bottom_gather_work_buf_.Reshape(bottom_gather_shape);
}

template <typename Dtype>
void MnActivationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype *bottom_work_buf = (Dtype*)bottom[0]->cpu_data();
  if (model_parts_in_ > 1) {
    distrib_in_->gather<Dtype,MLSL::GT_MODEL>(
      (Dtype*)bottom[0]->cpu_data(), bottom[0]->count(),
      bottom_gather_buf_.mutable_cpu_data());
    if (data_parts_in_ == data_parts_out_) {
      bottom_work_buf = top[0]->mutable_cpu_data();
    } else {
      bottom_work_buf = bottom_gather_work_buf_.mutable_cpu_data();
    }
    Unpack(
      bottom_gather_buf_.cpu_data(),
      bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->count(2),
      model_parts_in_,
      bottom_work_buf);
  }
  if (data_parts_in_ > data_parts_out_) {
    distrib_data_in_out_->gather<Dtype,MLSL::GT_DATA>(
      bottom_work_buf, bottom[0]->count() * model_parts_in_,
      top[0]->mutable_cpu_data());
  } else if (data_parts_in_ < data_parts_out_) {
    distrib_data_in_out_->scatter<Dtype,MLSL::GT_DATA>(
      bottom_work_buf, top[0]->mutable_cpu_data(),
      top[0]->count());
  } else {
    if (bottom_work_buf != top[0]->mutable_cpu_data()) {
      caffe_copy(
        top[0]->count(), bottom_work_buf, top[0]->mutable_cpu_data());
    }
  }
  distrib_out_->bcast<Dtype,MLSL::GT_MODEL>(
    top[0]->mutable_cpu_data(), top[0]->count());
}

template <typename Dtype>
bool MnActivationLayer<Dtype>::Backward_cpu_fast(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) {
  if (num_nodes_in_ == num_nodes_out_ &&
      model_parts_in_ == model_parts_out_ &&
      model_parts_in_ > 1) {
    Pack(top[0]->cpu_diff(), bottom_gather_work_buf_.mutable_cpu_data(),
         bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->count(2),
         model_parts_in_);
    distrib_out_->reducescatter<Dtype,MLSL::RT_SUM,MLSL::GT_MODEL>(
      bottom_gather_work_buf_.mutable_cpu_data(),
      bottom[0]->mutable_cpu_diff(), bottom[0]->count());
    return true;
  }
  return false;
}

template <typename Dtype>
void MnActivationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    if (Backward_cpu_fast(top, bottom)) return;
    Dtype *top_work_buf = (Dtype*)top[0]->cpu_diff();
    if (model_parts_out_ > 1 &&
        this->layer_param_.mn_activation_param().need_reduce()) {
      distrib_out_->reduce<Dtype,MLSL::RT_SUM,MLSL::GT_MODEL>(
        (Dtype*)top[0]->cpu_diff(), top_reduce_buf_.mutable_cpu_data(),
        top_reduce_buf_.count());
      top_work_buf = top_reduce_buf_.mutable_cpu_data();
    }
    Dtype *bottom_work_buf = bottom[0]->mutable_cpu_diff();
    if (model_parts_in_ > 1) {
      bottom_work_buf = bottom_gather_buf_.mutable_cpu_data();
    }
    if (data_parts_in_ > data_parts_out_) {
      distrib_data_in_out_->scatter<Dtype,MLSL::GT_DATA>(
        top_work_buf, bottom_work_buf,
        bottom_gather_buf_.count());
    } else if (data_parts_in_ < data_parts_out_) {
      distrib_data_in_out_->gather<Dtype,MLSL::GT_DATA>(
        top_work_buf, top[0]->count(),
        bottom_work_buf);
    } else {
      if (model_parts_in_ > 1) {
        bottom_work_buf = top_work_buf;
      } else {
        caffe_copy(
          bottom[0]->count(), top_work_buf, bottom_work_buf);
      }
    }
    if (model_parts_in_ > 1) {
      Pack(bottom_work_buf, bottom_gather_work_buf_.mutable_cpu_data(),
           bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->count(2),
           model_parts_in_);
      distrib_in_->scatter<Dtype,MLSL::GT_MODEL>(
        bottom_gather_work_buf_.mutable_cpu_data(),
        bottom[0]->mutable_cpu_diff(), bottom[0]->count());
    }
  }
}

template <typename Dtype>
void MnActivationLayer<Dtype>::Unpack(const Dtype *src, int N, int C, int HW, int numC, Dtype *dst) {
  int dstC = numC * C;
#pragma omp parallel for collapse (2)
  for (int iN = 0; iN < N; iN++) {
    for (int iC = 0; iC < dstC; iC++) {
      int iSrc =  iC / C;
      int iSrcC = iC % C;
      for (int iHW = 0; iHW < HW; iHW++) {
        dst[iN*dstC*HW + iC*HW + iHW] =
          src[iSrc*N*C*HW + iN*C*HW + iSrcC*HW + iHW];
      }
    }
  }
}

template <typename Dtype>
void MnActivationLayer<Dtype>::Pack(const Dtype *src, Dtype *dst, int N, int C, int HW, int numC) {
  int srcC = numC * C;
  for (int iDst = 0; iDst < numC; iDst++) {
#pragma omp parallel for collapse (2)
    for (int iN = 0; iN < N; iN++) {
      for (int iC = 0; iC < C; iC++) {
        int iSrcC = iDst * C + iC;
        for (int iHW = 0; iHW < HW; iHW++) {
          dst[iDst*N*C*HW + iN*C*HW + iC*HW + iHW] =
            src[iN*srcC*HW + iSrcC*HW + iHW];
        }
      }
    }
  }
}

template <typename Dtype>
bool MnActivationLayer<Dtype>::Bypass(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  return distrib_in_->get_group_id() > 0 && distrib_out_->get_group_id() > 0;
}

#ifdef CPU_ONLY
STUB_GPU(MnActivationLayer);
#endif

INSTANTIATE_CLASS(MnActivationLayer);
REGISTER_LAYER_CLASS(MnActivation);
}  // namespace caffe

#endif

