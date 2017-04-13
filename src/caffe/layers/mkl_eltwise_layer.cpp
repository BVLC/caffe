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

#if defined(MKL2017_SUPPORTED)
#include <cfloat>
#include <vector>

#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/performance.hpp"

namespace caffe {

template <typename Dtype>
MKLEltwiseLayer<Dtype>::~MKLEltwiseLayer() {
  dnnDelete<Dtype>(sumPrimitive);
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Init(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();

  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();

  num_bottoms = bottom.size();
  size_t dim_src = bottom[0]->shape().size();
  size_t sizes_src[dim_src], strides_src[dim_src];
  for (size_t d = 0; d < dim_src; ++d) {
      sizes_src[d] = bottom[0]->shape()[dim_src - d - 1];
      strides_src[d] = (d == 0) ? 1 : strides_src[d-1]*sizes_src[d-1];
  }

  for (size_t i = 0; i < num_bottoms; ++i) {
      fwd_bottom_data.push_back(
        shared_ptr<MKLData<Dtype> >(new MKLData<Dtype>));
      bwd_bottom_diff.push_back(
        shared_ptr<MKLDiff<Dtype> >(new MKLDiff<Dtype>));
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      fwd_bottom_data[i]->create_user_layout(dim_src,
                                             sizes_src,
                                             strides_src,
                                             false);
      bwd_bottom_diff[i]->create_user_layout(dim_src,
                                             sizes_src,
                                             strides_src,
                                             false);
  }

  fwd_top_data->create_user_layout(dim_src, sizes_src, strides_src, false);

  dnnDelete<Dtype>(sumPrimitive);
}


template <typename Dtype>
void MKLEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "MKLEltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "MKLEltwise layer only takes coefficients for summation.";

  CHECK(this->layer_param().eltwise_param().operation() ==
    EltwiseParameter_EltwiseOp_SUM)
      << "MKLEltwise Layer only process summation.";

  Init(bottom, top);
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }

  if (channels_ == bottom[0]->channels() &&
      height_ == bottom[0]->height() &&
      width_ == bottom[0]->width() &&
      num_ == bottom[0]->num() &&
      num_bottoms == bottom.size()) {
    return;
  }

  Init(bottom, top);
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  dnnError_t e;
  vector<void*> bottom_data;
  bool num_prv = 0;
  for (size_t i = 0; i < num_bottoms; i++) {
    bottom_data.push_back(
      reinterpret_cast<void *>(const_cast<Dtype*>(bottom[i]->prv_data())));
    if (bottom_data[i] != NULL) {
      num_prv += 1;
    } else {
      bottom_data[i] =
        reinterpret_cast<void *>(const_cast<Dtype*>(bottom[i]->cpu_data()));
    }
  }

  if (num_prv > 0) {
    if (sumPrimitive == NULL) {
      dnnLayout_t int_layout = NULL;
      for (size_t i = 0; i < num_bottoms; ++i) {
        if (bottom[i]->prv_data() != NULL) {
          CHECK((bottom[i]->get_prv_data_descriptor())->get_descr_type()
            == PrvMemDescr::PRV_DESCR_MKL2017);
          shared_ptr<MKLData<Dtype> > mem_descr =
              boost::static_pointer_cast<MKLData<Dtype> >(
                bottom[i]->get_prv_data_descriptor());
          CHECK(mem_descr != NULL);
          fwd_bottom_data[i] = mem_descr;
          if (int_layout == NULL) {
            int_layout = mem_descr->layout_int;
          }
        }
      }
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL,
        num_bottoms, int_layout, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);

      fwd_top_data->create_internal_layout(sumPrimitive, dnnResourceDst);

      for (int i = 0; i < num_bottoms; ++i) {
        if (bottom[i]->prv_data() == NULL) {
          fwd_bottom_data[i]->create_internal_layout(sumPrimitive,
              (dnnResourceType_t)(dnnResourceMultipleSrc + i));
        }
      }
    }
  } else {
    if (sumPrimitive == NULL) {
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL, num_bottoms,
        fwd_top_data->layout_usr, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);
    }
  }

  switch (op_) {
  case EltwiseParameter_EltwiseOp_SUM:
    void *eltwise_res[dnnResourceNumber];
    for (int i = 0; i < num_bottoms; ++i) {
      if (fwd_bottom_data[i]->convert_to_int) {
        eltwise_res[dnnResourceMultipleSrc + i] =
          fwd_bottom_data[i]->get_converted_prv(bottom[i], false);
      } else {
        eltwise_res[dnnResourceMultipleSrc + i] =
          reinterpret_cast<void *>(bottom_data[i]);
      }
    }

    if (fwd_top_data->conversion_needed()) {
      top[0]->set_prv_data_descriptor(fwd_top_data);
      eltwise_res[dnnResourceDst] =
        reinterpret_cast<void*>(const_cast<Dtype*>(top[0]->mutable_prv_data()));
    } else {
      eltwise_res[dnnResourceDst] =
        reinterpret_cast<void*>(const_cast<Dtype*>(top[0]->mutable_cpu_data()));
    }

    { // local scope needed since the macro below contains variable declaration
      PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKL_NAME("FW"));
      PERFORMANCE_MEASUREMENT_BEGIN();
      e = dnnExecute<Dtype>(sumPrimitive, eltwise_res);
      PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);
    }
    CHECK_EQ(e, E_SUCCESS);

    break;
  case EltwiseParameter_EltwiseOp_PROD:
  case EltwiseParameter_EltwiseOp_MAX:
    LOG(FATAL) << "Unsupported elementwise operation.";
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  bool is_top_diff_prv = top[0]->prv_diff() == NULL ? false : true;

  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      switch (op_) {
      case EltwiseParameter_EltwiseOp_SUM:
        CHECK_EQ(coeffs_[i], Dtype(1)) << "Not supported yet";
        if (is_top_diff_prv == false) {
          bottom[i]->set_cpu_diff(top[0]->mutable_cpu_diff());
        } else {
          if (!bwd_bottom_diff[i]->layout_int) {
            bwd_bottom_diff[i]->create_internal_layout(sumPrimitive,
              (dnnResourceType_t)(dnnResourceMultipleSrc + i));
          }
          CHECK_EQ(true, bwd_bottom_diff[i]->layout_compare(
                  top[0]->get_prv_diff_descriptor()));
          bottom[i]->set_prv_diff_descriptor(top[0]->get_prv_diff_descriptor(),
                                             false);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
      case EltwiseParameter_EltwiseOp_PROD:
        LOG(FATAL) << "Unsupported elementwise operation.";
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKLEltwiseLayer);
#else
template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLEltwiseLayer);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED)
