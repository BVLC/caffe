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
#include <vector>

#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/performance.hpp"

namespace caffe {

template <typename Dtype>
MKLSplitLayer<Dtype>::~MKLSplitLayer() {
  dnnDelete<Dtype>(sumPrimitive);
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Init(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_tops = top.size();
  size_t dim_src = bottom[0]->shape().size();
  this->sizes_src_.resize(dim_src);
  this->strides_src_.resize(dim_src);
  for (size_t d = 0; d < dim_src; ++d) {
    this->sizes_src_[d] = bottom[0]->shape()[dim_src - d - 1];
    this->strides_src_[d] = (d == 0) ?
                1 : this->strides_src_[d-1]*this->sizes_src_[d-1];
  }

  for (size_t i = 0; i < num_tops; ++i) {
    bwd_top_diff.push_back(shared_ptr<MKLDiff<Dtype> >(new MKLDiff<Dtype>));
    bwd_top_diff[i]->create_user_layout(dim_src,
                                        &(this->sizes_src_[0]),
                                        &(this->strides_src_[0]),
                                        false);
  }

  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(top.size(), 1);

  bwd_bottom_diff->create_user_layout(dim_src,
                                      &(this->sizes_src_[0]),
                                      &(this->strides_src_[0]),
                                      false);

  // Primitive will be created at first time it is to be used
  dnnDelete<Dtype>(sumPrimitive);
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Init(bottom, top);
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }

  // Here we check
  // Here I check for sizes whther to destroy primitives
  size_t dim_src = bottom[0]->shape().size();

  // If dimensions of blobs are the same as they were then
  // do not really destroy primitives
  if (dim_src == this->sizes_src_.size()) {
    // .. check for strides and size dims if they corresspond each other

    // TODO: speedup comparison?
    bool is_match = true;
    for (size_t d = 0; d < dim_src; ++d) {
        is_match = is_match && (this->sizes_src_[d] ==
                                bottom[0]->shape()[dim_src - 1 - d]);
        is_match = is_match && (this->strides_src_[d] == ((d == 0) ? 1 :
                                this->strides_src_[d-1]*this->sizes_src_[d-1]));
    }

    // If no new modification was done to layout sizes,
    // strides realtivly to previous iteration then
    // no primitives recreation is needed
    if (is_match) {
      return;
    }
  }

  Init(bottom, top);
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void MKLSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  dnnError_t e;
  vector<void*> top_diff;
  bool num_prv = 0;
  for (size_t i = 0; i < num_tops; i++) {
    top_diff.push_back(reinterpret_cast<void *>(
      const_cast<Dtype*>(top[i]->prv_diff())));
    if (top_diff[i] != NULL) {
      num_prv += 1;
    } else {
      top_diff[i] = reinterpret_cast<void*>(
      reinterpret_cast<void *>(const_cast<Dtype*>(top[i]->cpu_diff())));
    }
  }

  if (num_prv > 0) {
    if (sumPrimitive == NULL) {
      dnnLayout_t int_layout = NULL;
      for (size_t i = 0; i < num_tops; ++i) {
        if (top[i]->prv_diff() != NULL) {
          CHECK((top[i]->get_prv_diff_descriptor())->get_descr_type() ==
            PrvMemDescr::PRV_DESCR_MKL2017);
          shared_ptr<MKLDiff<Dtype> > mem_descr =
            boost::static_pointer_cast<MKLDiff<Dtype> >(
                top[i]->get_prv_diff_descriptor());
          CHECK(mem_descr != NULL);
          bwd_top_diff[i] = mem_descr;
          if (int_layout == NULL) {
            int_layout = mem_descr->layout_int;
          }
        }
      }
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL, num_tops,
        int_layout, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);

      bwd_bottom_diff->create_internal_layout(sumPrimitive, dnnResourceDst);

      for (size_t i = 0; i < num_tops; ++i) {
        if (top[i]->prv_diff() == NULL) {
          bwd_top_diff[i]->create_internal_layout(sumPrimitive,
                  (dnnResourceType_t)(dnnResourceMultipleSrc + i));
        }
      }
    }
  } else {
    if (sumPrimitive == NULL) {
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL, num_tops,
        bwd_bottom_diff->layout_usr, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);
    }
  }

  void *sum_res[dnnResourceNumber];
  for (int i = 0; i < num_tops; ++i) {
    if (bwd_top_diff[i]->convert_to_int) {
      sum_res[dnnResourceMultipleSrc + i] =
        bwd_top_diff[i]->get_converted_prv(top[i], false);
    } else {
      sum_res[dnnResourceMultipleSrc + i] =
        reinterpret_cast<void*>(top_diff[i]);
    }
  }

  if (bwd_bottom_diff->conversion_needed()) {
    bottom[0]->set_prv_diff_descriptor(bwd_bottom_diff);
    sum_res[dnnResourceDst] =
        reinterpret_cast<void*>(bottom[0]->mutable_prv_diff());
  } else {
    sum_res[dnnResourceDst] =
        reinterpret_cast<void*>(bottom[0]->mutable_cpu_diff());
  }

  PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKL_NAME("BW"));
  PERFORMANCE_MEASUREMENT_BEGIN();
  e = dnnExecute<Dtype>(sumPrimitive, sum_res);
  PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);

  CHECK_EQ(e, E_SUCCESS);
}

#ifdef CPU_ONLY
STUB_GPU(MKLSplitLayer);
#else
template <typename Dtype>
void MKLSplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLSplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLSplitLayer);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED)
