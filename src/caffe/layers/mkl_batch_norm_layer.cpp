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

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/performance.hpp"

namespace caffe {

template <typename Dtype>
MKLBatchNormLayer<Dtype>::~MKLBatchNormLayer() {
  dnnDelete<Dtype>(batchNormFwd);
  dnnDelete<Dtype>(batchNormBwdData);
  dnnDelete<Dtype>(batchNormBwdScaleShift);

  dnnLayoutDelete<Dtype>(layout_usr_);
  dnnReleaseBuffer<Dtype>(workspace_buffer_);
  dnnReleaseBuffer<Dtype>(scaleShift_buffer_);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Init(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  eps_ = this->layer_param_.batch_norm_param().eps();
  use_weight_bias_ = this->layer_param_.batch_norm_param().use_weight_bias();
  bias_term_ = this->layer_param_.batch_norm_param().bias_term();

  // Workaround. Checking count of parameters in order to handle
  // topology for reference BatchNorm layer which don't have scaling
  if (this->layer_param_.param_size() == 3) {
    this->blobs_.resize(3);
    use_weight_bias_ = false;
  }

  size_t dim = 4, sizes[4], strides[4];

  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  num_      = bottom[0]->num();

  sizes[0] = width_;
  sizes[1] = height_;
  sizes[2] = channels_;
  sizes[3] = num_;

  strides[0] = 1;
  strides[1] = sizes[0];
  strides[2] = sizes[0]*sizes[1];
  strides[3] = sizes[0]*sizes[1]*sizes[2];

  // Names are for debugging only
  fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data->name =    "fwd_top_data      @ " + this->layer_param_.name();
  bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->layer_param_.name();
  bwd_top_diff->name =    "bwd_top_diff      @ " + this->layer_param_.name();

  // TODO: Make a cleanup routine to avoid
  // copy of following code in the Destructor

  dnnError_t e;
  dnnLayoutDelete<Dtype>(layout_usr_);
  e = dnnLayoutCreate<Dtype>(&layout_usr_, dim, sizes, strides);
  CHECK_EQ(e, E_SUCCESS);

  fwd_bottom_data->create_user_layout(dim, sizes, strides, false);
  fwd_top_data   ->create_user_layout(dim, sizes, strides, false);
  bwd_bottom_diff->create_user_layout(dim, sizes, strides, false);
  bwd_top_diff   ->create_user_layout(dim, sizes, strides, false);

  dnnReleaseBuffer<Dtype>(workspace_buffer_);
  dnnReleaseBuffer<Dtype>(scaleShift_buffer_);
  // "Lazy" allocation because here we don't know
  // what layout is used by neighbours.

  // Primitives will be allocated during the first fwd pass
  dnnDelete<Dtype>(batchNormFwd);
  dnnDelete<Dtype>(batchNormBwdData);
  dnnDelete<Dtype>(batchNormBwdScaleShift);

  if (use_weight_bias_) {
    if ( bias_term_ ) {
        this->blobs_.resize(2);
    } else {
        this->blobs_.resize(1);
    }
    // Initialize scale and shift
    vector<int> scaleshift_shape(1);
    scaleshift_shape[0] = channels_;

    this->blobs_[0].reset(new Blob<Dtype>(scaleshift_shape));
    FillerParameter filler_param(
      this->layer_param_.batch_norm_param().filler());
    if (!this->layer_param_.batch_norm_param().has_filler()) {
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());

    if ( bias_term_ ) {
      this->blobs_[1].reset(new Blob<Dtype>(scaleshift_shape));
      FillerParameter bias_filler_param(
        this->layer_param_.batch_norm_param().bias_filler());
      if (!this->layer_param_.batch_norm_param().has_bias_filler()) {
        bias_filler_param.set_type("constant");
        bias_filler_param.set_value(0);
      }
      shared_ptr<Filler<Dtype> > bias_filler(
        GetFiller<Dtype>(bias_filler_param));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Init(bottom, top);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bool reshaping = true;
  if ((num_ == bottom[0]->num()) &&
      channels_ == bottom[0]->channels() &&
      height_ == bottom[0]->height() &&
      width_ == bottom[0]->width()) {
    reshaping = false;
  }

  if (bottom[0] == top[0]) {  // in-place computation
    temp_.ReshapeLike(*bottom[0]);
  } else {
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_ = bottom[0]->num();
    top[0]->Reshape(num_, channels_, height_, width_);
  }

  if (reshaping == true) {
    Init(bottom, top);
  }
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  void* bottom_data =
    reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->prv_data()));
  int is_first_pass = 0;

  if (NULL != bottom_data) {
    // Is it the first pass? Create a primitive.
    if (batchNormFwd == NULL) {
      is_first_pass = 1;

      CHECK((bottom[0]->get_prv_data_descriptor())->get_descr_type() ==
        PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKLData<Dtype> >(
           bottom[0]->get_prv_data_descriptor());
      CHECK(mem_descr != NULL);

      DLOG(INFO) << "Using layout of " << mem_descr->name
              << " as input layout for " << this->layer_param_.name();

      fwd_bottom_data = mem_descr;

      dnnError_t e;
      e = dnnBatchNormalizationCreateForward<Dtype>(
        &batchNormFwd, NULL, mem_descr->layout_int, eps_);
      CHECK_EQ(e, E_SUCCESS);

      fwd_top_data   ->create_internal_layout(batchNormFwd, dnnResourceDst);
      bwd_top_diff   ->create_internal_layout(batchNormFwd, dnnResourceDst);
      bwd_bottom_diff->create_internal_layout(batchNormFwd, dnnResourceSrc);


      e = dnnBatchNormalizationCreateBackwardData<Dtype>(
        &batchNormBwdData, NULL, mem_descr->layout_int, eps_);
      CHECK_EQ(e, E_SUCCESS);

      if (use_weight_bias_) {
        e = dnnBatchNormalizationCreateBackwardScaleShift<Dtype>(
          &batchNormBwdScaleShift, NULL, mem_descr->layout_int, eps_);
        CHECK_EQ(e, E_SUCCESS);
      }
    }
  } else {
    DLOG(INFO) << "Using cpu_data in MKLBatchNormLayer.";
    if (batchNormFwd == NULL) {
      // First pass
      is_first_pass = 1;

      dnnError_t e;
      e = dnnBatchNormalizationCreateForward<Dtype>(
        &batchNormFwd, NULL, layout_usr_, eps_);
      CHECK_EQ(e, E_SUCCESS);

      e = dnnBatchNormalizationCreateBackwardData<Dtype>(
        &batchNormBwdData, NULL, layout_usr_, eps_);
      CHECK_EQ(e, E_SUCCESS);

      if (use_weight_bias_) {
        e = dnnBatchNormalizationCreateBackwardScaleShift<Dtype>(
          &batchNormBwdScaleShift, NULL, layout_usr_, eps_);
        CHECK_EQ(e, E_SUCCESS);
      }
    }
    bottom_data =
      reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->cpu_data()));
  }
  if (is_first_pass == 1) {
      dnnError_t e;

      dnnLayout_t workspace_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<Dtype>(
        &workspace_buffer_l, batchNormFwd, dnnResourceWorkspace);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>(
        reinterpret_cast<void**>(&workspace_buffer_), workspace_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(workspace_buffer_l);

      dnnLayout_t scaleShift_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<Dtype>(
        &scaleShift_buffer_l, batchNormFwd, dnnResourceScaleShift);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>(
        reinterpret_cast<void**>(&scaleShift_buffer_), scaleShift_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(scaleShift_buffer_l);
      if (!use_weight_bias_) {
         for (int i = 0; i < channels_; i++) {
            scaleShift_buffer_[i] = 1.0;
            scaleShift_buffer_[channels_ + i] = 0;
         }
      }
  }

  if (use_weight_bias_) {
    // Fill ScaleShift buffer
    for (int i = 0; i < channels_; i++) {
      scaleShift_buffer_[i] = this->blobs_[0]->cpu_data()[i];
      scaleShift_buffer_[channels_ + i] = 0;
      if (bias_term_) {
         scaleShift_buffer_[channels_ + i] = this->blobs_[1]->cpu_data()[i];
      }
    }
  }

  if (bottom[0] == top[0] && this->phase_ == TRAIN) {
    // In-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we skip this if not
    // doing Backward
    caffe_copy(bottom[0]->count(), static_cast<Dtype*>(bottom_data),
                                                      temp_.mutable_cpu_data());
  }

  dnnError_t e;
  void* BatchNorm_res[dnnResourceNumber];
  BatchNorm_res[dnnResourceSrc] = bottom_data;
  BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
  BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;
  if (fwd_top_data->conversion_needed()) {
    top[0]->set_prv_data_descriptor(fwd_top_data);
    BatchNorm_res[dnnResourceDst] =
            reinterpret_cast<void *>(top[0]->mutable_prv_data());
  } else {
    BatchNorm_res[dnnResourceDst] =
            reinterpret_cast<void *>(top[0]->mutable_cpu_data());
    DLOG(INFO) << "Using cpu_data for top in DnnBatchNorm.";
  }

  PERFORMANCE_MEASUREMENT_BEGIN();
  e = dnnExecute<Dtype>(batchNormFwd, BatchNorm_res);
  PERFORMANCE_MEASUREMENT_END_STATIC("BW_mkl_batch_norm");

  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  void *bottom_data = NULL;
  if (bottom[0] == top[0]) {
    bottom_data = reinterpret_cast<void *>(
                        const_cast<Dtype*>(temp_.cpu_data()));
  } else {
    bottom_data =
            reinterpret_cast<void *>(
                        const_cast<Dtype*>(bottom[0]->prv_data()));
    if (NULL == bottom_data)
      bottom_data =
            reinterpret_cast<void *>(
                        const_cast<Dtype*>(bottom[0]->cpu_data()));
  }

  dnnError_t e;
  void* BatchNorm_res[dnnResourceNumber];
  BatchNorm_res[dnnResourceSrc] = bottom_data;
  BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
  BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;

  BatchNorm_res[dnnResourceDiffDst] = bwd_top_diff->get_converted_prv(top[0],
          true);
  if (bwd_bottom_diff->conversion_needed()) {
    bottom[0]->set_prv_diff_descriptor(bwd_bottom_diff);
    BatchNorm_res[dnnResourceDiffSrc] = bottom[0]->mutable_prv_diff();
  } else {
    BatchNorm_res[dnnResourceDiffSrc] = bottom[0]->mutable_cpu_diff();
  }

  PERFORMANCE_MEASUREMENT_BEGIN();
  e = dnnExecute<Dtype>(batchNormBwdData, BatchNorm_res);
  PERFORMANCE_MEASUREMENT_END_STATIC("BW_mkl_batch_norm");

  CHECK_EQ(e, E_SUCCESS);

  if (use_weight_bias_) {
    void* BatchNormBwdScaleShift_res[dnnResourceNumber];
    BatchNormBwdScaleShift_res[dnnResourceSrc] = bottom_data;
    BatchNormBwdScaleShift_res[dnnResourceWorkspace] = workspace_buffer_;
    BatchNormBwdScaleShift_res[dnnResourceDiffScaleShift] = scaleShift_buffer_;
    BatchNormBwdScaleShift_res[dnnResourceDiffDst] =
        BatchNorm_res[dnnResourceDiffDst];

    PERFORMANCE_MEASUREMENT_BEGIN();
    e = dnnExecute<Dtype>(batchNormBwdScaleShift, BatchNormBwdScaleShift_res);
    PERFORMANCE_MEASUREMENT_END_STATIC("BW_mkl_batch_norm");

    CHECK_EQ(e, E_SUCCESS);
    // Store ScaleShift blobs
    Dtype* diff_scale = this->blobs_[0]->mutable_cpu_diff();
    Dtype* diff_shift = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < channels_; i++) {
      diff_scale[i] =  scaleShift_buffer_[i];
      diff_shift[i] =  0;
      if (bias_term_) {
         diff_shift[i] =  scaleShift_buffer_[channels_ + i];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(MKLBatchNormLayer);
#else
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLBatchNormLayer);
// REGISTER_LAYER_CLASS(MKLBatchNorm);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED)
