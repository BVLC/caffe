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

#ifdef MKL2017_SUPPORTED
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/performance.hpp"

namespace caffe {

template <typename Dtype>
MKLPoolingLayer<Dtype>::~MKLPoolingLayer() {
  dnnDelete<Dtype>(poolingFwd);
  dnnDelete<Dtype>(poolingBwd);
}

template <typename Dtype>
void MKLPoolingLayer<Dtype>::Init(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();

  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }

  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      bottom[0]->height() + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      bottom[0]->width() + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  bool force_exclude_padding_flag_ = false;
  if (pad_h_ || pad_w_ || kernel_h_ == 1 || kernel_w_ == 1) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= bottom[0]->height() + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= bottom[0]->width() + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, bottom[0]->height() + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, bottom[0]->width() + pad_w_);
  }
  else
  {
    force_exclude_padding_flag_ = true;
  }

  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    (reinterpret_cast<Blob<size_t>* > (top[1]) )->Reshape(bottom[0]->num(),
            channels_, pooled_height_, pooled_width_);
  }
  // If max/min/avg pooling, we will initialize the vector index part.
  if (top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    this->algorithm = dnnAlgorithmPoolingMax;
    break;
  case PoolingParameter_PoolMethod_AVE:
    if (this->layer_param_.pooling_param().avg_include_pad()) {
        this->algorithm = dnnAlgorithmPoolingAvgIncludePadding;
    }
    else {
        this->algorithm = dnnAlgorithmPoolingAvgExcludePadding;
    }
    // If user did not define padding
    // bottom[0]->height/width() + kernel_h/w_ cannot be exact division by stride_h/w_
    // use the exclude padding to align with the result of Caffe
    // for exact division situation, exclude padding and include padding will have the same results
    if (force_exclude_padding_flag_ == true)
    {
        this->algorithm = dnnAlgorithmPoolingAvgExcludePadding;
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }


  dim = 4;

  src_sizes[0] = bottom[0]->width();
  src_sizes[1] = bottom[0]->height();
  src_sizes[2] = bottom[0]->channels();
  src_sizes[3] = bottom[0]->num();

  src_strides[0] = 1;
  src_strides[1] = src_sizes[0];
  src_strides[2] = src_sizes[0]*src_sizes[1];
  src_strides[3] = src_sizes[0]*src_sizes[1]*src_sizes[2];

  dst_sizes[0] = pooled_width_;
  dst_sizes[1] = pooled_height_;
  dst_sizes[2] = src_sizes[2];
  dst_sizes[3] = src_sizes[3];

  dst_strides[0] = 1;
  dst_strides[1] = dst_sizes[0];
  dst_strides[2] = dst_sizes[0]*dst_sizes[1];
  dst_strides[3] = dst_sizes[0]*dst_sizes[1]*dst_sizes[2];

  src_offset[0] = -pad_w_;
  src_offset[1] = -pad_h_;

  kernel_stride[0] = stride_w_;
  kernel_stride[1] = stride_h_;

  kernel_size[0] = kernel_w_;
  kernel_size[1] = kernel_h_;

  // Names are for debugging only
  fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
  fwd_top_data->name =    "fwd_top_data      @ " + this->layer_param_.name();
  bwd_top_diff->name =    "bwd_top_diff      @ " + this->layer_param_.name();
  bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->layer_param_.name();

  fwd_top_data   ->create_user_layout(dim, dst_sizes, dst_strides, false);
  bwd_bottom_diff->create_user_layout(dim, src_sizes, src_strides, false);
  bwd_top_diff   ->create_user_layout(dim, dst_sizes, dst_strides, false);
  // Primitives will be allocated during the first fwd pass
  dnnDelete<Dtype>(poolingFwd);
  dnnDelete<Dtype>(poolingBwd);
}

template <typename Dtype>
void MKLPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Init(bottom, top);
}

template <typename Dtype>
void MKLPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";

  if (channels_ == bottom[0]->channels() &&
      height_ == bottom[0]->height() &&
      width_ == bottom[0]->width() &&
      num_ == bottom[0]->num()) {
    reshape = false;
    return;
  }
  reshape = true;
  Init(bottom, top);
}

template <typename Dtype>
void MKLPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // We'll output the mask to top[1] if it's of size >1.
  size_t* mask = NULL;  // suppress warnings about uninitalized variables

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  dnnError_t status;
  void* pooling_res[dnnResourceNumber];

  mask = (use_top_mask) ?
      reinterpret_cast<size_t*>(top[1]->mutable_cpu_data()) :
      (max_idx_.mutable_cpu_data());
  pooling_res[dnnResourceWorkspace] = reinterpret_cast<void*>(mask);

  void* bottom_data =
    reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->prv_data()));
  if (NULL == bottom_data) {
    bottom_data =
      reinterpret_cast<void *>(const_cast<Dtype*>(bottom[0]->cpu_data()));
    if (NULL == poolingFwd || reshape) {
      // Now create poolingFwd
      fwd_bottom_data->create_user_layout(dim, src_sizes, src_strides, false);
      status = dnnPoolingCreateForward<Dtype>(&poolingFwd, NULL,
              this->algorithm, fwd_bottom_data->layout_usr,
              kernel_size, kernel_stride, src_offset, dnnBorderZeros);
      CHECK_EQ(status, E_SUCCESS);

      // Now create poolingBwd
      status = dnnPoolingCreateBackward<Dtype>(&poolingBwd, NULL,
              this->algorithm, fwd_bottom_data->layout_usr,
              kernel_size, kernel_stride, src_offset, dnnBorderZeros);
      CHECK_EQ(status, E_SUCCESS);
    }
  } else if (NULL == poolingFwd || reshape) {
    // Is it the first pass? Create a primitive.
    CHECK_EQ((bottom[0]->get_prv_data_descriptor())->get_descr_type(),
            PrvMemDescr::PRV_DESCR_MKL2017);
    shared_ptr<MKLData<Dtype> > mem_descr
      =  boost::static_pointer_cast<MKLData<Dtype> >
            (bottom[0]->get_prv_data_descriptor());
    CHECK(mem_descr != NULL);

    DLOG(INFO) << "Using layout of " << mem_descr->name
            << " as input layout for " << this->layer_param_.name();
    // copy shared_ptr
    fwd_bottom_data = mem_descr;

    // Now create poolingFwd
    status = dnnPoolingCreateForward<Dtype>(&poolingFwd, NULL,
            this->algorithm, fwd_bottom_data->layout_int, kernel_size,
            kernel_stride, src_offset, dnnBorderZeros);
    CHECK_EQ(status, E_SUCCESS);

    fwd_top_data->create_internal_layout(poolingFwd, dnnResourceDst);

    // Now create poolingBwd
    status = dnnPoolingCreateBackward<Dtype>(&poolingBwd, NULL,
            this->algorithm, fwd_bottom_data->layout_int, kernel_size,
            kernel_stride, src_offset, dnnBorderZeros);
    CHECK_EQ(status, E_SUCCESS);

    bwd_top_diff   ->create_internal_layout(poolingFwd, dnnResourceDst);
    bwd_bottom_diff->create_internal_layout(poolingFwd, dnnResourceSrc);
  }

  pooling_res[dnnResourceSrc] = bottom_data;
  if (fwd_top_data->conversion_needed()) {
    top[0]->set_prv_data_descriptor(fwd_top_data);
    pooling_res[dnnResourceDst] =
            reinterpret_cast<void *>(top[0]->mutable_prv_data());
  } else {
    pooling_res[dnnResourceDst] =
            reinterpret_cast<void *>(top[0]->mutable_cpu_data());
    DLOG(INFO) << "Using cpu_data for top in DnnPooling.";
  }
  PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKL_NAME("FW"));
  PERFORMANCE_MEASUREMENT_BEGIN();
  status = dnnExecute<Dtype>(poolingFwd, pooling_res);
  PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);

  CHECK_EQ(status, E_SUCCESS);
}

template <typename Dtype>
void MKLPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.

  const size_t* mask = NULL;  // suppress warnings about uninitialized variables

  // The main loop
  dnnError_t e;
  void* pooling_res[dnnResourceNumber];

  mask = (top.size() > 1) ?
    reinterpret_cast<const size_t*>(top[1]->cpu_data()) :
    (max_idx_.cpu_data());
  pooling_res[dnnResourceWorkspace] =
    reinterpret_cast<void *>(const_cast<size_t*>(mask));

  pooling_res[dnnResourceDiffDst] = bwd_top_diff->get_converted_prv(top[0],
          true);

  if (bwd_bottom_diff->conversion_needed()) {
    bottom[0]->set_prv_diff_descriptor(bwd_bottom_diff);
    pooling_res[dnnResourceDiffSrc] = bottom[0]->mutable_prv_diff();
  } else {
    pooling_res[dnnResourceDiffSrc] = bottom[0]->mutable_cpu_diff();
  }
  caffe_set(bottom[0]->count(), Dtype(0),
          reinterpret_cast<Dtype *>(pooling_res[dnnResourceDiffSrc]));

  PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKL_NAME("BW"));
  PERFORMANCE_MEASUREMENT_BEGIN();
  e = dnnExecute<Dtype>(poolingBwd, pooling_res);
  PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);

  CHECK_EQ(e, E_SUCCESS);
}


#ifdef CPU_ONLY
STUB_GPU(MKLPoolingLayer);
#else
template <typename Dtype>
void MKLPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLPoolingLayer);
}  // namespace caffe
#endif  // #ifdef MKL2017_SUPPORTED
