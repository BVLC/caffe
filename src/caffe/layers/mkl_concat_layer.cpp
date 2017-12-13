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

#include "caffe/layer.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/performance.hpp"

namespace caffe {

template <typename Dtype> MKLConcatLayer<Dtype>::~MKLConcatLayer() {
  dnnDelete<Dtype>(concatFwd_);
  dnnDelete<Dtype>(concatBwd_);
  delete[] split_channels_;
}

template <typename Dtype>
void MKLConcatLayer<Dtype>::Init(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  size_t dim_src = bottom[0]->shape().size();
  size_t dim_dst = dim_src;

  num_concats_ = bottom.size();
  channels_ = 0;

  for (size_t i = 1; i < num_concats_; ++i) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    CHECK_EQ(bottom[0]->height(), bottom[i]->height());
    CHECK_EQ(bottom[0]->width(), bottom[i]->width());
  }


  delete[] split_channels_;
  split_channels_ = new size_t[num_concats_];
  for (size_t i = 0; i < num_concats_; ++i) {
    CHECK_EQ(dim_src, bottom[i]->shape().size());

    fwd_bottom_data_.push_back(shared_ptr<MKLData<Dtype> >(new MKLData<Dtype>));
    bwd_bottom_diff_.push_back(shared_ptr<MKLDiff<Dtype> >(new MKLDiff<Dtype>));
    fwd_bottom_data_[i]->name = "fwd_bottom_data_[i]";
    bwd_bottom_diff_[i]->name = "bwd_bottom_data[i]";

    // TODO: should be a helper function
    size_t sizes_src[dim_src], strides_src[dim_src];
    for (size_t d = 0; d < dim_src; ++d) {
        sizes_src[d] = bottom[i]->shape()[dim_src - d - 1];
        strides_src[d] = (d == 0) ? 1 : strides_src[d - 1] * sizes_src[d - 1];
    }

    split_channels_[i] = bottom[i]->channels();
    channels_ += split_channels_[i];
    fwd_bottom_data_[i]->create_user_layout(dim_src,
                                            sizes_src,
                                            strides_src,
                                            false);
    bwd_bottom_diff_[i]->create_user_layout(dim_src,
                                            sizes_src,
                                            strides_src,
                                            false);
  }

  // XXX: almost the same computations as above for src
  size_t sizes_dst[dim_dst], strides_dst[dim_dst];
  for (size_t d = 0; d < dim_dst; ++d) {
    if (d == 2)
      sizes_dst[d] = channels_;
    else
      sizes_dst[d] = bottom[0]->shape()[dim_dst - 1 - d];
    strides_dst[d] = (d == 0) ? 1 : strides_dst[d - 1] * sizes_dst[d - 1];
  }
  bwd_top_diff_->create_user_layout(dim_dst, sizes_dst, strides_dst, false);
  fwd_top_data_->create_user_layout(dim_dst, sizes_dst, strides_dst, false);

  dnnDelete<Dtype>(concatFwd_);
  dnnDelete<Dtype>(concatBwd_);
}

template <typename Dtype>
void MKLConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  num_ = 0;
  height_ = 0;
  width_ = 0;
  Init(bottom, top);
}

template <typename Dtype>
void MKLConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  bool has_spatial = (bottom[0]->shape().size() != 2);
  if (has_spatial == false)
  {
#ifdef DEBUG
      LOG(INFO) << "size of bottom blob: " << bottom[0]->shape().size();
      LOG(INFO) << "size of top blob: " << top[0]->shape().size();
      LOG(INFO) << "MKL concat layer only support 4D blob as input! Reshape the 2D input blob into 4D for calculation!";
#endif
      for (auto i = 0; i < num_concats_; i++)
      {
          vector<int> bottom_4D_shape;
          int bottom_4D_height = 1;
          int bottom_4D_width = 1;
          bottom_4D_shape.push_back(bottom[i]->num());
          bottom_4D_shape.push_back(bottom[i]->channels());
          bottom_4D_shape.push_back(bottom_4D_height);
          bottom_4D_shape.push_back(bottom_4D_width);
          bottom[i]->Reshape(bottom_4D_shape, false);
      }      
  }
  if ((num_ == bottom[0]->num()) &&
       height_ == bottom[0]->height() &&
       width_ == bottom[0]->width()) {
       top[0]->Reshape(num_, channels_, height_, width_);
    return;
  }

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(num_, channels_, height_, width_);
  Init(bottom, top);
}

template <typename Dtype>
void MKLConcatLayer<Dtype>::Forward_cpu(const vector <Blob<Dtype>*>& bottom,
  const vector <Blob<Dtype>*>& top) {
  dnnError_t e;
  vector<void*> bottom_data;
  bool isFirstPass = (concatFwd_ == NULL);
  dnnLayout_t *layouts = NULL;
  if (isFirstPass) {
      layouts = new dnnLayout_t[num_concats_];
  }

  for (size_t n = 0; n < num_concats_; n++) {
    bottom_data.push_back(reinterpret_cast<void *>(
      const_cast<Dtype*>(bottom[n]->prv_data())));

    if (bottom_data[n] == NULL) {
      bottom_data[n] =
        reinterpret_cast<void *>(const_cast<Dtype*>(bottom[n]->cpu_data()));
      if (isFirstPass) {
        layouts[n] = fwd_bottom_data_[n]->layout_usr;
      }
    } else if (isFirstPass) {
      CHECK((bottom[n]->get_prv_data_descriptor())->get_descr_type() ==
        PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLData<Dtype> > mem_descr =
        boost::static_pointer_cast<MKLData<Dtype> >(
          bottom[n]->get_prv_data_descriptor());
      CHECK(mem_descr != NULL);

      fwd_bottom_data_[n] = mem_descr;
      layouts[n] = mem_descr->layout_int;
    }
  }

  if (isFirstPass) {
    e = dnnConcatCreate<Dtype>(&concatFwd_, NULL, num_concats_, layouts);
    CHECK_EQ(e, E_SUCCESS);

    fwd_top_data_->create_internal_layout(concatFwd_, dnnResourceDst);
    bwd_top_diff_->create_internal_layout(concatFwd_, dnnResourceDst);

    e = dnnSplitCreate<Dtype>(&concatBwd_, NULL, num_concats_,
      bwd_top_diff_->layout_int, split_channels_);
    CHECK_EQ(e, E_SUCCESS);

    for (size_t n = 0; n < num_concats_; ++n) {
      bwd_bottom_diff_[n]->create_internal_layout(concatBwd_,
          (dnnResourceType_t)(dnnResourceMultipleDst + n));
    }
  }

  delete[] layouts;

  void *concat_res[dnnResourceNumber];
  for (int n = 0; n < num_concats_; ++n) {
    concat_res[dnnResourceMultipleSrc + n]
      = reinterpret_cast<void*>(bottom_data[n]);
  }

  if (fwd_top_data_->conversion_needed()) {
    top[0]->set_prv_data_descriptor(fwd_top_data_);
    concat_res[dnnResourceDst] =
      reinterpret_cast<void*>(top[0]->mutable_prv_data());
  } else {
    concat_res[dnnResourceDst] =
      reinterpret_cast<void*>(top[0]->mutable_cpu_data());
  }

  PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKL_NAME("FW"));
  PERFORMANCE_MEASUREMENT_BEGIN();
  e = dnnExecute<Dtype>(concatFwd_, concat_res);
  PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);

  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MKLConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector <Blob<Dtype>*>& bottom) {
  int need_bwd = 0;
  for (size_t n = 0; n < num_concats_; n++) {
    need_bwd += propagate_down[n];
  }
  if (!need_bwd) {
    return;
  }

  dnnError_t e;
  void *concat_res[dnnResourceNumber];

  concat_res[dnnResourceSrc] = bwd_top_diff_->get_converted_prv(top[0], true);

  for (size_t i = 0; i < num_concats_; ++i) {
    if (bwd_bottom_diff_[i]->conversion_needed()) {
      bottom[i]->set_prv_diff_descriptor(bwd_bottom_diff_[i]);
      concat_res[dnnResourceMultipleDst + i] = bottom[i]->mutable_prv_diff();
    } else {
      concat_res[dnnResourceMultipleDst + i] = bottom[i]->mutable_cpu_diff();
    }
  }

  PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKL_NAME("BW"));
  PERFORMANCE_MEASUREMENT_BEGIN();
  e = dnnExecute<Dtype>(concatBwd_, concat_res);
  PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);

  CHECK_EQ(e, E_SUCCESS);
}

#ifdef CPU_ONLY
STUB_GPU(MKLConcatLayer);
#else
template <typename Dtype>
void MKLConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLConcatLayer);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED)
