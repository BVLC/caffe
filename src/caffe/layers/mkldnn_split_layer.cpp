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
#ifdef MKLDNN_SUPPORTED
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
MKLDNNSplitLayer<Dtype>::~MKLDNNSplitLayer() { }

template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_t count = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count, top[i]->count());
  }
  size_t dim_src = bottom[0]->shape().size();
  this->reshape = false;
  if (this->sizes_src_.size() != dim_src || this->strides_src_.size() != dim_src) {
    this->sizes_src_.resize(dim_src);
    this->strides_src_.resize(dim_src);
    this->reshape = true;
  }
  for (size_t d = 0; d < dim_src; ++d) {
    if (this->sizes_src_[d] != bottom[0]->shape()[d]) {
      this->sizes_src_[d] = bottom[0]->shape()[d];
      this->reshape = true;
    }
    size_t stride = (d == 0) ? 1 : this->strides_src_[d-1]*this->sizes_src_[d-1];
    if (this->strides_src_[d] != stride) {
      this->strides_src_[d] = stride;
      this->reshape = true;
    }
  }

  // TODO: Add checking to reinitialize Backward, to be
  // done when Reshape is to be supported by MKLDNN layers
}

template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::InitSplitBwd(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;

  // We just do simple adding so scale is 1.0 for all inputs we have
  std::vector<float> scale(top.size(), 1.0);
  engine cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type data_type = memory::data_type::f32;
  // TODO: shouldn't we have format here that is well suited for earlier layer.
  // eg. Netcompiler should some of knowledge provided
  memory::format mfmt_nchw = memory::format::nchw;
  memory::format diff_dst_mfmt = mfmt_nchw;

  // Dimensions of bottom and top blobs. There is a number of
  // top blobs each of the same size as the bottom one
  memory::dims bottom_tz;
  bottom_tz.resize(4);
  for(int i=0; i<4; i++) {
    if(i < this->sizes_src_.size()) {
      bottom_tz[i] = static_cast<int>(this->sizes_src_[i]);
    } else {
      bottom_tz[i] = 1;
    }
  }

  shared_ptr<memory::primitive_desc> prv_diff_dst_mpd;
  shared_ptr<memory::primitive_desc> usr_diff_dst_mpd(
    new memory::primitive_desc({bottom_tz, data_type, mfmt_nchw},
        cpu_engine));

  // We will get final destination layout of bottom diff after first top...
  bool first_top_diff_is_prv = (const_cast<Dtype*>(top[0]->prv_diff()) != NULL);

  if (first_top_diff_is_prv) {
    shared_ptr<MKLDNNMemoryDescriptor<Dtype, true> > mem_descr
      = get_mkldnn_prv_descriptor<Dtype, true>(top[0]);
    diff_dst_mfmt = static_cast<memory::format>(
        mem_descr->prv_memory_pd()->desc().data.format);
  }
  prv_diff_dst_mpd.reset(new memory::primitive_desc(
        {bottom_tz, data_type, diff_dst_mfmt}, cpu_engine));

  bwd_bottom_diff_.reset(new MKLDNNDiff<Dtype>(
        usr_diff_dst_mpd, prv_diff_dst_mpd, bottom[0], this));
  bwd_bottom_diff_memory_ = bwd_bottom_diff_->create_output_memory();

  memory::dims top_tz = bottom_tz;
  shared_ptr<memory::primitive_desc> usr_diff_src_mpd(
    new memory::primitive_desc({top_tz, data_type, mfmt_nchw},
        cpu_engine));

  // Gather diff descriptors of top difs (inputs for BW)
  std::vector<memory::primitive_desc> prv_diff_srcs_mpd;
  boost::shared_ptr<memory::primitive_desc> mpd_ptr;
  bwd_top_diffs_.clear();
  bwd_top_diff_primitives_.clear();
  bwd_top_diffs_primitives_at_.clear();
  for (int i = 0; i < top.size(); ++i) {
    // If diff is in private layout then copy descriptor from it
    memory::format diff_src_mfmt = mfmt_nchw;
    bool top_diff_is_prv = top[i]->prv_diff() != NULL;
    if (top_diff_is_prv) {
      shared_ptr<MKLDNNMemoryDescriptor<Dtype, true> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype, true>(top[i]);
      diff_src_mfmt = static_cast<memory::format>(
          mem_descr->prv_memory_pd()->desc().data.format);
    }
    prv_diff_srcs_mpd.push_back(memory::primitive_desc(
          {top_tz, data_type, diff_src_mfmt}, cpu_engine));

    mpd_ptr.reset(new memory::primitive_desc({top_tz, data_type, diff_src_mfmt},
                                             cpu_engine) );
    bwd_top_diffs_.push_back(boost::shared_ptr<MKLDNNDiff<Dtype> >());
    bwd_top_diffs_[i].reset(new MKLDNNDiff<Dtype>(
          usr_diff_src_mpd, mpd_ptr, top[i], this));
    bwd_top_diff_primitives_.push_back(bwd_top_diffs_[i]->create_input(false));
    bwd_top_diffs_primitives_at_.push_back(*bwd_top_diff_primitives_[i]);
  }

  // ---- Determining engine to use -----------------------
  std::string subengines = this->layer_param_.engine();
  if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
    subengines = "MKLDNN:CPU";
  splitBwd_pd_.reset(new sum::primitive_desc({bottom_tz, data_type, diff_dst_mfmt},scale, prv_diff_srcs_mpd));
  CHECK(splitBwd_pd_);

  splitBwd_.reset(new sum(*splitBwd_pd_, bwd_top_diffs_primitives_at_, *bwd_bottom_diff_memory_));

  // Descriptors need to have Split primitive referenced as
  // there may be reorders to be done for inputs(tops' diffs) 
  // so it match SplitBwd primitive inputs format expectations
  for(int i = 0; i < top.size(); ++i) {
    //bwd_top_diffs_[i]->set_mkldnn_primitive(splitBwd_);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_top_diff_primitive_transfer(bwd_top_diff_primitives_[i]);
    bwd_top_diffs_[i]->set_mkldnn_primitive(bwd_top_diff_primitive_transfer);
  }

  //bwd_bottom_diff_->set_mkldnn_primitive(splitBwd_);        //Wrong passed primitive! (TODO: Checking!)
  MKLDNNPrimitive<Dtype> bwd_bottom_diff_memory_transfer(bwd_bottom_diff_memory_);
  bwd_bottom_diff_->set_mkldnn_primitive(bwd_bottom_diff_memory_transfer);
}


template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // TODO: consider doing something
}

template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNSplitLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
    // If no gradient to be computed for eariler layers then we do need to do
    //  any computation
    if (!propagate_down[0]) {
        return;
    }
    if (splitBwd_pd_ == NULL || this->reshape) {
        InitSplitBwd(bottom, top);
    }
    
    for(int i = 0; i < top.size(); ++i) {
      bwd_top_diffs_[i]->sync_before_read();
    }

    bwd_bottom_diff_->sync_before_write();

    PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKLDNN_NAME("BW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    splitBwd_.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNSplitLayer);
#else
template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLDNNSplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLDNNSplitLayer);

} // namespace caffe

#endif
