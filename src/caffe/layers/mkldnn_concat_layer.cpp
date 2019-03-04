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
void MKLDNNConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  VLOG(1) << "MKLDNNConcatLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";

  int dim_src = bottom[0]->shape().size();
  //  int dim_dst = dim_src;

  num_concats_ = bottom.size();

  const int num_axes = bottom[0]->num_axes();
  if (concat_param.has_concat_dim()) {
    concat_dimension = static_cast<int>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost certainly unintended.
    CHECK_GE(concat_dimension, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dimension < " << kMaxBlobAxes;
    CHECK_LT(concat_dimension, num_axes) << "concat_dimension out of range.";
  } else {
    concat_dimension = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }

  vector<int> bottom_0_shape = bottom[0]->shape();
  bottom_0_shape[concat_dimension] = 0;
  for (auto i = 1; i < num_concats_; ++i) {
    vector<int> bottom_i_shape = bottom[i]->shape();
    bottom_i_shape[concat_dimension] = 0;
    if (concat_dimension == 1 && concat_param.per_fla_fuse()) {
      for(int i = 0; i < concat_dimension + 1; i++) CHECK_EQ(bottom_i_shape[i] == bottom_0_shape[i], true);
      continue;
    }
    CHECK_EQ(bottom_0_shape == bottom_i_shape, true);
  }

  split_dims.reserve(num_concats_);
  shape_ = bottom[0]->shape();
  shape_[concat_dimension] = 0;
  if (concat_dimension == 1 && concat_param.per_fla_fuse()) {
    for(int i= concat_dimension + 1; i < shape_.size(); i++) shape_[i] = 1;
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->count(concat_dimension);
      shape_[concat_dimension] += split_dims[i];
    }
  } else {
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->shape(concat_dimension);
      shape_[concat_dimension] += split_dims[i];
    }
  }
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  VLOG(1) << "MKLDNNConcatLayer<Dtype>::Reshape: "  << this->layer_param_.name();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();

  vector<int> dim = bottom[0]->shape();
  int dim_src = bottom[0]->shape().size();

  split_dims.clear();
  dim[concat_dimension] = 0;
  if (concat_dimension == 1 && concat_param.per_fla_fuse()) {
    for(int i= concat_dimension + 1; i < shape_.size(); i++) dim[i] = 1;
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->count(concat_dimension);
      dim[concat_dimension] += split_dims[i];
    }
  } else {
    for (auto i = 0; i < num_concats_; ++i) {
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      split_dims[i] = bottom[i]->shape(concat_dimension);
      dim[concat_dimension] += split_dims[i];
    }
  } 

  this->reshape = (dim != shape_);
  shape_ = dim;

  top[0]->Reshape(shape_);
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::InitConcatFwd(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;

  engine cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type usr_dt = memory::data_type::f32;
  memory::data_type prv_dt = usr_dt;
  // memory::format mfmt_any = memory::format::any;
  memory::format mfmt_out;
  if(this->shape_.size() == 5) {
    mfmt_out = memory::format::ncdhw;
  } else {
    CHECK_LE(this->shape_.size(), 4) << "mkldnn concat layer doesn't support this dim size!";
    mfmt_out = memory::format::nchw;
  }

  memory::dims output_tz = this->shape_;
  if(output_tz.size() < 4) output_tz.resize(4, 1); // resize to nchw with dim 1
  std::vector<memory::primitive_desc> srcs_mpd;
  std::vector<primitive::at> srcs;
  fwd_bottom_data.clear();
  fwd_input_primitives_.clear();
  fwd_input_primitives_at_.clear();

  float scale = 1.;
  float scale_min = 1.;
  for (auto i = 0; i < num_concats_; i++) {
      if (const_cast<Dtype*>(bottom[i]->prv_data()) != NULL) {
          shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[i]);
          scale = mem_descr->get_scale(0);
          if (scale_min == 1.) scale_min = scale;
          if(scale != 1. && scale < scale_min) scale_min = scale;
      }
  }
  std::vector<memory::format> src_mfmts;
  std::vector<shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> >> mem_descr;

  std::vector<memory::data_type> prv_dt_tmp(num_concats_, memory::data_type::f32);
  bool different_input_dt = false;
  for(auto i = 0; i < num_concats_; i++) {
    if (const_cast<Dtype*>(bottom[i]->prv_data()) != NULL) {
      shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr_tmp = get_mkldnn_prv_descriptor<Dtype, false>(bottom[i]);
      prv_dt_tmp[i] = static_cast<memory::data_type>(mem_descr_tmp->prv_memory_pd()->desc().data.data_type);
    }
  }

  memory::data_type first_prv_dt = prv_dt_tmp[0];
  for (auto i = 0; i < prv_dt_tmp.size(); i++){
    if (prv_dt_tmp[i] != first_prv_dt){
      different_input_dt = true;
    }
  }

  for (auto i = 0; i < num_concats_; i++) {
    fwd_bottom_data.push_back(boost::shared_ptr<MKLDNNData<Dtype> >());
    mem_descr.push_back(boost::shared_ptr<MKLDNNMemoryDescriptor<Dtype, false>>());

    memory::dims input_tz = this->shape_;
    if(input_tz.size() < 4) input_tz.resize(4, 1); // resize to nchw with dim 1
    input_tz[concat_dimension] = split_dims[i];

    memory::format src_mfmt = mfmt_out;
    shared_ptr<memory::primitive_desc> prv_src_mpd;
    shared_ptr<memory::primitive_desc> usr_src_mpd(
        new memory::primitive_desc({input_tz, usr_dt, mfmt_out}, cpu_engine));
 
    if (const_cast<Dtype*>(bottom[i]->prv_data()) != NULL) {
      scale = 1.;
      mem_descr[i]  = get_mkldnn_prv_descriptor<Dtype, false>(bottom[i]);
      if(!different_input_dt){
        src_mfmt = static_cast<memory::format>(
            mem_descr[i]->prv_memory_pd()->desc().data.format);
        prv_dt = static_cast<memory::data_type>(mem_descr[i]->prv_memory_pd()->desc().data.data_type);
        scale = mem_descr[i]->get_scale(0);
        if(scale != 1.) scale = scale_min;
      } 
      prv_src_mpd.reset(new memory::primitive_desc(
            {input_tz, prv_dt, src_mfmt}, cpu_engine));
    }
    std::vector<float> scale_bottom;
    scale_bottom.push_back(scale);

    src_mfmts.push_back(src_mfmt);
    srcs_mpd.push_back(memory::primitive_desc(
          {input_tz, prv_dt, src_mfmt}, cpu_engine));

    fwd_bottom_data[i].reset(new MKLDNNData<Dtype>(
          usr_src_mpd, prv_src_mpd, bottom[i], this, scale_bottom));
    fwd_input_primitives_.push_back(fwd_bottom_data[i]->create_input(false));
    fwd_input_primitives_at_.push_back(*fwd_input_primitives_[i]);
  }

  shared_ptr<memory::primitive_desc> usr_dst_mpd(new memory::primitive_desc(
        {output_tz, usr_dt, mfmt_out}, cpu_engine));

  concatFwd_pd.reset(new concat::primitive_desc(concat_dimension, srcs_mpd));

  shared_ptr<memory::primitive_desc> prv_dst_mpd(new memory::primitive_desc(
        concatFwd_pd->dst_primitive_desc()));

  std::vector<float> scale_top;
  if(!different_input_dt){
    scale_top.push_back(scale_min);
  } else{
    scale_top.push_back(1.);
  }
  fwd_top_data.reset(new MKLDNNData<Dtype>(usr_dst_mpd, prv_dst_mpd, top[0],
        this, scale_top));
 
  fwd_output_memory = fwd_top_data->create_output_memory();

  memory::format base_mfmt = mfmt_out;
  this->in_place_ = true;

  for(auto i = 0; i < num_concats_; i++){
    if(i == 0) {
      base_mfmt = src_mfmts[i];
    }
    else if((concat_dimension != 0 && bottom[i]->shape()[concat_dimension - 1] != 1) || base_mfmt != src_mfmts[i] || scale_top[0] != 1.|| different_input_dt || bottom[i]->prv_data() == NULL) {
      this->in_place_ = false;
      break;
    }
  }

  if(this->in_place_ && this->phase_ == TEST) {
    size_t offset = 0;     
    for(auto i = 0; i < num_concats_; i++){
      if(bottom[i]->prv_data()){
          caffe_copy(bottom[i]->count(), static_cast<Dtype*>(mem_descr[i]->get_prv_memory()->get_data_handle()), static_cast<Dtype*>(fwd_output_memory->get_data_handle()) + offset);
          mem_descr[i]->get_prv_memory()->set_data_handle(static_cast<Dtype*>(fwd_output_memory->get_data_handle()) + offset);
      } else{
        caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), static_cast<Dtype*>(fwd_output_memory->get_data_handle()) + offset);
        bottom[i]->set_cpu_data(static_cast<Dtype*>(fwd_output_memory->get_data_handle()) + offset);
      }
      offset += bottom[i]->count();
    }
  }

  concatFwd.reset(new concat(*concatFwd_pd, fwd_input_primitives_at_, *fwd_output_memory));

  for (auto i = 0; i < num_concats_; i++) {
    //fwd_bottom_data[i]->set_mkldnn_primitive(concatFwd);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(fwd_input_primitives_[i]);
    fwd_bottom_data[i]->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);
  }
  //fwd_top_data->set_mkldnn_primitive(concatFwd);          //Wrong passed primitive! (TODO: Checking!)
  MKLDNNPrimitive<Dtype> fwd_top_data_memory_transfer(fwd_output_memory);
  fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::InitConcatBwd(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;

  engine cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type data_type = memory::data_type::f32;
  // memory::format mfmt_any = memory::format::any;
  memory::format mfmt_out;
  memory::dims offsets;
  if(this->shape_.size() == 5) {
    mfmt_out =  memory::format::ncdhw;
    offsets = {0, 0, 0, 0, 0};
  } else {
    mfmt_out =  memory::format::nchw;
    offsets = {0, 0, 0, 0};
  }

  memory::format diff_dst_mfmt = mfmt_out;

  memory::dims input_tz = this->shape_;
  if (input_tz.size() < 4) input_tz.resize(4, 1);

  shared_ptr<memory::primitive_desc> prv_diff_dst_mpd;
  shared_ptr<memory::primitive_desc> usr_diff_dst_mpd(
    new memory::primitive_desc({input_tz, data_type, mfmt_out},
        cpu_engine));

  bool top_diff_is_prv = (const_cast<Dtype*>(top[0]->prv_diff()) != NULL);

  if (top_diff_is_prv) {
    shared_ptr<MKLDNNMemoryDescriptor<Dtype, true> > mem_descr
      = get_mkldnn_prv_descriptor<Dtype, true>(top[0]);
    diff_dst_mfmt = static_cast<memory::format>(
        mem_descr->prv_memory_pd()->desc().data.format);
    prv_diff_dst_mpd.reset(new memory::primitive_desc(
          {input_tz, data_type, diff_dst_mfmt}, cpu_engine));
  }

  bwd_top_diff.reset(new MKLDNNDiff<Dtype>(
        usr_diff_dst_mpd, prv_diff_dst_mpd, top[0], this));

  bwd_reorder_input_memory = bwd_top_diff->create_input(false);

  bwd_bottom_diff.clear();
  reorders.clear();
  bwd_reorder_output_memory.clear();
  for (auto i = 0; i < num_concats_; i++) {
    bwd_bottom_diff.push_back(boost::shared_ptr<MKLDNNDiff<Dtype> >());
    reorders.push_back(MKLDNNPrimitive<Dtype>());

    memory::dims dims = this->shape_;
    if (dims.size() < 4) dims.resize(4, 1);
    dims[concat_dimension] = split_dims[i];

    shared_ptr<memory::primitive_desc> usr_diff_src_mpd(
      new memory::primitive_desc({dims, data_type, mfmt_out},
          cpu_engine));
    shared_ptr<memory::primitive_desc> prv_diff_src_mpd(
      new memory::primitive_desc({dims, data_type, diff_dst_mfmt},
          cpu_engine));
    bwd_bottom_diff[i].reset(new MKLDNNDiff<Dtype>(
          usr_diff_src_mpd, prv_diff_src_mpd, bottom[i], this));

    auto view_pd = top_diff_is_prv ?
      view::primitive_desc(*prv_diff_dst_mpd, dims, offsets) :
      view::primitive_desc(*usr_diff_dst_mpd, dims, offsets);
    auto view_dst_pd = view_pd.dst_primitive_desc();
    auto reorder_pd = reorder::primitive_desc(view_dst_pd, *prv_diff_src_mpd);

    bwd_reorder_output_memory.push_back(bwd_bottom_diff[i]->create_output_memory());

    reorders[i].reset(new reorder(reorder_pd, *bwd_reorder_input_memory,
          *bwd_reorder_output_memory[i]));

    offsets[concat_dimension] += dims[concat_dimension];

    //bwd_bottom_diff[i]->set_mkldnn_primitive(reorders[i]);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_bottom_diff_memory_transfer(bwd_reorder_output_memory[i]);
    bwd_bottom_diff[i]->set_mkldnn_primitive(bwd_bottom_diff_memory_transfer);
  }

  //bwd_top_diff->set_mkldnn_primitive(reorders[0]);          //Wrong passed primitive! (TODO: Checking!)
  MKLDNNPrimitive<Dtype> bwd_top_diff_memory_transfer(bwd_reorder_input_memory);
  bwd_top_diff->set_mkldnn_primitive(bwd_top_diff_memory_transfer);
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  VLOG(1) << "MKLDNNConcatLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
  LOG(INFO) << "MKLDNNConcatLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#endif
  bool _mkldnn_primitive = false;
  if ((NULL == concatFwd_pd) || (true == this->reshape)) {
    InitConcatFwd(bottom, top);
    _mkldnn_primitive = true;
  }

  for (auto i = 0; i < num_concats_; i++) {
    // making reorders if needed.
    fwd_bottom_data[i]->sync_before_read();
  }
    // update top that head at prv
    fwd_top_data->sync_before_write();

  if(!this->in_place_ || this->phase_ != TEST) {
    PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    concatFwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);

    if(_mkldnn_primitive) {
      for (auto i = 0; i < num_concats_; i++) {
        CircleBuf::Instance()->DecRefCnt(bottom[i]->prv_data());
      }
    } 
  }
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                           ,const vector<bool>& propagate_down
                                           ,const vector<Blob<Dtype>*>& bottom)
{
  VLOG(1) << "MKLDNNConcatLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
  LOG(INFO) << "MKLDNNConcatLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#endif

  if ((reorders.size() == 0) || (true == this->reshape)) {
    bool concat_axis = true;
    for (auto i = 0; i < num_concats_; i++) {
      if (bottom[i]->shape()[concat_dimension] % 16 != 0) {
        concat_axis = false;
        break;
      }
    }
    // mkldnn view primitive creation has restriction if viewed area (offset or size) is not
    // aligned on block size (would assert if such case happens).
    // This case usually would be triggered when the bottom blob's dim along concat axis is
    // not 16-dividable and the top blob's block size becomes 16 due to the following mkldnn
    // layer's reorder.
    // To avoid such assertion, we have to explicitly convert the top blob state from prv to cpu state.
    if (top[0]->prv_diff() != NULL && !concat_axis)
      top[0]->mutable_cpu_diff();
    InitConcatBwd(top, propagate_down, bottom);
  }
  bwd_top_diff->sync_before_read();
  for (auto i = 0; i < num_concats_; ++i) {
    bwd_bottom_diff[i]->sync_before_write();
    PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKLDNN_NAME("BW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    reorders[i].submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNConcatLayer);
#else
template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNConcatLayer);

} // namespace caffe

#endif
