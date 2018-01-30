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
#include <cstdlib>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
//#include "mkl_service.h"

// TODO: Correct process case if there are no bias
// TODO: Exception handling - mkl-dnn produces exceptions on errors

namespace caffe {

template <typename Dtype>
MKLDNNConvolutionLayer<Dtype>::MKLDNNConvolutionLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(param), ConvolutionLayer<Dtype>(param)
            , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
            , bwdd_weights_data(NULL), bwdw_bottom_data(NULL)
            , bwdd_bottom_diff(NULL), bwdd_top_diff(NULL)
            , bwdw_top_diff(NULL), bwdw_weights_diff(NULL), bwdw_bias_diff(NULL)
            , convFwd_pd(NULL), convBwdData_pd(NULL), convBwdWeights_pd(NULL)
            , fwd_top_data_memory(NULL), bwdd_bottom_diff_memory(NULL)
            , bwdw_weights_diff_memory(NULL), bwdw_bias_diff_memory(NULL)
            , fwd_bottom_data_primitive(NULL), fwd_weights_data_primitive(NULL), fwd_bias_data_primitive(NULL)
            , bwdd_top_diff_primitive(NULL), bwdd_weights_data_primitive(NULL)
            , bwdw_top_diff_primitive(NULL), bwdw_bottom_data_primitive(NULL)
            , width_(0), height_(0), width_out_(0), height_out_(0), kernel_w_(0), kernel_h_(0)
            , stride_w_(0), stride_h_(0), pad_w_(0), pad_h_(0),
            bwdw_weights_diff_iter(NULL),
            bwdw_bias_diff_iter(NULL),
            bwdw_weights_diff_memory_iter(NULL),
            bwdw_bias_diff_memory_iter(NULL)
{
  PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
  PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
  PERFORMANCE_EVENT_ID_RESET(perf_id_bw_weights_);
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::compute_output_shape()
{
    ConvolutionLayer<Dtype>::compute_output_shape();
    this->height_out_ = this->output_shape_[0];
    this->width_out_ = this->output_shape_[1];
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::init_properties(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    this->stride_w_ = this->stride_.cpu_data()[1];
    this->stride_h_ = this->stride_.cpu_data()[0];
    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->channels_ = bottom[0]->channels();
    this->num_ = bottom[0]->num();
    this->pad_w_ = this->pad_.cpu_data()[1];
    this->pad_h_ = this->pad_.cpu_data()[0];
    this->kernel_w_ = this->kernel_shape_.cpu_data()[1];
    this->kernel_h_  = this->kernel_shape_.cpu_data()[0];

    string _conv_algorithm = this->layer_param_.convolution_param().conv_algorithm();
    if(_conv_algorithm == "direct")
    {
        conv_algorithm = algorithm::convolution_direct;
    }
    else if(_conv_algorithm == "winograd")
    {
        conv_algorithm = algorithm::convolution_winograd;
    }
    else
    {
        LOG(ERROR) << "Unsupported convolution algorithm.";
        CHECK(false);
    }
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "<< MKLDNNConvolutionLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();
    if (this->layer_param_.has_quantization_param() && this->phase_ == TEST) this->need_quantize_ = true;

    ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
    init_properties(bottom, top);
    this->bottom_shape_ = &bottom[0]->shape();

    // support for (iter_size > 1) requires additional buffer for weights diff and bias diff
    // Because Net is initialized before Caffe::set_iter_size, so additional buffer should be new and set here
    bwdw_weights_diff_iter_blob.reset(new Blob<Dtype>());
    bwdw_weights_diff_iter_blob->ReshapeLike(*(this->blobs_[0]));
    if (this->bias_term_) {
      bwdw_bias_diff_iter_blob.reset(new Blob<Dtype>());
      bwdw_bias_diff_iter_blob->ReshapeLike(*(this->blobs_[1]));
    }
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << " MKLDNNConvolutionLayer<Dtype>::Reshape: " << this->layer_param_.name();
    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;
    init_properties(bottom, top);
    BaseConvolutionLayer<Dtype>::ReshapeForMKL(bottom, top);
#ifndef DISABLE_CONV_SUM_FUSION
    if (bottom.size() > 1) {
        top[0]->ShareData(*bottom[1]);
    }
#endif
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::InitConvolutionFwd(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value)   NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;
    bool relu = this->layer_param_.convolution_param().relu();
    Dtype negative_slope = 0;
    if(relu)
    {
        propagation = prop_kind::forward_inference;
        negative_slope = this->layer_param_.relu_param().negative_slope();
    }

    int32_t g  = std::max(this->group_, 1);
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    int32_t ow = this->width_out_;
    int32_t oh = this->height_out_;
    int32_t oc = this->num_output_;

    int32_t kw = this->kernel_w_;
    int32_t kh = this->kernel_h_;

    int32_t sw = this->stride_w_;
    int32_t sh = this->stride_h_;

    int32_t pw = this->pad_w_;
    int32_t ph = this->pad_h_;
    memory::dims convolutionStrides {sh, sw};
    memory::dims padding {ph, pw};
    memory::dims padding_r;
    memory::dims dilation;
    bool dilated_conv = false;
    const int* dilation_data = this->dilation_.cpu_data();
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      dilation.push_back(dilation_data[i] - 1);
      if (dilation_data[i] != 1) dilated_conv = true;
    }
    padding_r.push_back((oh - 1) * sh - ih + ((kh - 1) * (dilation_data[0]) + 1) - ph);
    padding_r.push_back((ow - 1) * sw - iw + ((kw - 1) * (dilation_data[1]) + 1) - pw);

    // ---- Initialize memory descriptors (fromat = any) to create convolution descriptor -------------
    memory::data_type mpcsn = memory::data_type::f32;
    memory::data_type bottom_dt = this->need_quantize_ ? memory::data_type::u8 : memory::data_type::f32;
    memory::data_type top_dt = memory::data_type::f32;

    if (this->need_quantize_) {
      if (this->bw_layer_out_ == 8) {
        if (relu) {
          top_dt = memory::data_type::u8;
        }
        else {
          top_dt = memory::data_type::s8;
        }
      }
      else {
        top_dt = memory::data_type::s32;
      }
    }

    bool is_sum;
    if (bottom.size() > 1) {
      is_sum = true;

      memory::data_type bottom_1_dt = memory::data_type::f32;
      if (const_cast<Dtype*>(bottom[1]->prv_data()) != NULL){
    
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > bottom_1_desc =
            get_mkldnn_prv_descriptor<Dtype, false>(bottom[1]);
        bottom_1_dt = static_cast<memory::data_type>(bottom_1_desc->prv_memory_pd()->desc().data.data_type);
      } 

      if (top_dt != bottom_1_dt) {
        top_dt = bottom_1_dt;
      }
    }

    memory::data_type weights_dt = this->need_quantize_ ? memory::data_type::s8 : memory::data_type::f32;
    memory::data_type bias_dt = this->need_quantize_ ? memory::data_type::s32 : memory::data_type::f32;
    memory::format mfmt_any = memory::format::any;

    memory::dims bottom_tz = {n, ic, ih, iw};
    memory::dims bias_tz = {oc};
    memory::dims top_tz = {n, oc, oh, ow};
    memory::dims weights_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};

    // ---- Memory descriptors for initializing of convolution primitive descriptor -------------
    memory::desc init_bottom_md({bottom_tz}, bottom_dt, mfmt_any);
    memory::desc init_bias_md({bias_tz}, bias_dt, mfmt_any);
    memory::desc init_top_md({top_tz}, top_dt, mfmt_any);
    memory::desc init_weights_md({weights_tz}, weights_dt, mfmt_any);

    primitive_attr attr;
    if (this->need_quantize_) {
      if(this->scale_in_.size() > 0) this->is_float_ = true;
      int mask = 0;
      int count = 1; //single channel
      if(this->fl_params_.size() > 1 || this->scale_params_.size() > 1){
          int oc_dim_id = 1;
          mask = 1 << oc_dim_id;
          count = oc;  //multi channel
      }
      std::vector<float> scales(count);
      float scale;
      if(this->is_float_){
        #pragma omp parallel for if (count > 1)
        for(int i=0; i<count; i++){
          scale = this->scale_out_[0] / (this->scale_in_[0] * this->scale_params_[i]);
          scales[i] = scale;
        }
      } else {
        int output_shift;
        #pragma omp parallel for if (count > 1)
        for(int i=0; i<count; i++){
          output_shift = this->fl_layer_out_[0] - this->fl_layer_in_[0] - this->fl_params_[i];
          scale = pow(2. ,output_shift);
          scales[i] = scale;
        }
      }
      attr.set_output_scales(mask, scales);
      attr.set_int_output_round_mode(round_nearest);
    }
    
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    mkldnn::algorithm eligibleAlgorithms[2] = {conv_algorithm, algorithm::convolution_direct};
    convFwd_pd = NULL;
    mkldnn::post_ops ops;

#ifndef DISABLE_CONV_SUM_FUSION
    if(relu || bottom.size() > 1) {
#else
    if(relu) {
#endif
        float scale = 1.0f;
        Dtype alpha = negative_slope;  // negative slope for mkldnn_eltwise_relu.
        float beta = 1.0f;  //ignored for mkldnn_eltwise_relu.
#ifndef DISABLE_CONV_SUM_FUSION
        if (bottom.size() > 1) {
          if (this->need_quantize_) {
            float sum_scale;
            if(this->is_float_){
                sum_scale = this->scale_out_[0] /
                      get_mkldnn_prv_descriptor<Dtype, false>(bottom[1])->get_scale(0);
            } else{
                int sum_shift =
                    this->fl_layer_out_[0] -
                    get_mkldnn_prv_descriptor<Dtype, false>(bottom[1])->get_fl(0);          
                sum_scale = pow(2., sum_shift);
            } 
            ops.append_sum(sum_scale);
          } else {
            ops.append_sum(1.0f);
          }
        }
#endif
        ops.append_eltwise(scale, eltwise_relu, alpha, beta);
        attr.set_post_ops(ops);
    }

    for (auto& convAlgorithm : eligibleAlgorithms) {
      // ---- Initialize convolution primitive descriptor -------------
      shared_ptr<convolution_forward::desc> convFwd_desc;
      if (this->bias_term_) {
          if (dilated_conv)
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_bias_md, init_top_md, convolutionStrides, dilation, padding, padding_r,
                  padding_kind::zero));
          else
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_bias_md, init_top_md, convolutionStrides, padding, padding,
                  padding_kind::zero));
      } else {
          if (dilated_conv)
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_top_md, convolutionStrides, dilation, padding, padding_r,
                  padding_kind::zero));
          else
              convFwd_desc.reset(new convolution_forward::desc(
                  propagation, convAlgorithm, init_bottom_md, init_weights_md,
                  init_top_md, convolutionStrides, padding, padding,
                  padding_kind::zero));
      }

      for (subEngineIndex = 0; subEngineIndex < ep.getNumberOfSubEngines();
           subEngineIndex++) {
        try {
#ifndef DISABLE_CONV_SUM_FUSION
            if(this->need_quantize_ || relu || bottom.size() > 1) {
#else
            if(relu) {
#endif
                convFwd_pd.reset(new convolution_forward::primitive_desc(
                *convFwd_desc, attr, ep.getMKLDNNSubEngine(subEngineIndex)));
          } else {
            convFwd_pd.reset(new convolution_forward::primitive_desc(
                *convFwd_desc, ep.getMKLDNNSubEngine(subEngineIndex)));
          }

        } catch (...) {
            continue;
        }
        
        break;
      }
      if (convFwd_pd) break;
    }

    CHECK(convFwd_pd);
    engine cpu_engine = CpuEngine::Instance().get_engine();

    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(convFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(convFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(new MemPD(convFwd_pd->weights_primitive_desc()));

    // ---- Log prv memory primitive descriptors -------------
    info_mem_pd<Dtype>(prv_fwd_bottom_data_memory_pd, "conv_src:" + this->layer_param_.name());
    info_mem_pd<Dtype>(prv_fwd_top_data_memory_pd, "conv_dst:" + this->layer_param_.name());
    
    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format weights_mfmt = (g!= 1) ? memory::format::goihw : memory::format::oihw;

    // TODO: There should not be a problem to use this for Backward as well
    
    shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_bias_data_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    bool bottom_is_float = false;
    if (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        bottom_is_float = blob_prv_mkldnn_mem_descr->get_float();
    }
    if (this->need_quantize_){
      if(this->is_float_ || bottom_is_float){
        std::vector<float> scale_bottom;
        scale_bottom.push_back(this->scale_in_[0]);
        fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this, true, scale_bottom));
      } else{
        std::vector<int> fl_bottom;
        fl_bottom.push_back(this->fl_layer_in_[0]);
        fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this, fl_bottom));
      }       
    } else if(bottom_is_float){
      fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this, false));
    } else{
      fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this));
    }
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    if (this->need_quantize_){
      if(this->is_float_ || bottom_is_float){
        std::vector<float> scale_top;
        scale_top.push_back(this->scale_out_[0]);
        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this, true, scale_top, is_sum));
      } else{
        std::vector<int> fl_top;
        fl_top.push_back(this->fl_layer_out_[0]);
        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this, fl_top, is_sum));
      }
    } else if(bottom_is_float){ 
      fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this, false));
    } else{
      fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this));
    }
    fwd_top_data->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    if (this->need_quantize_){
      int count = 1; //single channel
      if(this->is_float_ || bottom_is_float){
        if(this->scale_params_.size() > 1){
            count = oc;  //multi channel
        }
        std::vector<float> scale_weight(count);
        #pragma omp parallel for if (count > 1)
        for(int i=0; i<count; i++){
          scale_weight[i] = this->scale_params_[i];
        }
        fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this, true, scale_weight));
      } else{
        if(this->fl_params_.size() > 1){
            count = oc;  //multi channel
        }
        std::vector<int> fl_weight(count);
        #pragma omp parallel for if (count > 1)
        for(int i=0; i<count; i++){
          fl_weight[i] = this->fl_params_[i];
        }
        fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this, fl_weight));
      }
    } else if(bottom_is_float){
      fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this, false));
    } else{
      fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this));
    }
    fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
    fwd_weights_data_primitive = fwd_weights_data->create_input(true);

    if (this->bias_term_) {
        shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(new MemPD(convFwd_pd->bias_primitive_desc()));
        if (this->need_quantize_){
          int count = 1;  //single channel
          if(this->is_float_ || bottom_is_float){
            if(this->scale_params_.size() > 1){
                count = oc;  //multi channel
            }
            std::vector<float> scale_bias(count);
            #pragma omp parallel for if (count > 1)
            for(int i=0; i<count; i++){
              scale_bias[i] = this->scale_in_[0] * this->scale_params_[i];
            }
            fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this, true, scale_bias));
          } else{
            if(this->fl_params_.size() > 1){
                count = oc;  //multi channel
            }
            std::vector<int> fl_bias(count);
            #pragma omp parallel for if (count > 1)
            for(int i=0; i<count; i++){
              fl_bias[i] = this->fl_layer_in_[0] + this->fl_params_[i];
            }
            fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this, fl_bias));
          }
        } else if(bottom_is_float){
          fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this, false));
        } else{
          fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this));
        }
        fwd_bias_data->name = "fwd_bias_data     @ " + this->layer_param_.name();
        fwd_bias_data_primitive = fwd_bias_data->create_input(true);
        convFwd.reset(new convolution_forward(*convFwd_pd
                        , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                        , *fwd_bias_data_primitive, *fwd_top_data_memory));
        //fwd_bias_data->set_mkldnn_primitive(convFwd);   //Wrong passed primitive! (For sure!)
        MKLDNNPrimitive<Dtype> fwd_bias_data_primitive_transfer(fwd_bias_data_primitive);
        fwd_bias_data->set_mkldnn_primitive(fwd_bias_data_primitive_transfer);
    } else {
        convFwd.reset(new convolution_forward(*convFwd_pd
                        , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                        , *fwd_top_data_memory));
    }
    //fwd_bottom_data->set_mkldnn_primitive(convFwd);   //Wrong passed primitive! (For sure!)
    MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(convFwd);      //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);

    //fwd_weights_data->set_mkldnn_primitive(convFwd);  //Wrong passed primitive! (For sure!)
    MKLDNNPrimitive<Dtype> fwd_weights_data_primitive_transfer(fwd_weights_data_primitive);
    fwd_weights_data->set_mkldnn_primitive(fwd_weights_data_primitive_transfer);

    // Names are for debugging purposes only.
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNConvolutionLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();

    if( convFwd_pd == NULL || this->reshape)
        InitConvolutionFwd(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    fwd_weights_data->sync_before_read();
    if (this->bias_term_)
        fwd_bias_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    convFwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);
}


template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::InitConvolutionBwd(const vector<Blob<Dtype>*>& top
                                                    , const vector<bool>& propagate_down
                                                    , const vector<Blob<Dtype>*>& bottom)
{
    if (std::is_same<Dtype, double>::value)   NOT_IMPLEMENTED;

    int32_t g  = std::max(this->group_, 1);
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    int32_t ow = this->width_out_;
    int32_t oh = this->height_out_;
    int32_t oc = this->num_output_;

    int32_t kw = this->kernel_w_;
    int32_t kh = this->kernel_h_;

    int32_t sw = this->stride_w_;
    int32_t sh = this->stride_h_;

    int32_t pw = this->pad_w_;
    int32_t ph = this->pad_h_;
    memory::dims convolutionStrides {sh, sw};
    memory::dims padding {ph, pw};
    memory::dims padding_r;
    memory::dims dilation;
    bool dilated_conv = false;
    const int* dilation_data = this->dilation_.cpu_data();
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      dilation.push_back(dilation_data[i] - 1);
      if (dilation_data[i] != 1) dilated_conv = true;
    }
    padding_r.push_back((oh - 1) * sh - ih + ((kh - 1) * (dilation_data[0]) + 1) - ph);
    padding_r.push_back((ow - 1) * sw - iw + ((kw - 1) * (dilation_data[1]) + 1) - pw);

    // ---- Initialize memory descriptors (fromat = any) to create convolution descriptor -------------
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_any = memory::format::any;

    memory::dims bottom_tz = {n, ic, ih, iw};
    memory::dims bias_tz = {oc};
    memory::dims top_tz = {n, oc, oh, ow};
    memory::dims weights_tz = ( g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};

    // ---- Memory descriptors for initializing of convolution primitive descriptor -------------
    memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt_any);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt_any);
    memory::desc init_top_md({top_tz}, mpcsn, mfmt_any);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt_any);

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;

    auto eligibleAlgorithms = {conv_algorithm, algorithm::convolution_direct};
    convBwdData_pd = NULL;
    convBwdWeights_pd = NULL;
    for (auto &convAlgorithm : eligibleAlgorithms) {
        // ---- Initialize convolution primitive descriptor -------------
        shared_ptr<convolution_backward_data::desc> convBwdData_desc;
        shared_ptr<convolution_backward_weights::desc> convBwdWeights_desc;
        if (this->bias_term_) {
            if (dilated_conv)
                convBwdWeights_desc.reset(new convolution_backward_weights::desc(convAlgorithm
                            , init_bottom_md, init_weights_md, init_bias_md, init_top_md
                            , convolutionStrides, dilation, padding, padding_r, padding_kind::zero));
            else
                convBwdWeights_desc.reset(new convolution_backward_weights::desc(convAlgorithm
                            , init_bottom_md, init_weights_md, init_bias_md, init_top_md
                            , convolutionStrides, padding, padding, padding_kind::zero));
        } else {
            if (dilated_conv)
                convBwdWeights_desc.reset(new convolution_backward_weights::desc(convAlgorithm
                            , init_bottom_md, init_weights_md, init_top_md
                            , convolutionStrides, dilation, padding, padding_r, padding_kind::zero));
            else
                convBwdWeights_desc.reset(new convolution_backward_weights::desc(convAlgorithm
                            , init_bottom_md, init_weights_md, init_top_md
                            , convolutionStrides, padding, padding, padding_kind::zero));
        }
       
        if (dilated_conv)
            convBwdData_desc.reset(new convolution_backward_data::desc(convAlgorithm
                            , init_bottom_md, init_weights_md, init_top_md
                            , convolutionStrides, dilation, padding, padding_r, padding_kind::zero));
        else
            convBwdData_desc.reset(new convolution_backward_data::desc(convAlgorithm
                            , init_bottom_md, init_weights_md, init_top_md
                            , convolutionStrides, padding, padding, padding_kind::zero));

        for(subEngineIndex=0; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
            try {
                convBwdData_pd.reset(new convolution_backward_data::primitive_desc(*convBwdData_desc,
                                          ep.getMKLDNNSubEngine(subEngineIndex), *convFwd_pd));

                convBwdWeights_pd.reset(new convolution_backward_weights::primitive_desc(*convBwdWeights_desc,
                                          ep.getMKLDNNSubEngine(subEngineIndex), *convFwd_pd));
            }
            catch(...) {
                continue;
            }
            break;
        }
        if (convBwdData_pd && convBwdWeights_pd)
            break;
    }

    CHECK(convBwdData_pd);
    CHECK(convBwdWeights_pd);
    engine cpu_engine = CpuEngine::Instance().get_engine();

    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(new MemPD(convBwdData_pd->diff_src_primitive_desc()));
    shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(new MemPD(convBwdData_pd->diff_dst_primitive_desc()));
    shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(new MemPD(convBwdData_pd->weights_primitive_desc()));

    shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(new MemPD(convBwdWeights_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(new MemPD(convBwdWeights_pd->diff_dst_primitive_desc()));
    shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(new MemPD(convBwdWeights_pd->diff_weights_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format weights_mfmt = ( g!= 1) ? memory::format::goihw : memory::format::oihw;

    // ???!!! can we use usr memory primitive descrittors for backward??
    shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_bias_data_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));


    // ---  init primitive and prv_memory descriptors ----------------------
    bwdd_bottom_diff.reset(new MKLDNNDiff<Dtype>(usr_bottom_data_memory_pd, prv_bwdd_bottom_diff_memory_pd, bottom[0], this));
    bwdd_bottom_diff ->name = "bwdd_bottom_diff   @ " + this->layer_param_.name();
    bwdd_bottom_diff_memory = bwdd_bottom_diff->create_output_memory();
    bwdw_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_bwdw_bottom_data_memory_pd, bottom[0], this));
    bwdw_bottom_data ->name = "bwdw_bottom_data   @ " + this->layer_param_.name();
    bwdw_bottom_data_primitive = bwdw_bottom_data->create_input(false);

    bwdd_top_diff.reset(new MKLDNNDiff<Dtype>(usr_top_data_memory_pd, prv_bwdd_top_diff_memory_pd, top[0], this));
    bwdd_top_diff    ->name = "bwdd_top_diff      @ " + this->layer_param_.name();
    bwdd_top_diff_primitive = bwdd_top_diff->create_input(false);
    bwdw_top_diff.reset(new MKLDNNDiff<Dtype>(usr_top_data_memory_pd, prv_bwdw_top_diff_memory_pd, top[0], this));
    bwdw_top_diff    ->name = "bwdw_top_diff      @ " + this->layer_param_.name();
    bwdw_top_diff_primitive = bwdw_top_diff->create_input(false);

    bwdd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_bwdd_weights_data_memory_pd, this->blobs_[0].get(), this));
    bwdd_weights_data->name = "bwdd_weights_data  @ " + this->layer_param_.name();
    bwdd_weights_data_primitive = bwdd_weights_data->create_input(false);
    bwdw_weights_diff.reset(new MKLDNNDiff<Dtype>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_pd, this->blobs_[0].get(), this));
    bwdw_weights_diff->name = "bwdw_weights_diff  @ " + this->layer_param_.name();
    bwdw_weights_diff_memory = bwdw_weights_diff->create_output_memory();

    if (Caffe::iter_size() > 1) {
      // support for (iter_size > 1) weights diff requires additional buffer
      shared_ptr<MemPD> prv_bwdw_weights_diff_memory_iter_pd(new MemPD(convBwdWeights_pd->diff_weights_primitive_desc()));
      bwdw_weights_diff_iter.reset(new MKLDNNDiff<Dtype>(usr_weights_data_memory_pd, prv_bwdw_weights_diff_memory_iter_pd, bwdw_weights_diff_iter_blob.get(), this));
      bwdw_weights_diff_memory_iter = bwdw_weights_diff_iter->create_output_memory();
    }

    if (this->bias_term_) {
        shared_ptr<MemPD> prv_bwdw_bias_diff_memory_pd(new MemPD(convBwdWeights_pd->diff_bias_primitive_desc()));
        bwdw_bias_diff.reset(new MKLDNNDiff<Dtype>(usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_pd, this->blobs_[1].get(), this));
        bwdw_bias_diff->name = "bwdw_bias_diff     @ " + this->layer_param_.name();
        bwdw_bias_diff_memory = bwdw_bias_diff->create_output_memory();

        if (Caffe::iter_size() > 1) {
          // support for (iter_size > 1) bias diff requires additional buffer
          shared_ptr<MemPD> prv_bwdw_bias_diff_memory_iter_pd(new MemPD(convBwdWeights_pd->diff_bias_primitive_desc()));
          bwdw_bias_diff_iter.reset(new MKLDNNDiff<Dtype>(usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_iter_pd, bwdw_bias_diff_iter_blob.get(), this));
          bwdw_bias_diff_memory_iter = bwdw_bias_diff_iter->create_output_memory();
          convBwdWeights.reset(new convolution_backward_weights(*convBwdWeights_pd
                        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
                        , *bwdw_weights_diff_memory_iter, *bwdw_bias_diff_memory_iter));
        } else {
          convBwdWeights.reset(new convolution_backward_weights(*convBwdWeights_pd
                        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
                        , *bwdw_weights_diff_memory, *bwdw_bias_diff_memory));
        }

        //bwdw_bias_diff->set_mkldnn_primitive(convBwdWeights);   //Wrong passed primitive! (For sure!)
        MKLDNNPrimitive<Dtype> bwdw_bias_diff_memory_transfer(bwdw_bias_diff_memory);
        bwdw_bias_diff->set_mkldnn_primitive(bwdw_bias_diff_memory_transfer);

        if (Caffe::iter_size() > 1) {
          // support for (iter_size > 1) bias diff requires additional buffer
          MKLDNNPrimitive<Dtype> bwdw_bias_diff_memory_iter_transfer(bwdw_bias_diff_memory_iter);
          bwdw_bias_diff_iter->set_mkldnn_primitive(bwdw_bias_diff_memory_iter_transfer);
        }
    } else {
        if (Caffe::iter_size() > 1) {
          // if (iter_size > 1) then weights diff should be accumulated across iterations
          convBwdWeights.reset(new convolution_backward_weights(*convBwdWeights_pd
                        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
                        , *bwdw_weights_diff_memory_iter));
        } else {
          convBwdWeights.reset(new convolution_backward_weights(*convBwdWeights_pd
                        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
                        , *bwdw_weights_diff_memory));
        }
    }

    convBwdData.reset(new convolution_backward_data(*convBwdData_pd
                    , *bwdd_top_diff_primitive, *bwdd_weights_data_primitive
                    , *bwdd_bottom_diff_memory));

    //bwdd_bottom_diff->set_mkldnn_primitive(convBwdData);      //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwdd_bottom_diff_memory_transfer(bwdd_bottom_diff_memory);
    bwdd_bottom_diff->set_mkldnn_primitive(bwdd_bottom_diff_memory_transfer);

    //bwdd_top_diff->set_mkldnn_primitive(convBwdData);         //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwdd_top_diff_primitive_transfer(bwdd_top_diff_primitive);
    bwdd_top_diff->set_mkldnn_primitive(bwdd_top_diff_primitive_transfer);

    //bwdd_weights_data->set_mkldnn_primitive(convBwdData);     //Wrong passed primitive! (For sure!)
    MKLDNNPrimitive<Dtype> bwdd_weights_data_primitive_transfer(bwdd_weights_data_primitive);
    bwdd_weights_data->set_mkldnn_primitive(bwdd_weights_data_primitive_transfer);

    //bwdw_bottom_data->set_mkldnn_primitive(convBwdWeights);   //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwdw_bottom_data_primitive_transfer(bwdw_bottom_data_primitive);
    bwdw_bottom_data->set_mkldnn_primitive(bwdw_bottom_data_primitive_transfer);

    //bwdw_top_diff->set_mkldnn_primitive(convBwdWeights);      //Wrong passed primitive! (For sure!)
    MKLDNNPrimitive<Dtype> bwdw_top_diff_primitive_transfer(bwdw_top_diff_primitive);
    bwdw_top_diff->set_mkldnn_primitive(bwdw_top_diff_primitive_transfer);

    //bwdw_weights_diff->set_mkldnn_primitive(convBwdWeights);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwdw_weights_diff_memory_transfer(bwdw_weights_diff_memory);
    bwdw_weights_diff->set_mkldnn_primitive(bwdw_weights_diff_memory_transfer);

    if (Caffe::iter_size() > 1) {
      // support for (iter_size > 1) weights diff requires additional buffer
      MKLDNNPrimitive<Dtype> bwdw_weights_diff_memory_iter_transfer(bwdw_weights_diff_memory_iter);
      bwdw_weights_diff_iter->set_mkldnn_primitive(bwdw_weights_diff_memory_iter_transfer);
    }

    // Names are for debugging purposes only.
}


template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNConvolutionLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
    bool top_diff_is_prv = (const_cast<Dtype*>(top[0]->prv_diff()) != NULL);

    if( convBwdData_pd == NULL || this->reshape)
        InitConvolutionBwd(top, propagate_down, bottom);
    if (propagate_down[0]) {
        // making reorders if needed.
        bwdd_top_diff->sync_before_read();
        bwdd_weights_data->sync_before_read();
        bwdd_bottom_diff->sync_before_write();

        PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKLDNN_NAME("BW"));
        PERFORMANCE_MEASUREMENT_BEGIN();
#ifdef DEBUG
        if (bottom[0]->prv_data() != NULL)
        {
            LOG(INFO) << "Debug: Bottom prv data: " << *bottom[0]->prv_data();
        }
        else
        {
            LOG(INFO) << "Debug: Bottom prv data is NULL!";
            //LOG(INFO) << "Debug: Bottom cpu data: " << *bottom[0]->cpu_data();
        }

        if (top[0]->prv_diff() != NULL)
        {
            LOG(INFO) << "Debug: Top prv diff: " << *top[0]->prv_diff();
        }
        else
        {
            LOG(INFO) << "Debug: Top prv diff is NULL!";
            LOG(INFO) << "Debug: Top cpu diff: " << *top[0]->cpu_diff();
        }

        if (this->blobs_[0]->prv_data() != NULL)
        {
            LOG(INFO) << "Debug: Weights prv data from blobs_[0]: " << *this->blobs_[0]->prv_data();
        }
        else
        {
            LOG(INFO) << "Debug: Weights prv data is NULL!";
            LOG(INFO) << "Debug: Weights cpu data: " << *this->blobs_[0]->cpu_data();
        }
        //Before submit, so get_prv_ptr() always has the value
        LOG(INFO) << "Debug: Weights prv data from get_prv_ptr: " << *bwdd_weights_data->get_prv_ptr();
#endif
        convBwdData.submit();
#ifdef DEBUG
        if (bottom[0]->prv_diff() != NULL)
        {
            LOG(INFO) << "Debug: Bottom prv diff: " << *bottom[0]->prv_diff();
        }
        else
        {
            LOG(INFO) << "Debug: Bottom prv diff is NULL!";
            LOG(INFO) << "Debug: Bottom cpu diff: " << *bottom[0]->cpu_diff();
        }
#endif
        PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);
    }
    if (this->param_propagate_down(0)) {
        // We have to sync top diff to cpu explicitly. This is used to make
        // bwdw_top_diff->sync_before_read() have chance to get coverted data as
        // bwdd_top_diff->sync_before_read() have updated top diff's prv_data
        // to self. This issue only happens when MKLDNN conv layer is followed
        // by a CAFFE layer and conversion is needed.
        if (!top_diff_is_prv && propagate_down[0])
          top[0]->mutable_cpu_diff();
        // making reorders if needed.
        bwdw_top_diff->sync_before_read();
        bwdw_bottom_data->sync_before_read();
        // update top that head at prv
        bwdw_weights_diff->sync_before_write();
        if (this->param_propagate_down(1)) {
            CHECK(bwdw_bias_diff);
            bwdw_bias_diff->sync_before_write();
        }
        PERFORMANCE_EVENT_ID_INIT(perf_id_bw_weights_,
          PERFORMANCE_MKLDNN_NAME_DETAILED("BW", "_weights"));
        PERFORMANCE_MEASUREMENT_BEGIN();
        convBwdWeights.submit();
        PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_weights_);

        if (Caffe::iter_size() > 1) {
          // if (iter_size > 1) then weights diff should be accumulated across iterations
          if (this->blobs_[0]->prv_diff() != NULL) {
            caffe_axpy(this->blobs_[0]->prv_diff_count(), Dtype(1),
              (Dtype*)(bwdw_weights_diff_memory_iter->get_data_handle()),
              this->blobs_[0]->mutable_prv_diff());
          } else {
            caffe_axpy(this->blobs_[0]->count(), Dtype(1),
              (Dtype*)(bwdw_weights_diff_memory_iter->get_data_handle()),
              this->blobs_[0]->mutable_cpu_diff());
          }
        }

        if (this->param_propagate_down(1)) {
          if (Caffe::iter_size() > 1) {
            // if (iter_size > 1) then bias diff should be accumulated across iterations
            if (this->blobs_[1]->prv_diff() != NULL) {
              caffe_axpy(this->blobs_[1]->prv_diff_count(), Dtype(1),
                (Dtype*)(bwdw_bias_diff_memory_iter->get_data_handle()),
                this->blobs_[1]->mutable_prv_diff());
            } else {
              caffe_axpy(this->blobs_[1]->count(), Dtype(1),
                (Dtype*)(bwdw_bias_diff_memory_iter->get_data_handle()),
                this->blobs_[1]->mutable_cpu_diff());
            }
          }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNConvolutionLayer);
#else

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNConvolutionLayer);

}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
