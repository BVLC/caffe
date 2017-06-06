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

#if 0
#include "mkldnn_types.h"

using namespace mkldnn;
#endif

// TODO: Add transposed weights support

namespace caffe {
template <typename Dtype>
MKLDNNInnerProductLayer<Dtype>::MKLDNNInnerProductLayer(
            const LayerParameter& param) :
            MKLDNNLayer<Dtype>(),
            InnerProductLayer<Dtype>(param),
            fwd_bottom_data(NULL),
            fwd_top_data(NULL),
            fwd_weights_data(NULL),
            fwd_bias_data(NULL),
            bwdd_weights_data(NULL),
            bwdw_bottom_data(NULL),
            bwdd_bottom_diff(NULL),
            bwdd_top_diff(NULL),
            bwdw_top_diff(NULL),
            bwdw_weights_diff(NULL),
            bwdw_bias_diff(NULL),
            ipFwd_pd(NULL),
            ipBwdData_pd(NULL),
            ipBwdWeights_pd(NULL),
            fwd_top_data_memory(NULL),
            bwdd_bottom_diff_memory(NULL),
            bwdw_weights_diff_memory(NULL),
            bwdw_bias_diff_memory(NULL),
            fwd_bottom_data_primitive(NULL),
            fwd_weights_data_primitive(NULL),
            fwd_bias_data_primitive(NULL),
            bwdd_top_diff_primitive(NULL),
            bwdd_weights_data_primitive(NULL),
            bwdw_top_diff_primitive(NULL),
            bwdw_bottom_data_primitive(NULL),
            w_(0),
            h_(0)
{
  PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
  PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
  PERFORMANCE_EVENT_ID_RESET(perf_id_bw_weights_);
}

template <typename Dtype>
MKLDNNInnerProductLayer<Dtype>::~MKLDNNInnerProductLayer()
{
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();
    InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::Reshape: " << this->layer_param_.name();
    InnerProductLayer<Dtype>::Reshape(bottom, top);

    this->w_ = bottom[0]->width();
    this->h_ = bottom[0]->height();
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::InitInnerProductFwd(const vector<Blob<Dtype>*>& bottom
                                                    , const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    int32_t n  = this->M_;
    int32_t w = this->w_;
    int32_t h = this->h_;
    int32_t oc = this->N_;
    int32_t ic = this->K_/h_/w_;
    bool has_spatial = (bottom[0]->shape().size() != 2);

    // Initialize memory descriptors (fromat = any) to create inner_product descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;

    memory::dims bottom_tz = (has_spatial) ? memory::dims{n, ic, h, w} : memory::dims{n, ic};
    memory::dims top_tz = {n, oc};
    memory::dims weights_tz = (has_spatial) ? memory::dims {oc, ic, h, w} : memory::dims{oc, ic};
    memory::dims bias_tz = {oc};

#ifdef DEBUG
    LOG(INFO) << "has_spatial flag value: " << has_spatial;
    if (has_spatial)
    {
        LOG(INFO) << "Dimension of bottom for MKLDNN: " << n << " " << ic << " " << h << " " << w;
        LOG(INFO) << "Dimension of weights for MKLDNN: " << oc << " " << ic << " " << h << " " << w;
    }
    else
    {
        LOG(INFO) << "Dimension of bottom for MKLDNN: " << n << " " << ic;
        LOG(INFO) << "Dimension of weights for MKLDNN: " << oc << " " << ic;
    }
#endif

    memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt);
    memory::desc init_top_md({top_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);

    // Initialize inner_product primitive descriptor
    shared_ptr<inner_product_forward::desc> ipFwd_desc;
 
    if (this->bias_term_) {
        ipFwd_desc.reset(new inner_product_forward::desc(propagation, init_bottom_md, init_weights_md
                                                ,init_bias_md, init_top_md));
     } else {
        ipFwd_desc.reset(new inner_product_forward::desc(propagation, init_bottom_md, init_weights_md
                                                , init_top_md));
    }

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        ipFwd_pd.reset(new inner_product_forward::primitive_desc(*ipFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(ipFwd_pd);

    // Create priv memory primitive descriptors stored as class members
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(ipFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(ipFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_weights_data_memory_pd(new MemPD(ipFwd_pd->weights_primitive_desc()));
 
    // Create usr memory primitive descriptors stored as class members
    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
    shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, input_mfmt}, cpu_engine));
    shared_ptr<MemPD> usr_bias_data_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, memory::format::nc}, cpu_engine));
    memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
    shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));
#ifdef DEBUG
    LOG(INFO) << "Memory format of usr_bottom_data_memory_pd: " << input_mfmt;
    LOG(INFO) << "Memory format of usr_weights_data_memory_pd: " << weights_mfmt;
#endif

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_bottom_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this));
    fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this));
    fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_data_memory_pd, prv_fwd_weights_data_memory_pd, this->blobs_[0].get(), this));
    fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
    fwd_weights_data_primitive = fwd_weights_data->create_input(true);

    if (this->bias_term_) {
        shared_ptr<MemPD> prv_fwd_bias_data_memory_pd(new MemPD(ipFwd_pd->bias_primitive_desc()));
        fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_data_memory_pd, prv_fwd_bias_data_memory_pd, this->blobs_[1].get(), this));
        fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();
        fwd_bias_data_primitive = fwd_bias_data->create_input(true);
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
                            , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                            , *fwd_bias_data_primitive, *fwd_top_data_memory));
    } else {
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
                            , *fwd_bottom_data_primitive, *fwd_weights_data_primitive
                            , *fwd_top_data_memory));
    }
    
    //Because the inputs of inner product layer always come from user memory, so will not trigger the wrong reorder from extprv to prv
    fwd_bottom_data->set_mkldnn_primitive(ipFwd);     //Wrong passed primitive! (TODO: Checking!)
    //MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    //fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    fwd_top_data->set_mkldnn_primitive(ipFwd);
    
    fwd_weights_data->set_mkldnn_primitive(ipFwd);    //Wrong passed primitive! (TODO: Checking!)
    //MKLDNNPrimitive<Dtype> fwd_weights_data_primitive_transfer(fwd_weights_data_primitive);    
    //fwd_weights_data->set_mkldnn_primitive(fwd_weights_data_primitive_transfer);

    if (this->bias_term_) 
      fwd_bias_data->set_mkldnn_primitive(ipFwd);
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNInnerProductLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#endif

    if( ipFwd_pd == NULL)
        InitInnerProductFwd(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    fwd_weights_data->sync_before_read();
    if (this->bias_term_) 
      fwd_bias_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    ipFwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::InitInnerProductBwd(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;

    int32_t n  = this->M_;
    int32_t w = this->w_;
    int32_t h = this->h_;
    int32_t oc = this->N_;
    int32_t ic = this->K_/h_/w_;
    bool has_spatial = (bottom[0]->shape().size() != 2);

    // Initialize memory descriptors (format = any) to create inner_product descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;

    memory::dims bottom_tz = (has_spatial) ? memory::dims{n, ic, h, w} : memory::dims{n, ic};
    memory::dims top_tz = {n, oc};
    memory::dims weights_tz = (has_spatial) ? memory::dims {oc, ic, h, w} : memory::dims{oc, ic};
    memory::dims bias_tz = {oc};

#ifdef DEBUG
    LOG(INFO) << "has_spatial flag value: " << has_spatial;
    if (has_spatial)
    {
        LOG(INFO) << "Dimension of bottom for MKLDNN: " << n << " " << ic << " " << h << " " << w;
        LOG(INFO) << "Dimension of weights for MKLDNN: " << oc << " " << ic << " " << h << " " << w;
    }
    else
    {
        LOG(INFO) << "Dimension of bottom for MKLDNN: " << n << " " << ic;
        LOG(INFO) << "Dimension of weights for MKLDNN: " << oc << " " << ic;
    }
#endif

    memory::desc init_bottom_md({bottom_tz}, mpcsn, mfmt);
    memory::desc init_top_md({top_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);

    // Initialize inner_product primitive descriptor
    shared_ptr<inner_product_backward_data::desc> ipBwdData_desc;
    shared_ptr<inner_product_backward_weights::desc> ipBwdWeights_desc;
 if (this->bias_term_)
    ipBwdWeights_desc.reset(new inner_product_backward_weights::desc(init_bottom_md, init_weights_md
                        , init_bias_md, init_top_md));
 else
    ipBwdWeights_desc.reset(new inner_product_backward_weights::desc(init_bottom_md, init_weights_md
                        , init_top_md));
    
    ipBwdData_desc.reset(new inner_product_backward_data::desc(init_bottom_md, init_weights_md, init_top_md));

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        ipBwdData_pd.reset(new inner_product_backward_data::primitive_desc(*ipBwdData_desc,
                ep.getMKLDNNSubEngine(subEngineIndex), *ipFwd_pd));

        ipBwdWeights_pd.reset(new inner_product_backward_weights::primitive_desc(*ipBwdWeights_desc,
                ep.getMKLDNNSubEngine(subEngineIndex), *ipFwd_pd));
       }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(ipBwdData_pd);
    CHECK(ipBwdWeights_pd);

    // Create priv memory primitive descriptors stored as class members
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_bwdd_bottom_diff_memory_pd(new MemPD(ipBwdData_pd->diff_src_primitive_desc()));
    shared_ptr<MemPD> prv_bwdd_top_diff_memory_pd(new MemPD(ipBwdData_pd->diff_dst_primitive_desc()));
    shared_ptr<MemPD> prv_bwdd_weights_data_memory_pd(new MemPD(ipBwdData_pd->weights_primitive_desc()));

    shared_ptr<MemPD> prv_bwdw_bottom_data_memory_pd(new MemPD(ipBwdWeights_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_bwdw_top_diff_memory_pd(new MemPD(ipBwdWeights_pd->diff_dst_primitive_desc()));
    shared_ptr<MemPD> prv_bwdw_weights_diff_memory_pd(new MemPD(ipBwdWeights_pd->diff_weights_primitive_desc()));

    // Create usr memory primitive descriptors stored as class members
    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;    
    shared_ptr<MemPD> usr_bottom_data_memory_pd(new MemPD({{bottom_tz}, mpcsn, input_mfmt}, cpu_engine));
    shared_ptr<MemPD> usr_bias_data_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_top_data_memory_pd(new MemPD({{top_tz}, mpcsn, memory::format::nc}, cpu_engine));
    memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
    shared_ptr<MemPD> usr_weights_data_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));
#ifdef DEBUG
    LOG(INFO) << "Memory format of usr_bottom_data_memory_pd: " << input_mfmt;
    LOG(INFO) << "Memory format of usr_weights_data_memory_pd: " << weights_mfmt;
#endif

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

    if (this->bias_term_) {
        shared_ptr<MemPD> prv_bwdw_bias_diff_memory_pd(new MemPD(ipBwdWeights_pd->diff_bias_primitive_desc()));
        bwdw_bias_diff.reset(new MKLDNNDiff<Dtype>(usr_bias_data_memory_pd, prv_bwdw_bias_diff_memory_pd, this->blobs_[1].get(), this));
        bwdw_bias_diff   ->name = "bwdw_bias_diff     @ " + this->layer_param_.name();
        bwdw_bias_diff_memory = bwdw_bias_diff->create_output_memory();

        ipBwdWeights.reset(new inner_product_backward_weights(*ipBwdWeights_pd
                        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
                        , *bwdw_weights_diff_memory, *bwdw_bias_diff_memory));
    } else {
        ipBwdWeights.reset(new inner_product_backward_weights(*ipBwdWeights_pd
                        , *bwdw_bottom_data_primitive, *bwdw_top_diff_primitive
                        , *bwdw_weights_diff_memory));
    }

    ipBwdData.reset(new inner_product_backward_data(*ipBwdData_pd
                    , *bwdd_top_diff_primitive, *bwdd_weights_data_primitive
                    , *bwdd_bottom_diff_memory));

    bwdd_bottom_diff->set_mkldnn_primitive(ipBwdData);
    
    bwdd_top_diff->set_mkldnn_primitive(ipBwdData);           //Wrong passed primitive! (TODO: Checking!)
    //MKLDNNPrimitive<Dtype> bwdd_top_diff_primitive_transfer(bwdd_top_diff_primitive);
    //bwdd_top_diff->set_mkldnn_primitive(bwdd_top_diff_primitive_transfer);

    bwdd_weights_data->set_mkldnn_primitive(ipBwdData);       //Wrong passed primitive! (TODO: Checking!)
    //MKLDNNPrimitive<Dtype> bwdd_weights_data_primitive_transfer(bwdd_weights_data_primitive);
    //bwdd_weights_data->set_mkldnn_primitive(bwdd_weights_data_primitive_transfer);


    bwdw_bottom_data->set_mkldnn_primitive(ipBwdWeights);     //Wrong passed primitive! (TODO: Checking!)
    //MKLDNNPrimitive<Dtype> bwdw_bottom_data_primitive_transfer(bwdw_bottom_data_primitive);
    //bwdw_bottom_data->set_mkldnn_primitive(bwdw_bottom_data_primitive_transfer);

    bwdw_top_diff->set_mkldnn_primitive(ipBwdWeights);        //Wrong passed primitive! (TODO: Checking!)
    //MKLDNNPrimitive<Dtype> bwdw_top_diff_primitive_transfer(bwdw_top_diff_primitive);
    //bwdw_top_diff->set_mkldnn_primitive(bwdw_top_diff_primitive_transfer);

    bwdw_weights_diff->set_mkldnn_primitive(ipBwdWeights);

    if (this->bias_term_)
        bwdw_bias_diff->set_mkldnn_primitive(ipBwdWeights);
}



template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNInnerProductLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#endif

    if( ipBwdData_pd == NULL)
        InitInnerProductBwd(top, propagate_down, bottom);
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
            //Chong: if don't have this LOG print, will cause: this->_cpu_ptr == cpu_ptr crash, without the fix in dropout_layer.cpp
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
        ipBwdData.submit();
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
        ipBwdWeights.submit();
        PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_weights_);
    }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNInnerProductLayer);
#else

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNInnerProductLayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
