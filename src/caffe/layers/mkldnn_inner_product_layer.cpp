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
#include "mkl_service.h"

#if 0
#include "mkldnn_types.h"

using namespace mkldnn;
#endif

// TODO: Add transposed weights support

namespace caffe {
template <typename Dtype>
MKLDNNInnerProductLayer<Dtype>::MKLDNNInnerProductLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(), InnerProductLayer<Dtype>(param)
            , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
            , ipFwd_pd(NULL), output_memory(NULL)
            , input_primitive(NULL), weights_primitive(NULL), bias_primitive(NULL)
            , w_(0), h_(0)
{
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
void MKLDNNInnerProductLayer<Dtype>::InitInnerProduct(const vector<Blob<Dtype>*>& bottom
                                                    , const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    int32_t n  = this->M_;
    int32_t w = this->w_;
    int32_t h = this->h_;
    int32_t oc = this->N_;
    int32_t ic = this->K_/h_/w_;
    bool has_spatial = h > 1 || w > 1;

    // Initialize memory descriptors (fromat = any) to create inner_product descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;

    memory::dims input_tz = (has_spatial) ? memory::dims{n, ic, h, w} : memory::dims{n, ic};
    memory::dims output_tz = {n, oc};
    memory::dims weights_tz = (has_spatial) ? memory::dims {oc, ic, h, w} : memory::dims{oc, ic};
    memory::dims bias_tz = {oc};

    memory::desc init_input_md({input_tz}, mpcsn, mfmt);
    memory::desc init_output_md({ output_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);

    // Initialize inner_product primitive descriptor
    shared_ptr<inner_product_forward::desc> ipFwd_desc;
    if (this->bias_term_) {
        ipFwd_desc.reset(new inner_product_forward::desc(propagation, init_input_md, init_weights_md
                                                ,init_bias_md, init_output_md));
    } else {
        ipFwd_desc.reset(new inner_product_forward::desc(propagation, init_input_md, init_weights_md
                                                , init_output_md));
    }

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine_sequence();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        ipFwd_pd.reset(new inner_product_forward::primitive_desc(*ipFwd_desc,
                ep.getSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(ipFwd_pd);
    engine engine = ep.getSubEngine(subEngineIndex);

    // Create priv memory primitive descriptors stored as class members
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_input_memory_pd(new MemPD(ipFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_bias_memory_pd(new MemPD(ipFwd_pd->bias_primitive_desc()));
    shared_ptr<MemPD> prv_output_memory_pd(new MemPD(ipFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_weights_memory_pd(new MemPD(ipFwd_pd->weights_primitive_desc()));

    // Create usr memory primitive descriptors stored as class members
    memory::format input_mfmt = has_spatial ? memory::format::nchw : memory::format::nc;
    shared_ptr<MemPD> usr_input_memory_pd(new MemPD({{input_tz}, mpcsn, input_mfmt}, engine));
    shared_ptr<MemPD> usr_bias_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, engine));
    shared_ptr<MemPD> usr_output_memory_pd(new MemPD({{output_tz}, mpcsn, memory::format::nc}, engine));
    memory::format weights_mfmt = has_spatial ? memory::format::oihw : memory::format::oi;
    shared_ptr<MemPD> usr_weights_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_input_memory_pd, prv_input_memory_pd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_output_memory_pd, prv_output_memory_pd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_memory_pd, prv_weights_memory_pd, this->blobs_[0].get(), this));
    weights_primitive = fwd_weights_data->create_input(false);

    if (this->bias_term_) {
        fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_memory_pd, prv_bias_memory_pd, this->blobs_[1].get(), this));
        bias_primitive = fwd_bias_data->create_input(false);
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
                            , *input_primitive, *weights_primitive
                            , *bias_primitive, *output_memory));
    } else {
        ipFwd.reset(new inner_product_forward(*ipFwd_pd
                            , *input_primitive, *weights_primitive
                            , *output_memory));
    }
    fwd_bottom_data->set_mkldnn_primitive(ipFwd);
    fwd_top_data->set_mkldnn_primitive(ipFwd);
    fwd_weights_data->set_mkldnn_primitive(ipFwd);
    fwd_bias_data->set_mkldnn_primitive(ipFwd);

    // Names are for debugging purposes only.
    fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
    fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNInnerProductLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if( ipFwd_pd == NULL)
        InitInnerProduct(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read(false);
    fwd_weights_data->sync_before_read(true);
    fwd_bias_data->sync_before_read(true);
    // update top that head at prv
    fwd_top_data->sync_before_write();

    ipFwd.submit();
}

template <typename Dtype>
void MKLDNNInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
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
