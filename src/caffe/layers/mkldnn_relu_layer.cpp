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
#include <vector>

#include "caffe/layers/mkldnn_layers.hpp"

namespace caffe {

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

    NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                    ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Reshape: " << this->layer_param_.name();

    NeuronLayer<Dtype>::Reshape(bottom, top);

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::InitReLUFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);
    bool inplace = (bottom[0] == top[0]);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> bottom_data_md, top_data_md;
    shared_ptr<memory::primitive_desc> usr_data_mpd(NULL), prv_data_mpd(NULL);
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        bottom_data_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_data_mpd = mem_descr->usr_memory_pd();
        prv_data_mpd = mem_descr->prv_memory_pd();
    } else {
        bottom_data_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
        usr_data_mpd.reset(new memory::primitive_desc(*bottom_data_md, cpu_engine));
    }
    top_data_md = bottom_data_md;

    // ---- Initialize relu primitive descriptor -------------
    relu_forward::desc reluFwd_desc(propagation, *bottom_data_md, negative_slope);

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        reluFwd_pd.reset(new relu_forward::primitive_desc(reluFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }
    CHECK(reluFwd_pd);

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_data_mpd, prv_data_mpd, bottom[0], this));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_data_mpd, prv_data_mpd, top[0], this));
    fwd_top_data->name = "fwd_top_data   @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory(inplace);

    reluFwd.reset(new relu_forward(*reluFwd_pd, *fwd_bottom_data_primitive, *fwd_top_data_memory));
    fwd_bottom_data->set_mkldnn_primitive(reluFwd);
    fwd_top_data->set_mkldnn_primitive(reluFwd);

}


template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    bool inplace = (bottom[0] == top[0]);
    if( reluFwd_pd == NULL)
        InitReLUFwd(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write(inplace);

    reluFwd.submit();
}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::InitReLUBwd(const vector<Blob<Dtype>*>& top
                                         ,const vector<bool>& propagate_down
                                         ,const vector<Blob<Dtype>*>& bottom)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;

    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    bool top_diff_is_prv = top[0]->prv_diff() != NULL;
    bool inplace = (bottom[0] == top[0]);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;

    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> bottom_diff_md;
    shared_ptr<memory::desc> top_diff_md;
    shared_ptr<memory::desc> top_data_md;
    
    shared_ptr<memory::primitive_desc> usr_diff_mpd;
    shared_ptr<memory::primitive_desc> prv_diff_mpd;
    
    if (top_diff_is_prv) {
      shared_ptr<MKLDNNMemoryDescriptor<Dtype, /* is_diff */ true> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype, /* is_diff */ true>(top[0]);
      top_diff_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
      usr_diff_mpd = mem_descr->usr_memory_pd();
      prv_diff_mpd = mem_descr->prv_memory_pd();
    } else {
      top_diff_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
      usr_diff_mpd.reset(new memory::primitive_desc(*top_diff_md, cpu_engine));
    }
    
    top_data_md = top_diff_md;
    bottom_diff_md = top_diff_md;

    // ---- Initialize relu primitive descriptor -------------
    relu_backward::desc reluBwd_desc(*top_diff_md, *top_data_md, negative_slope);

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        reluBwd_pd.reset(new relu_backward::primitive_desc(reluBwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex), *reluFwd_pd));
      }
      catch(...) {
        continue;
      }
      break;
    }
    CHECK(reluBwd_pd);

    // ---  init primitive and prv_memory descriptors ----------------------
    bwd_top_diff.reset(new MKLDNNDiff<Dtype>(usr_diff_mpd, prv_diff_mpd, top[0], this));
    bwd_top_diff->name = "bwd_top_diff_data   @ " + this->layer_param_.name();
    bwd_top_diff_primitive = bwd_top_diff->create_input(/* set_prv_ptr */ false);

    bwd_bottom_diff.reset(new MKLDNNDiff<Dtype>(usr_diff_mpd, prv_diff_mpd, bottom[0], this));
    bwd_bottom_diff->name = "bwd_bottom_diff_data   @ " + this->layer_param_.name();
    bwd_bottom_diff_memory = bwd_bottom_diff->create_output_memory(inplace);

    reluBwd.reset(new relu_backward(*reluBwd_pd, *fwd_bottom_data_primitive, *bwd_top_diff_primitive, *bwd_bottom_diff_memory));
    bwd_top_diff->set_mkldnn_primitive(reluBwd);
    bwd_bottom_diff->set_mkldnn_primitive(reluBwd);
}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                          , const vector<bool>& propagate_down
                                          , const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
    bool inplace = (bottom[0] == top[0]);
    if (!propagate_down[0]) {
        return;
    }
    if (reluBwd_pd == NULL) {
        InitReLUBwd(top, propagate_down, bottom);
    }
    
    bwd_top_diff->sync_before_read();
    bwd_bottom_diff->sync_before_write(inplace);

    reluBwd.submit();
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNReLULayer);
#else
template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{ NOT_IMPLEMENTED; }

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }
#endif

INSTANTIATE_CLASS(MKLDNNReLULayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
