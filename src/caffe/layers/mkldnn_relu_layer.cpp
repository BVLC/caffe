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
void MKLDNNReLULayer<Dtype>::InitReLU(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, output_md;
    shared_ptr<memory::primitive_desc> usr_mpd(NULL), prv_mpd(NULL);
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        input_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_mpd = mem_descr->usr_memory_pd();
        prv_mpd = mem_descr->prv_memory_pd();
    } else {
        input_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
    }
    output_md = input_md;

    // ---- Initialize relu primitive descriptor -------------
    relu_forward::desc reluFwd_desc(propagation, *input_md, negative_slope);

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine_sequence();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        reluFwd_pd.reset(new relu_forward::primitive_desc(reluFwd_desc,
                ep.getSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(reluFwd_pd);
    engine engine = ep.getSubEngine(subEngineIndex);

    // ---- Initialize remaining memory descriptors -------------
    if (!bottom_data_is_prv)
      usr_mpd.reset(new memory::primitive_desc(*input_md, engine));
    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    reluFwd.reset(new relu_forward(*reluFwd_pd, *input_primitive, *output_memory));
    fwd_bottom_data->set_mkldnn_primitive(reluFwd);
    fwd_top_data->set_mkldnn_primitive(reluFwd);
}


template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if( reluFwd_pd == NULL)
        InitReLU(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read(false);
    // update top that head at prv
    fwd_top_data->sync_before_write();

    reluFwd.submit();
}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }

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
