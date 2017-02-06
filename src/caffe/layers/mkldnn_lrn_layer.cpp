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
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNLRNLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

    Layer<Dtype>::LayerSetUp(bottom, top);

    size_ = this->layer_param_.lrn_param().local_size();
    CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";

  // Fwd, Bwd primitives and lrn_buffer_ are allocated in  "Lazy"
  // mode, because here we don't know
  // what layout is used by neighbours.
}

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                    ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNLRNLayer<Dtype>::Reshape: " << this->layer_param_.name();
    alpha_ = this->layer_param_.lrn_param().alpha();
    beta_ = this->layer_param_.lrn_param().beta();

    // TODO: k_ is not used now in mkldnn
    k_ = this->layer_param_.lrn_param().k();

    width_ = bottom[0]->width();
    height_ = bottom[0]->height();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();

    CHECK_EQ(4, bottom[0]->num_axes())
            << "Input must have 4 axes, corresponding to (num, channels, height, width)";
    switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        top[0]->Reshape(num_, channels_, height_, width_);
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        top[0]->Reshape(num_, channels_, height_, width_);
        break;
    default:
        LOG(FATAL) << "Unknown normalization region.";
    }
}

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::InitLRN(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    algorithm  lrn_algorithm;
    switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        lrn_algorithm = algorithm::lrn_across_channels;
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        lrn_algorithm = algorithm::lrn_within_channel;
        break;
    default:
        LOG(FATAL) << "Unknown normalization region.";
    }

    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, output_md;
    shared_ptr<memory::primitive_desc> usr_mpd, prv_mpd;
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        input_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_mpd = mem_descr->usr_memory_pd();
        prv_mpd = mem_descr->prv_memory_pd();
    } else {
        input_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
        usr_mpd.reset(new memory::primitive_desc(*input_md, cpu_engine));
    }
    output_md = input_md;

    // ---- Initialize LRN primitive descriptor -------------
    lrn_forward::desc lrnFwd_desc(propagation, lrn_algorithm, *input_md
                            , size_, alpha_, beta_);
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        lrnFwd_pd.reset(new lrn_forward::primitive_desc(lrnFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(lrnFwd_pd);

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    if ( propagation == prop_kind::forward_training ) {
        memory::primitive_desc scratch_mpd(lrnFwd_pd->workspace_primitive_desc());
        scratch_.reset(new memory(scratch_mpd));
        lrnFwd.reset(new lrn_forward(*lrnFwd_pd, *input_primitive, *scratch_, *output_memory));
    } else {
        lrnFwd.reset(new lrn_forward(*lrnFwd_pd, *input_primitive, *output_memory));
    }
    fwd_bottom_data->set_mkldnn_primitive(lrnFwd);
    fwd_top_data->set_mkldnn_primitive(lrnFwd);
}


template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNLRNLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if( lrnFwd_pd == NULL)
        InitLRN(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    lrnFwd.submit();
}

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                        ,const vector<bool>& propagate_down
                                        ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }

#ifdef CPU_ONLY
STUB_GPU(MKLDNNLRNLayer);
#else
template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down
                                        ,const vector<Blob<Dtype>*>& bottom)
{NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLDNNLRNLayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
