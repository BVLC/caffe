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
MKLDNNLRNLayer<Dtype>::MKLDNNLRNLayer(const LayerParameter& param)
        : MKLDNNLayer<Dtype>(param), Layer<Dtype>(param)
        , fwd_top_data(NULL), fwd_bottom_data(NULL)
        , bwd_top_diff(NULL), bwd_bottom_diff(NULL)
        , lrnFwd_pd(NULL), lrnBwd_pd(NULL)
		, fwd_top_data_memory(NULL), bwd_bottom_diff_memory(NULL)
		, scratch_memory(NULL)
		, fwd_bottom_data_primitive(NULL), bwd_top_diff_primitive(NULL)
		, alpha_(0), beta_(0), k_(0)
		, size_(0), num_(0), width_(0), height_(0), channels_(0)
{
  PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
  PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
}

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

    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;
    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

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
void MKLDNNLRNLayer<Dtype>::InitLRNFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    algorithm  lrn_algorithm;
    switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        lrn_algorithm = algorithm::lrn_across_channels;
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        if (this->phase_ == TEST)
            lrn_algorithm = algorithm::lrn_within_channel;
        else
            NOT_IMPLEMENTED;
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
    memory::dims tz = {n, ic, ih, iw};
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format cmfmt = mfmt_nchw;

    // ---- Initialize memory descriptors -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    // ---- Create usr memory primitive descriptors -------------
    shared_ptr<MemPD> usr_data_memory_pd(new MemPD({{tz}, mpcsn, mfmt_nchw}, cpu_engine));

    // ---- Create prv memory descriptors -------------------
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
    }

    bottom_md.reset(new memory::desc({tz}, mpcsn, cmfmt));

    // ---- Initialize LRN primitive descriptor -------------
    lrn_forward::desc lrnFwd_desc(propagation, lrn_algorithm, *bottom_md,
                        size_, alpha_, beta_);
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    lrnFwd_pd = NULL;
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
    // ---- Create priv memory primitive descriptors stored as class members -------------
    shared_ptr<MemPD> prv_fwd_bottom_data_memory_pd(new MemPD(lrnFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_fwd_top_data_memory_pd(new MemPD(lrnFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_memory_pd(new MemPD(lrnFwd_pd->dst_primitive_desc()));

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_data_memory_pd, prv_fwd_bottom_data_memory_pd, bottom[0], this));
    fwd_bottom_data->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_bottom_data_primitive = fwd_bottom_data->create_input(false);
    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_data_memory_pd, prv_fwd_top_data_memory_pd, top[0], this));
    fwd_top_data->name = "fwd_top_data   @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    if ( propagation == prop_kind::forward_training ) {
        memory::primitive_desc scratch_mpd(lrnFwd_pd->workspace_primitive_desc());
        scratch_memory.reset(new memory(scratch_mpd));
        lrnFwd.reset(new lrn_forward(*lrnFwd_pd, *fwd_bottom_data_primitive, *scratch_memory, *fwd_top_data_memory));
    } else {
        lrnFwd.reset(new lrn_forward(*lrnFwd_pd, *fwd_bottom_data_primitive, *fwd_top_data_memory));
    }
    //fwd_bottom_data->set_mkldnn_primitive(lrnFwd);      //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(lrnFwd);         //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}


template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNLRNLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if( lrnFwd_pd == NULL || this->reshape)
        InitLRNFwd(bottom, top);

    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    lrnFwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);
}

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::InitLRNBwd(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{
    if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;

    algorithm  lrn_algorithm;
    switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        lrn_algorithm = algorithm::lrn_across_channels;
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        NOT_IMPLEMENTED;
        break;
    default:
        LOG(FATAL) << "Unknown normalization region.";
    }

    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    bool top_diff_is_prv = (const_cast<Dtype*>(top[0]->prv_diff()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------
    memory::dims tz = {n, ic, ih, iw};
    shared_ptr<memory::desc> bottom_diff_md, top_diff_md;
    shared_ptr<memory::primitive_desc> usr_diff_mpd, prv_diff_mpd;
    if (top_diff_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, true> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, true>(top[0]);
        memory::format bwd_prv_top_diff_mfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
#ifdef DEBUG
        LOG(INFO) << "MKLDNNLRNLayer<Dtype>::InitLRNBwd: memory format of prv top diff is: " << bwd_prv_top_diff_mfmt;
#endif        
        top_diff_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_diff_mpd = mem_descr->usr_memory_pd();
        prv_diff_mpd = mem_descr->prv_memory_pd();

        bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);
        if (bottom_data_is_prv) {
            shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
                = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
            memory::format fwd_prv_bottom_data_mfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
#ifdef DEBUG
            LOG(INFO) << "MKLDNNLRNLayer<Dtype>::InitLRNBwd: memory format of prv bottom data is: " << fwd_prv_bottom_data_mfmt;
#endif
            if (bwd_prv_top_diff_mfmt != fwd_prv_bottom_data_mfmt)
            {
#ifdef DEBUG
                LOG(INFO) << "MKLDNNLRNLayer<Dtype>::InitLRNBwd: Reorder the prv top/bottom diff to the format of prv bottom data! (Performance consideration)";
#endif
                top_diff_md.reset(new memory::desc({tz}, mpcsn, fwd_prv_bottom_data_mfmt));
            }
            //top[0]->set_prv_diff_descriptor(NULL);
        }
    } else {
        memory::format bwd_cmfmt = memory::format::nchw;
        bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);
        if (bottom_data_is_prv) {
            shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
                = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
            memory::format fwd_prv_bottom_data_mfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
#ifdef DEBUG
            LOG(INFO) << "MKLDNNLRNLayer<Dtype>::InitLRNBwd: memory format of prv bottom data is: " << fwd_prv_bottom_data_mfmt;
            LOG(INFO) << "MKLDNNLRNLayer<Dtype>::InitLRNBwd: Reorder the usr top/bottom diff to the format of prv bottom data! (Performance consideration)";
#endif
            bwd_cmfmt = fwd_prv_bottom_data_mfmt;
            //top[0]->set_prv_diff_descriptor(NULL);
        }

        top_diff_md.reset(new memory::desc({tz}, mpcsn, bwd_cmfmt));
        usr_diff_mpd.reset(new memory::primitive_desc(*top_diff_md, cpu_engine));
    }
    bottom_diff_md = top_diff_md;

    // ---- Initialize LRN primitive descriptor -------------
    lrn_backward::desc lrnBwd_desc(lrn_algorithm, *bottom_md, *top_diff_md,
                        size_, alpha_, beta_);
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    lrnBwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        lrnBwd_pd.reset(new lrn_backward::primitive_desc(lrnBwd_desc,
            ep.getMKLDNNSubEngine(subEngineIndex), *lrnFwd_pd));
      }
      catch(...) {
        continue;
      }
      break;
    }
    CHECK(lrnBwd_pd);
    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc
    shared_ptr<MemPD> prv_bwd_bottom_diff_memory_pd(new MemPD(lrnBwd_pd->diff_src_primitive_desc()));
    shared_ptr<MemPD> prv_bwd_top_diff_memory_pd(new MemPD(lrnBwd_pd->diff_dst_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;

    shared_ptr<MemPD> usr_data_memory_pd(new MemPD({{tz}, mpcsn, mfmt_nchw}, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    bwd_bottom_diff.reset(new MKLDNNDiff<Dtype>(usr_data_memory_pd, prv_bwd_bottom_diff_memory_pd, bottom[0], this));
    bwd_bottom_diff->name = "bwd_bottom_diff_data   @ " + this->layer_param_.name();
    bwd_bottom_diff_memory = bwd_bottom_diff->create_output_memory();
    bwd_top_diff.reset(new MKLDNNDiff<Dtype>(usr_diff_mpd, prv_bwd_top_diff_memory_pd, top[0], this));
    bwd_top_diff->name = "bwd_top_diff_data   @ " + this->layer_param_.name();
    bwd_top_diff_primitive = bwd_top_diff->create_input(false);

    lrnBwd.reset(new lrn_backward(*lrnBwd_pd, *fwd_bottom_data_primitive, *bwd_top_diff_primitive, *scratch_memory, *bwd_bottom_diff_memory));
    //bwd_bottom_diff->set_mkldnn_primitive(lrnBwd);        //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_bottom_diff_memory_transfer(bwd_bottom_diff_memory);
    bwd_bottom_diff->set_mkldnn_primitive(bwd_bottom_diff_memory_transfer);

    //bwd_top_diff->set_mkldnn_primitive(lrnBwd);           //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_top_diff_primitive_transfer(bwd_top_diff_primitive);
    bwd_top_diff->set_mkldnn_primitive(bwd_top_diff_primitive_transfer);
}


template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                        ,const vector<bool>& propagate_down
                                        ,const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNLRNLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
    if (!propagate_down[0]) {
        return;
    }
    if( lrnBwd_pd == NULL || this->reshape)
        InitLRNBwd(top, propagate_down, bottom);
    bwd_top_diff->sync_before_read();
    bwd_bottom_diff->sync_before_write();

    PERFORMANCE_EVENT_ID_INIT(perf_id_bw_, PERFORMANCE_MKLDNN_NAME("BW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    lrnBwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);
}

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
