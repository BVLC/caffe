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

    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;
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
    //relu_forward::desc reluFwd_desc(propagation, *bottom_data_md, negative_slope);
    // MKLDNN is deprecating standalone relu primitive in MKL-DNN.
    // Now MKLDNN has eltwise primitive with eltwise_relu algorithm inside.
    eltwise_forward::desc eltwise_reluFwd_desc(propagation, eltwise_relu, *bottom_data_md, negative_slope);

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    reluFwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        reluFwd_pd.reset(new relu_forward::primitive_desc(eltwise_reluFwd_desc,
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
    //fwd_bottom_data->set_mkldnn_primitive(reluFwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(reluFwd);        //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}


template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNReLULayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#endif

    bool inplace = (bottom[0] == top[0]);
    if( reluFwd_pd == NULL || this->reshape)
        InitReLUFwd(bottom, top);

    if(this->layer_param_.relu_param().fuse()) {
      top[0]->ShareData(*bottom[0]);
      return;
    }
    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write(inplace);

    PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    reluFwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);
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
    shared_ptr<memory::desc> bottom_diff_md = NULL;
    shared_ptr<memory::desc> top_diff_md = NULL;
    shared_ptr<memory::desc> top_data_md = NULL;

    shared_ptr<memory::primitive_desc> usr_diff_mpd = NULL;
    shared_ptr<memory::primitive_desc> prv_diff_mpd = NULL;

    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> bottom_data_md;
    shared_ptr<memory::primitive_desc> usr_data_mpd(NULL), prv_data_mpd(NULL);
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        usr_data_mpd = mem_descr->usr_memory_pd();
        prv_data_mpd = mem_descr->prv_memory_pd();
    } else {
        bottom_data_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
        usr_data_mpd.reset(new memory::primitive_desc(*bottom_data_md, cpu_engine));
    }

    if (top_diff_is_prv) {
      shared_ptr<MKLDNNMemoryDescriptor<Dtype, /* is_diff */ true> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype, /* is_diff */ true>(top[0]);
      memory::format bwd_prv_top_diff_mfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
#ifdef DEBUG
      LOG(INFO) << "MKLDNNReLULayer<Dtype>::InitReLUBwd: memory format of prv top diff is: " << bwd_prv_top_diff_mfmt;
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
          LOG(INFO) << "MKLDNNReLULayer<Dtype>::InitReLUBwd: memory format of prv bottom data is: " << fwd_prv_bottom_data_mfmt;
#endif
          if (bwd_prv_top_diff_mfmt != fwd_prv_bottom_data_mfmt)
          {
#ifdef DEBUG
              LOG(INFO) << "MKLDNNReLULayer<Dtype>::InitReLUBwd: Reorder the prv top/bottom diff to the format of prv bottom data! (Performance consideration)";
#endif
              prv_diff_mpd = mem_descr->prv_memory_pd();
          }
          //top[0]->set_prv_diff_descriptor(NULL);
      }
    } else {
      bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);
      if (bottom_data_is_prv) {
          shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
              = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
#ifdef DEBUG
          memory::format fwd_prv_bottom_data_mfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
          LOG(INFO) << "MKLDNNReLULayer<Dtype>::InitReLUBwd: memory format of prv bottom data is: " << fwd_prv_bottom_data_mfmt;
          LOG(INFO) << "MKLDNNReLULayer<Dtype>::InitReLUBwd: Reorder the usr top/bottom diff to the format of prv bottom data! (Performance consideration)";
#endif
          prv_diff_mpd = mem_descr->prv_memory_pd();
          //top[0]->prv_data() is empty, however top[0]->get_prv_diff_descriptor() has value.
          //Find root cause in the mkldnn_memory: create_output_memory() and sync_before_write() functions.
          //But that a major fix, will lead the nan in the AlexNet training.
          //So need investigation further, however, this will fix ICL-84.
          top[0]->set_prv_diff_descriptor(NULL);
      }

      top_diff_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
      usr_diff_mpd.reset(new memory::primitive_desc(*top_diff_md, cpu_engine));
    }
    
    top_data_md = top_diff_md;
    bottom_diff_md = top_diff_md;

    // ---- Initialize relu primitive descriptor -------------
    //relu_backward::desc reluBwd_desc(*top_diff_md, *top_data_md, negative_slope);
    // MKLDNN is deprecating standalone relu primitive in MKL-DNN.
    // Now MKLDNN has eltwise primitive with eltwise_relu algorithm inside.
    eltwise_backward::desc eltwise_reluBwd_desc(eltwise_relu, *top_diff_md, *top_data_md, negative_slope);

    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    reluBwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        reluBwd_pd.reset(new relu_backward::primitive_desc(eltwise_reluBwd_desc,
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

    bwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_data_mpd, prv_data_mpd, bottom[0], this));
    bwd_bottom_data->name = "bwd_bottom_data   @ " + this->layer_param_.name();
    bwd_bottom_data_primitive = bwd_bottom_data->create_input(/* set_prv_ptr */ false);

    bwd_bottom_diff.reset(new MKLDNNDiff<Dtype>(usr_diff_mpd, prv_diff_mpd, bottom[0], this));
    bwd_bottom_diff->name = "bwd_bottom_diff_data   @ " + this->layer_param_.name();
    bwd_bottom_diff_memory = bwd_bottom_diff->create_output_memory(inplace);

    reluBwd.reset(new relu_backward(*reluBwd_pd, *bwd_bottom_data_primitive, *bwd_top_diff_primitive, *bwd_bottom_diff_memory));
    //bwd_top_diff->set_mkldnn_primitive(reluBwd);          //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_top_diff_primitive_transfer(bwd_top_diff_primitive);
    bwd_top_diff->set_mkldnn_primitive(bwd_top_diff_primitive_transfer);

    //bwd_bottom_diff->set_mkldnn_primitive(reluBwd);       //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_bottom_diff_memory_transfer(bwd_bottom_diff_memory);
    bwd_bottom_diff->set_mkldnn_primitive(bwd_bottom_diff_memory_transfer);
}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                          , const vector<bool>& propagate_down
                                          , const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNReLULayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#endif

    //bool inplace = (bottom[0] == top[0]);
    if (!propagate_down[0]) {
        return;
    }
    if (reluBwd_pd == NULL || this->reshape) {
        InitReLUBwd(top, propagate_down, bottom);
    }

    bwd_top_diff->sync_before_read();
    bwd_bottom_data->sync_before_read();
    //For MKLDNN, it always create two memory for input and output
    //For Intel Caffe, if we set the inplace flag to true, input and output will use one same buffer
    //Then the update of output will not pass to MKLDNN
    //bwd_bottom_diff->sync_before_write(inplace);   //Wrong due to the MKLDNN API design.
    bwd_bottom_diff->sync_before_write();

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
    }

    if (top[0]->prv_diff() != NULL)
    {
        LOG(INFO) << "Debug: Top prv diff: " << *top[0]->prv_diff();
    }
    else
    {
        LOG(INFO) << "Debug: Top prv diff is NULL!";
    }
#endif
    reluBwd.submit();
#ifdef DEBUG
    if (bottom[0]->prv_diff() != NULL)
    {
        LOG(INFO) << "Debug: Bottom prv diff: " << *bottom[0]->prv_diff();
    }
    else
    {
        LOG(INFO) << "Debug: Bottom prv diff is NULL!";
    }
#endif
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_bw_);
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
