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

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNEltwiseLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

    Layer<Dtype>::LayerSetUp(bottom, top);

    CHECK(this->layer_param().eltwise_param().coeff_size() == 0
        || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
        "Eltwise Layer takes one coefficient per bottom blob.";
    CHECK(!(this->layer_param().eltwise_param().operation() == EltwiseParameter_EltwiseOp_PROD
        && this->layer_param().eltwise_param().coeff_size())) <<
        "Eltwise layer only takes coefficients for summation.";
    op_ = this->layer_param_.eltwise_param().operation();
    // Blob-wise coefficients for the elementwise operation.
    coeffs_ = vector<Dtype>(bottom.size(), 1);
    if (this->layer_param().eltwise_param().coeff_size())
    {
        for (int i = 0; i < bottom.size(); ++i) 
        {
            coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
        }
    }
    num_bottoms_ = bottom.size();
    stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
}

template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNEltwiseLayer<Dtype>::Reshape: " << this->layer_param_.name();
    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

    switch (op_)
    {
    case EltwiseParameter_EltwiseOp_PROD:
        NOT_IMPLEMENTED;
        break;
    case EltwiseParameter_EltwiseOp_SUM:
        {
            for (int i = 1; i < num_bottoms_; ++i)
            {
                CHECK(bottom[i]->shape() == bottom[0]->shape());
            }
            top[0]->ReshapeLike(*bottom[0]);
        }
        break;
    case EltwiseParameter_EltwiseOp_MAX:
        NOT_IMPLEMENTED;
        /*
        {
            for (int i = 1; i < num_bottoms_; ++i)
            {
                CHECK(bottom[i]->shape() == bottom[0]->shape());
            }
            top[0]->ReshapeLike(*bottom[0]);
            // If max operation, we will initialize the vector index part.
            if (this->layer_param_.eltwise_param().operation() == EltwiseParameter_EltwiseOp_MAX && top.size() == 1)
            {
                max_idx_.Reshape(bottom[0]->shape());
            }
        }
        */
        break;
    default:
        LOG(FATAL) << "Unknown elementwise operation.";
    }
}

template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::InitEltwiseFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    // If we just do simple adding, scale is 1.0 for all inputs we have
    std::vector<double> scale(num_bottoms_, 1.0);
    //Eltwise layer is supporting multiplication coefficient and this scale value can be used for that.
    for (int i = 0; i < num_bottoms_; ++i) 
    {
        scale[i] = coeffs_[i];
    }

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_nchw = memory::format::nchw;

    // ---- Initialize memory descriptors -------------
    std::vector<memory::primitive_desc> bottom_data_mpd;
    fwd_bottom_data.clear();
    fwd_bottom_data_primitives_.clear();
    fwd_bottom_data_primitives_at_.clear();
    for (auto i = 0; i < num_bottoms_; i++) 
    {
        fwd_bottom_data.push_back(boost::shared_ptr<MKLDNNData<Dtype> >());
        memory::format bottom_data_mfmt = mfmt_nchw;
        shared_ptr<memory::primitive_desc> prv_bottom_data_mpd;
        shared_ptr<memory::primitive_desc> usr_bottom_data_mpd(
            new memory::primitive_desc({{n, ic, ih, iw}, mpcsn, bottom_data_mfmt}, cpu_engine));

        bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[i]->prv_data()) != NULL);
        if (bottom_data_is_prv)
        {
            shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
                = get_mkldnn_prv_descriptor<Dtype, false>(bottom[i]);
            bottom_data_mfmt = static_cast<memory::format>(
                mem_descr->prv_memory_pd()->desc().data.format);
            prv_bottom_data_mpd.reset(new memory::primitive_desc(
                {{n, ic, ih, iw}, mpcsn, bottom_data_mfmt}, cpu_engine));
        }

        bottom_data_mpd.push_back(memory::primitive_desc(
            {{n, ic, ih, iw}, mpcsn, bottom_data_mfmt}, cpu_engine));

        fwd_bottom_data[i].reset(new MKLDNNData<Dtype>(
            usr_bottom_data_mpd, prv_bottom_data_mpd, bottom[i], this));        
        fwd_bottom_data[i]->name = "fwd_bottom_data[i]   @ " + this->layer_param_.name();
        fwd_bottom_data_primitives_.push_back(fwd_bottom_data[i]->create_input(false));
        fwd_bottom_data_primitives_at_.push_back(*fwd_bottom_data_primitives_[i]);
    }

    shared_ptr<memory::primitive_desc> usr_top_data_mpd(new memory::primitive_desc(
        {{n, ic, ih, iw}, mpcsn, mfmt_nchw}, cpu_engine));
    
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines == "" || subengines == "MKLDNN")
        subengines = "MKLDNN:CPU";
    eltwiseFwd_pd.reset(new sum::primitive_desc({{n, ic, ih, iw}, mpcsn, memory::format::any}, scale, bottom_data_mpd));
    CHECK(eltwiseFwd_pd);

    shared_ptr<memory::primitive_desc> prv_top_data_mpd(new memory::primitive_desc(eltwiseFwd_pd->dst_primitive_desc()));

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_top_data_mpd, prv_top_data_mpd, top[0], this));
    fwd_top_data->name = "fwd_top_data   @ " + this->layer_param_.name();
    fwd_top_data_memory = fwd_top_data->create_output_memory();

    eltwiseFwd.reset(new sum(*eltwiseFwd_pd, fwd_bottom_data_primitives_at_, *fwd_top_data_memory));
    
    for (auto i = 0; i < num_bottoms_; i++)
    {
        //fwd_bottom_data[i]->set_mkldnn_primitive(eltwiseFwd);   //Wrong passed primitive! (TODO: Checking!)
        MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(fwd_bottom_data_primitives_[i]);
        fwd_bottom_data[i]->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);
    }
    //fwd_top_data->set_mkldnn_primitive(eltwiseFwd);             //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_top_data_memory_transfer(fwd_top_data_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);
}


template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNEltwiseLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();

    if(eltwiseFwd_pd == NULL || this->reshape)
        InitEltwiseFwd(bottom, top);
    for (auto i = 0; i < num_bottoms_; i++)
    {
        // making reorders if needed.
        fwd_bottom_data[i]->sync_before_read();
    }
    // update top that head at prv
    fwd_top_data->sync_before_write();

    PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
    PERFORMANCE_MEASUREMENT_BEGIN();
    eltwiseFwd.submit();
    PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);
}

template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                          , const vector<bool>& propagate_down
                                          , const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNEltwiseLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();

    for (int i = 0; i < num_bottoms_; ++i) 
    {
        //Eltwise layer is not supporting multiplication coefficient in Backward due to lack of supporting scale and copy primitives in MKL-DNN
        CHECK_EQ(coeffs_[i], Dtype(1)) << "Not supported yet";

        bottom[i]->ShareDiff(*top[0]);
    }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNEltwiseLayer);
#else

template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                           , const vector<bool>& propagate_down
                                           , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNEltwiseLayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
