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
#include "caffe/filler.hpp"

#include "caffe/layers/mkldnn_layers.hpp"

namespace caffe {

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::InitStatsBatchVars(int batch_size) {
    num_stats_batches_ = 1;
    stats_batch_size_ = batch_size;
    BatchNormParameter param = this->layer_param_.batch_norm_param();
    if (!use_global_stats_ && param.stats_batch_size() > 0) {
      CHECK_EQ(batch_size % param.stats_batch_size(), 0);
      num_stats_batches_ = batch_size / param.stats_batch_size();
      stats_batch_size_ = param.stats_batch_size();
    }
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

    Layer<Dtype>::LayerSetUp(bottom, top);

    channels_ = bottom[0]->channels();
    height_   = bottom[0]->height();
    width_    = bottom[0]->width();
    num_      = bottom[0]->num();

    eps_ = this->layer_param_.batch_norm_param().eps();
    use_weight_bias_ = this->layer_param_.batch_norm_param().use_weight_bias();
    bias_term_ = this->layer_param_.batch_norm_param().bias_term();
    moving_average_fraction_ = this->layer_param_.batch_norm_param().moving_average_fraction();
    use_global_stats_ = this->phase_ == TEST;
    if (this->layer_param_.batch_norm_param().has_use_global_stats())
      use_global_stats_ = this->layer_param_.batch_norm_param().use_global_stats();

    InitStatsBatchVars(num_);

    this->blobs_.resize(3 + (use_weight_bias_ ? 1:0) + (use_weight_bias_ && bias_term_ ? 1:0));

    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0]=1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
        caffe_set(this->blobs_[i]->count(), Dtype(0),
            this->blobs_[i]->mutable_cpu_data());
    }

    //IntelCaffe treat scale and shift as different blobs, so current MKL-DNN integration has additional copies from Caffe to MKL-DNN buffer on fwd pass and from MKL-DNN to Caffe buffer on bwd pass.
    //Optimization: use the temp blob to combine the scale and shift together. Avoid the additional copies.
    // Initialize scale and shift combination blob
    vector<int> scaleshift_blob_shape(1);
    scaleshift_blob_shape[0] = 2*channels_;
    scaleshift_blob_.reset(new Blob<Dtype>(scaleshift_blob_shape));
    //Should initialize the scaleshift_blob_ buffer to 0, because when bias_term_ == false, need to pass zero bias to MKLDNN
    caffe_set(scaleshift_blob_shape[0], static_cast<Dtype>(0),
              scaleshift_blob_->mutable_cpu_data());
    shared_ptr<Blob<Dtype> > scaleshift_diff_blob = scaleshift_blob_;
    scaleshift_acc_ = scaleshift_blob_;
    if (num_stats_batches_ > 1) {
      this->scaleshift_acc_.reset(new Blob<Dtype>(scaleshift_blob_shape));
      scaleshift_diff_blob = scaleshift_acc_;
    }

    if (use_weight_bias_) {
        // Initialize scale and shift
        vector<int> scaleshift_shape(1);
        scaleshift_shape[0] = channels_;
        VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: channels_  = " << channels_;

        this->blobs_[3].reset(new Blob<Dtype>(scaleshift_shape));
        this->blobs_[3]->set_cpu_data(scaleshift_blob_->mutable_cpu_data());
        this->blobs_[3]->set_cpu_diff(scaleshift_diff_blob->mutable_cpu_diff());
        FillerParameter filler_param(this->layer_param_.batch_norm_param().filler());
        if (!this->layer_param_.batch_norm_param().has_filler()) {
            filler_param.set_type("constant");
            filler_param.set_value(1);
        }
        shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
        VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: scaleshift " << __LINE__ << ":" << this->layer_param_.name();
        filler->Fill(this->blobs_[3].get());

        if (bias_term_) {
            this->blobs_[4].reset(new Blob<Dtype>(scaleshift_shape));
            this->blobs_[4]->set_cpu_data(scaleshift_blob_->mutable_cpu_data() + scaleshift_blob_->offset(channels_));
            this->blobs_[4]->set_cpu_diff(scaleshift_diff_blob->mutable_cpu_diff() + scaleshift_blob_->offset(channels_));
            FillerParameter bias_filler_param(this->layer_param_.batch_norm_param().bias_filler());
            if (!this->layer_param_.batch_norm_param().has_bias_filler()) {
                bias_filler_param.set_type("constant");
                bias_filler_param.set_value(0);
            }
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(bias_filler_param));
            VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: bias " << __LINE__ << ":" << this->layer_param_.name();
            bias_filler->Fill(this->blobs_[4].get());
        }
    }

    // Mask statistics from optimization by setting local learning rates
    // for mean, variance, and the bias correction to zero.
    for (int i = 0; i < 3; ++i) {
      if (this->layer_param_.param_size() == i) {
        ParamSpec* fixed_param_spec = this->layer_param_.add_param();
        fixed_param_spec->set_lr_mult(0.f);
      } else {
        CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
            << "Cannot configure batch normalization statistics as layer "
            << "parameters.";
      }
    }
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                    ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::Reshape: " << this->layer_param_.name();

    this->reshape = (this->width_ == bottom[0]->width() &&
                     this->height_ == bottom[0]->height() &&
                     this->channels_ == bottom[0]->channels() &&
                     this->num_ == bottom[0]->num()) ? false : true;

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

    InitStatsBatchVars(this->num_);

    //Fix: should reshape the top blob with the real size of bottom blob
    //top[0]->Reshape(this->num_, this->channels_, this->height_, this->width_);
#ifdef DEBUG
    LOG(INFO) << "size of bottom blob: " << bottom[0]->shape().size();
#endif
    top[0]->ReshapeLike(*bottom[0]);

    if(bottom[0] == top[0] && this->phase_ == TRAIN)
        inplace_buffer.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::InitBatchNorm(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    unsigned flags = 0;
    if (use_weight_bias_) flags |= use_scale_shift;
    if (use_global_stats_) flags |= use_global_stats;

    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;    

    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    bool inplace = (bottom[0] == top[0]);
    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, input_stats_md, output_md, scaleshift_md;
    shared_ptr<memory::primitive_desc> usr_mpd, prv_mpd;
    shared_ptr<memory::primitive_desc> scaleshift_mpd;
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        input_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_mpd = mem_descr->usr_memory_pd();
        prv_mpd = mem_descr->prv_memory_pd();
    } else {
        input_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));   //MKLDNN batch norm only support 4D memory descriptor!
        usr_mpd.reset(new memory::primitive_desc(*input_md, cpu_engine));
    }
    output_md = input_md;
    input_stats_md.reset(new memory::desc(*input_md));
    CHECK(input_stats_md->data.ndims > 0 &&
          input_stats_md->data.dims[0] == this->num_);
    input_stats_md->data.dims[0] = stats_batch_size_;

    // ---- Initialize BatchNorm primitive descriptor -------------
    batch_normalization_forward::desc BatchNormFwd_desc(propagation, *input_stats_md, eps_, flags);
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    BatchNormFwd_pd = NULL;
    bool relu = this->layer_param_.batch_norm_param().relu();
    mkldnn::primitive_attr attr;
    mkldnn::post_ops ops;
    if (relu) {
        ops.append_eltwise(1.f, eltwise_relu, 0.f, 0.f);
        attr.set_post_ops(ops);
    }
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        if (relu)
            BatchNormFwd_pd.reset(new batch_normalization_forward::primitive_desc(BatchNormFwd_desc, attr,
                ep.getMKLDNNSubEngine(subEngineIndex)));
        else
            BatchNormFwd_pd.reset(new batch_normalization_forward::primitive_desc(BatchNormFwd_desc,
                ep.getMKLDNNSubEngine(subEngineIndex)));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(BatchNormFwd_pd);

    // ---- Create memory  ---------------------
    if (use_weight_bias_) {
        //For test in train, memory address of blobs_[3] and blobs_[4] will be changed when share data from train net. If the address
        // of blobs_[3] and blobs_[4] are continued, we will use them immediately, otherwise we will copy them to scaleshift_blob_ in Forward.
        if((this->blobs_[3]->mutable_cpu_data() + this->blobs_[3]->offset(channels_)) == this->blobs_[4]->mutable_cpu_data()){
            scaleshift_memory.reset(new memory(BatchNormFwd_pd->weights_primitive_desc(), this->blobs_[3]->mutable_cpu_data()));
        }else {
            scaleshift_memory.reset(new memory(BatchNormFwd_pd->weights_primitive_desc(), this->scaleshift_blob_->mutable_cpu_data()));
        }
    }

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    if(inplace && this->phase_ == TRAIN) {
        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, &inplace_buffer, this));
    } else {
        fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, top[0], this));
    }
    output_memory = fwd_top_data->create_output_memory();

    mean_memory.resize(num_stats_batches_);
    variance_memory.resize(num_stats_batches_);
    input_stats.resize(num_stats_batches_);
    output_stats.resize(num_stats_batches_);
    BatchNormFwd.resize(num_stats_batches_);
    for (int i = 0; i < num_stats_batches_; i++) {
      InitBatchNormFwdPrimitive(i);
    }

    //fwd_bottom_data->set_mkldnn_primitive(BatchNormFwd);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_bottom_data_primitive_transfer(input_primitive);
    fwd_bottom_data->set_mkldnn_primitive(fwd_bottom_data_primitive_transfer);

    //fwd_top_data->set_mkldnn_primitive(BatchNormFwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> fwd_top_data_memory_transfer(output_memory);
    fwd_top_data->set_mkldnn_primitive(fwd_top_data_memory_transfer);

    //Fix: MKLDNN batch norm only support 4D memory descriptor! Use 4D for calculation and reshape to 2D for output!
    bool has_spatial = (bottom[0]->shape().size() != 2);
#ifdef DEBUG
    LOG(INFO) << "has_spatial flag value: " << has_spatial;
#endif
    if (has_spatial == false)
    {
#ifdef DEBUG
        LOG(INFO) << "size of bottom blob: " << bottom[0]->shape().size();
        LOG(INFO) << "MKLDNN batch norm only support 4D memory descriptor! Use 4D for calculation and reshape to 2D for output!";
#endif
        vector<int> top_shape;
        top_shape.push_back(bottom[0]->num());
        top_shape.push_back(bottom[0]->channels());
        top[0]->Reshape(top_shape);
    }
}

template <typename Dtype>
template <bool diff>
shared_ptr<memory> MKLDNNBatchNormLayer<Dtype>::GetStatsBatchMemory(
  shared_ptr<MKLDNNMemoryDescriptor<Dtype, diff> > mkldnn_mem, int idx) {
    long data_offset =
      idx * stats_batch_size_ * this->channels_ * this->width_ * this->height_;
    engine cpu_engine = CpuEngine::Instance().get_engine();
    shared_ptr<memory::desc> stats_md = mkldnn_mem->get_memory_desc();
    CHECK(stats_md->data.ndims > 0 &&
          stats_md->data.dims[0] == this->num_);
    stats_md->data.dims[0] = stats_batch_size_;
    shared_ptr<memory::primitive_desc> stats_mpd(
      new memory::primitive_desc(*stats_md, cpu_engine));
    shared_ptr<memory> stats(
      new memory(*stats_mpd, mkldnn_mem->get_memory_ptr(data_offset)));
    return stats;
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::InitBatchNormFwdPrimitive(int idx) {
    input_stats[idx] = GetStatsBatchMemory<false>(fwd_bottom_data, idx);
    output_stats[idx] = GetStatsBatchMemory<false>(fwd_top_data, idx);

    // ---- Create BatchNorm --------------------
    if (this->phase_ == TEST && !use_global_stats_) {
        if (use_weight_bias_) {
            BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                    *input_stats[idx], *scaleshift_memory,
                    *output_stats[idx]));
        } else {
            BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                    *input_stats[idx], *output_stats[idx]));
        }
    } else {
        mean_memory[idx].reset(new memory(BatchNormFwd_pd->mean_primitive_desc()));
        variance_memory[idx].reset(new memory(BatchNormFwd_pd->variance_primitive_desc()));

        if (use_global_stats_) {
            caffe_copy<Dtype>(this->channels_, this->blobs_[0]->cpu_data(),
                static_cast<Dtype *>(mean_memory[idx]->get_data_handle()));
            caffe_copy<Dtype>(this->channels_, this->blobs_[1]->cpu_data(),
               static_cast<Dtype *>(variance_memory[idx]->get_data_handle()));
            if (use_weight_bias_) {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], (const primitive::at)*mean_memory[idx],
                        (const primitive::at)*variance_memory[idx], *scaleshift_memory,
                        *output_stats[idx]));
            } else {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], (const primitive::at)*mean_memory[idx],
                        (const primitive::at)*variance_memory[idx], *output_stats[idx]));
            }
        } else {
            if (use_weight_bias_) {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], *scaleshift_memory, *output_stats[idx],
                        *mean_memory[idx], *variance_memory[idx]));
            } else {
                BatchNormFwd[idx].reset(new batch_normalization_forward(*BatchNormFwd_pd,
                        *input_stats[idx], *output_stats[idx], *mean_memory[idx], *variance_memory[idx]));
            }
        }
    }
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNBatchNormLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
#endif

    if(BatchNormFwd_pd == NULL || this->reshape)
        InitBatchNorm(bottom, top);
    bool inplace = (bottom[0] == top[0]);

    // making reorders if needed.
    fwd_bottom_data->sync_before_read();
    // update top that head at prv
    fwd_top_data->sync_before_write();

    if((this->blobs_[3]->mutable_cpu_data() + this->blobs_[3]->offset(channels_)) != this->blobs_[4]->mutable_cpu_data()){
        caffe_copy(channels_, this->blobs_[3]->cpu_data(), this->scaleshift_blob_->mutable_cpu_data());
        caffe_copy(channels_, this->blobs_[4]->cpu_data(), this->scaleshift_blob_->mutable_cpu_data() + scaleshift_blob_->offset(channels_));
    }

    for (int stats_batch_idx = 0; stats_batch_idx < num_stats_batches_; stats_batch_idx++) {
      if (use_global_stats_) {
        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
            0 : 1 / this->blobs_[2]->cpu_data()[0];
        Dtype *mean_buffer_ = (Dtype *)(mean_memory[stats_batch_idx]->get_data_handle());
        Dtype *variance_buffer_ = (Dtype *)(variance_memory[stats_batch_idx]->get_data_handle());

        //TODO: optimize, do this operation in the InitBatchNorm, so no need to calculate each time
        caffe_cpu_scale(this->blobs_[0]->count(), scale_factor,
                    this->blobs_[0]->cpu_data(), mean_buffer_);
        caffe_cpu_scale(this->blobs_[1]->count(), scale_factor,
                    this->blobs_[1]->cpu_data(), variance_buffer_);
      }
      
      PERFORMANCE_EVENT_ID_INIT(perf_id_fw_, PERFORMANCE_MKLDNN_NAME("FW"));
      PERFORMANCE_MEASUREMENT_BEGIN();
      BatchNormFwd[stats_batch_idx].submit();
      PERFORMANCE_MEASUREMENT_END_ID(perf_id_fw_);

      if (this->phase_ == TRAIN && !use_global_stats_) {
        // compute and save moving average
        Dtype *mean_buffer_ = (Dtype *)(mean_memory[stats_batch_idx]->get_data_handle());
        Dtype *variance_buffer_ = (Dtype *)(variance_memory[stats_batch_idx]->get_data_handle());
        this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        caffe_cpu_axpby<Dtype>(this->channels_, Dtype(1), mean_buffer_,
            moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
        int m = bottom[0]->count()/num_stats_batches_/channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
        caffe_cpu_axpby<Dtype>(this->channels_, bias_correction_factor,
            variance_buffer_, moving_average_fraction_,
            this->blobs_[1]->mutable_cpu_data());
      }
    }
    //the prv_descriptor_ will be exchanged back during the previous layer sync_before_write() call.
    if(inplace && this->phase_ == TRAIN)
        bottom[0]->data()->swap((inplace_buffer.data()));
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::InitBatchNormBwd(
        const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;

    int32_t n = this->num_;
    int32_t w = this->width_;
    int32_t h = this->height_;
    int32_t c = this->channels_;

    unsigned flags = 0;
    if (use_weight_bias_) flags |= use_scale_shift;
    if (use_global_stats_) flags |= use_global_stats;

    bool top_diff_is_prv = (const_cast<Dtype*>(top[0]->prv_diff()) != NULL);
    bool inplace = (bottom[0] == top[0]);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;

    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> top_diff_md, top_diff_stats_md, top_data_md, output_stats_md;
    shared_ptr<memory::primitive_desc> usr_diff_mpd(NULL), prv_diff_mpd(NULL);
    if (top_diff_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, true> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, true>(top[0]);
        top_diff_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_diff_mpd = mem_descr->usr_memory_pd();
        prv_diff_mpd = mem_descr->prv_memory_pd();
    } else {
        top_diff_md.reset(new memory::desc({{n, c, h, w}}, mpcsn, memory::format::nchw));   //MKLDNN batch norm only support 4D memory descriptor!
        usr_diff_mpd.reset(new memory::primitive_desc(*top_diff_md, cpu_engine));
    }
    top_diff_stats_md.reset(new memory::desc(*top_diff_md));
    CHECK(top_diff_stats_md->data.ndims > 0 &&
          top_diff_stats_md->data.dims[0] == this->num_);
    top_diff_stats_md->data.dims[0] = stats_batch_size_;
    output_stats_md.reset(new memory::desc(output_memory->get_primitive_desc().desc()));
    CHECK(output_stats_md->data.ndims > 0 &&
          output_stats_md->data.dims[0] == this->num_);
    output_stats_md->data.dims[0] = stats_batch_size_;

    // ---- Initialize bnrm primitive descriptor -------------
    batch_normalization_backward::desc BatchNormBwd_desc(prop_kind::backward,
            *top_diff_stats_md, *output_stats_md, eps_,
            flags);
    // ---- Determining engine to use -----------------------
    std::string subengines = this->layer_param_.engine();
    if (subengines.find("MKLDNN") == std::string::npos || subengines == "MKLDNN")
      subengines = "MKLDNN:CPU";
    EngineParser ep(subengines);
    unsigned subEngineIndex = 0;
    BatchNormBwd_pd = NULL;
    for(; subEngineIndex < ep.getNumberOfSubEngines(); subEngineIndex++) {
      try {
        BatchNormBwd_pd.reset(new batch_normalization_backward::primitive_desc(
                    BatchNormBwd_desc, ep.getMKLDNNSubEngine(subEngineIndex),
                    *BatchNormFwd_pd));
      }
      catch(...) {
        continue;
      }
      break;
    }

    CHECK(BatchNormBwd_pd);

    if (use_weight_bias_) {
        bwd_scaleshift_diff_memory.reset(new memory(
                    BatchNormFwd_pd->weights_primitive_desc(), this->scaleshift_blob_->mutable_cpu_diff()));
    }

    // ---  init primitive and prv_memory descriptors ----------------------
    bwd_top_diff.reset(new MKLDNNDiff<Dtype>(usr_diff_mpd, prv_diff_mpd, top[0], this));
    bwd_top_diff->name = "bwd_top_diff_data   @ " + this->layer_param_.name();
    bwd_top_diff_primitive = bwd_top_diff->create_input(false);

    bwd_bottom_diff.reset(new MKLDNNDiff<Dtype>(usr_diff_mpd, prv_diff_mpd, bottom[0], this));
    bwd_bottom_diff->name = "bwd_bottom_diff_data   @ " + this->layer_param_.name();
    bwd_bottom_diff_memory = bwd_bottom_diff->create_output_memory(inplace);

    top_diff_stats.resize(num_stats_batches_);
    bottom_diff_stats.resize(num_stats_batches_);
    BatchNormBwd.resize(num_stats_batches_);
    for (int i = 0; i < num_stats_batches_; i++) {
      InitBatchNormBwdPrimitive(i);
    }

    //bwd_top_diff->set_mkldnn_primitive(BatchNormBwd);     //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_top_diff_primitive_transfer(bwd_top_diff_primitive);
    bwd_top_diff->set_mkldnn_primitive(bwd_top_diff_primitive_transfer);

    //bwd_bottom_diff->set_mkldnn_primitive(BatchNormBwd);  //Wrong passed primitive! (TODO: Checking!)
    MKLDNNPrimitive<Dtype> bwd_bottom_diff_memory_transfer(bwd_bottom_diff_memory);
    bwd_bottom_diff->set_mkldnn_primitive(bwd_bottom_diff_memory_transfer);
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::InitBatchNormBwdPrimitive(int idx) {
    top_diff_stats[idx] = GetStatsBatchMemory<true>(bwd_top_diff, idx);
    bottom_diff_stats[idx] = GetStatsBatchMemory<true>(bwd_bottom_diff, idx);

    if (use_weight_bias_) {
        BatchNormBwd[idx].reset(new batch_normalization_backward(*BatchNormBwd_pd,
                    *input_stats[idx], *mean_memory[idx], *variance_memory[idx],
                    *top_diff_stats[idx], *scaleshift_memory,
                    *bottom_diff_stats[idx], *bwd_scaleshift_diff_memory));
    } else {
        BatchNormBwd[idx].reset(new batch_normalization_backward(*BatchNormBwd_pd,
                    *input_stats[idx], *mean_memory[idx], *variance_memory[idx],
                    *top_diff_stats[idx], *bottom_diff_stats[idx]));
    }
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#ifdef DEBUG
    LOG(INFO) << "MKLDNNBatchNormLayer<Dtype>::Backward_cpu: " << this->layer_param_.name();
#endif

    if (BatchNormBwd_pd == NULL || this->reshape)
        InitBatchNormBwd(top, propagate_down, bottom);
    // making reorders if needed.
    bwd_top_diff->sync_before_read();
    // update bottom that head at prv
    bwd_bottom_diff->sync_before_write();

    for (int stats_batch_idx = 0; stats_batch_idx < num_stats_batches_; stats_batch_idx++) {

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
        LOG(INFO) << "Debug: Top cpu diff: " << *top[0]->cpu_diff();
      }
#endif
      BatchNormBwd[stats_batch_idx].submit();
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
      if (num_stats_batches_ > 1) {
        CHECK(scaleshift_blob_ != scaleshift_acc_);
        CHECK(scaleshift_blob_->count() == scaleshift_acc_->count());
        caffe_cpu_axpby(scaleshift_acc_->count(), Dtype(1),
                        scaleshift_blob_->mutable_cpu_diff(),
                        Dtype(1), scaleshift_acc_->mutable_cpu_diff());
      }
    }
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNBatchNormLayer);
#else
template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{ NOT_IMPLEMENTED; }

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }
#endif

INSTANTIATE_CLASS(MKLDNNBatchNormLayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
