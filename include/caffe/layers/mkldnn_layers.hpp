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

#ifndef CAFFE_MKLDNN_LAYERS_HPP_
#define CAFFE_MKLDNN_LAYERS_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/engine_parser.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mkldnn_memory.hpp"
#include "mkldnn.hpp"

#include "caffe/util/performance.hpp"

using namespace mkldnn;

namespace caffe {

// =====  Log functions ==============================================
template <typename Dtype>
inline void info_mem_pd(shared_ptr<memory::primitive_desc> mem_pd, string name) {
#ifdef DEBUG        
    LOG(INFO) << name;
    // format of mem_pd
    switch (mem_pd->desc().data.format) {
        case memory::format::nchw: LOG(INFO) << "format: nchw"; break;
        case memory::format::nhwc: LOG(INFO) << "format: nhwc"; break;
        case memory::format::nChw8c: LOG(INFO) << "format: nChw8c"; break;
        case memory::format::nChw16c: LOG(INFO) << "format: nChw16c"; break;
        case memory::format::nc: LOG(INFO) << "format: nc"; break;
        default: assert(!"Error format");
    }
    // data_type
    switch (mem_pd->desc().data.data_type) {
        case memory::data_type::f32: LOG(INFO) << "data_type: f32"; break;
        case memory::data_type::u8: LOG(INFO) << "data_type: u8"; break;
        case memory::data_type::s8: LOG(INFO) << "data_type: s8"; break;
        case memory::data_type::s32: LOG(INFO) << "data_type: s32"; break;
        default: assert(!"Error data_type");
    }
#endif
}


// =====  MKLDNNBatchNormLayer =======================================
template <typename Dtype>
class MKLDNNBatchNormLayer : public MKLDNNLayer<Dtype>, public Layer<Dtype> {
public:
    explicit MKLDNNBatchNormLayer(const LayerParameter& param)
        : MKLDNNLayer<Dtype>(param), Layer<Dtype>(param)
        , fwd_top_data(), fwd_bottom_data()
        , bwd_top_diff(), bwd_bottom_diff()
        , BatchNormFwd_pd(), BatchNormBwd_pd()
        , scaleshift_memory(), bwd_scaleshift_diff_memory()
        , output_memory(), bwd_bottom_diff_memory()
        , input_primitive(), bwd_top_diff_primitive()
        {
          PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
          PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
    }
    ~MKLDNNBatchNormLayer() {}
#ifdef USE_MLSL
    virtual bool ParamNeedReduce(int param_id) { return param_id >= 3; }
#endif

protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "BatchNorm"; }
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
private:
    void InitBatchNorm(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitBatchNormBwd(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    void InitBatchNormFwdPrimitive(int stats_batch_idx);
    void InitBatchNormBwdPrimitive(int stats_batch_idx);
    template <bool diff> shared_ptr<memory> GetStatsBatchMemory(
      shared_ptr<MKLDNNMemoryDescriptor<Dtype, diff> > mkldnn_data, int idx);
    void InitStatsBatchVars(int batch_size);
    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;
    shared_ptr<batch_normalization_forward::primitive_desc> BatchNormFwd_pd;
    shared_ptr<batch_normalization_backward::primitive_desc> BatchNormBwd_pd;

    vector<MKLDNNPrimitive<Dtype> > BatchNormFwd, BatchNormBwd;
    vector<shared_ptr<memory> > mean_memory, variance_memory;

    shared_ptr<memory> scaleshift_memory, bwd_scaleshift_diff_memory;
    shared_ptr<memory> output_memory, bwd_bottom_diff_memory;

    vector<shared_ptr<memory> > input_stats, output_stats, top_diff_stats, bottom_diff_stats;

    shared_ptr<primitive> input_primitive, bwd_top_diff_primitive;

    vector<int> shape_;
    Dtype eps_, moving_average_fraction_;
    bool use_weight_bias_, bias_term_, use_global_stats_;
    int num_stats_batches_;
    int stats_batch_size_;
    shared_ptr<Blob<Dtype> > scaleshift_blob_;
    shared_ptr<Blob<Dtype> > scaleshift_acc_;
    Blob<Dtype> inplace_buffer;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
};

// =====  MKLDNNConvolutionLayer =======================================
template <typename Dtype>
class MKLDNNConvolutionLayer : public MKLDNNLayer<Dtype> , public ConvolutionLayer<Dtype> {
public:
    explicit MKLDNNConvolutionLayer(const LayerParameter& param);
    virtual ~MKLDNNConvolutionLayer() {}

    //For test the parameters of kernel/stride/pad
    int GetKernelWidth()  { return kernel_w_; }
    int GetKernelHeight() { return kernel_h_; }
    int GetKernelDepth()  { return kernel_d_; }

    int GetStrideWidth()  { return stride_w_; }
    int GetStrideHeight() { return stride_h_; }
    int GetStrideDepth()  { return stride_d_; }

    int GetPadWidth()     { return pad_w_; }
    int GetPadHeight()    { return pad_h_; }
    int GetPadDepth()     { return pad_d_; }
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    // Customized methods
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
private:
    virtual void compute_output_shape();
    virtual void init_properties(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitConvolutionFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitConvolutionBwd(const vector<Blob<Dtype>*>& top
                        , const vector<bool>& propagate_down
                        , const vector<Blob<Dtype>*>& bottom);

    shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data
                    , bwdd_weights_data, bwdw_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwdd_bottom_diff, bwdd_top_diff
                    , bwdw_top_diff, bwdw_weights_diff, bwdw_bias_diff;
    shared_ptr<convolution_forward::primitive_desc> convFwd_pd;
    shared_ptr<convolution_backward_data::primitive_desc> convBwdData_pd;
    shared_ptr<convolution_backward_weights::primitive_desc> convBwdWeights_pd;
    MKLDNNPrimitive<Dtype> convFwd, convBwdData, convBwdWeights;
    shared_ptr<memory> fwd_top_data_memory, bwdd_bottom_diff_memory
                    , bwdw_weights_diff_memory,  bwdw_bias_diff_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, fwd_weights_data_primitive, fwd_bias_data_primitive
                    , bwdd_top_diff_primitive, bwdd_weights_data_primitive
                    , bwdw_top_diff_primitive, bwdw_bottom_data_primitive;
    int32_t width_, height_, depth_, width_out_, height_out_, depth_out_, kernel_w_, kernel_h_, kernel_d_, stride_w_, stride_h_, stride_d_;
    int  pad_w_, pad_h_, pad_d_;
    mkldnn::algorithm  conv_algorithm;

    /* In case of (iter_size > 1) we need additional buffers */
    shared_ptr<MKLDNNDiff<Dtype> > bwdw_weights_diff_iter, bwdw_bias_diff_iter;
    shared_ptr<memory> bwdw_weights_diff_memory_iter, bwdw_bias_diff_memory_iter;
    shared_ptr<Blob<Dtype> > bwdw_weights_diff_iter_blob, bwdw_bias_diff_iter_blob;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_weights_);
};

// =====  MKLDNNInnerProductLayer =======================================
template <typename Dtype>
class MKLDNNInnerProductLayer : public MKLDNNLayer<Dtype> , public InnerProductLayer<Dtype>  {
public:
    explicit MKLDNNInnerProductLayer(const LayerParameter& param);
    virtual ~MKLDNNInnerProductLayer();
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    // Customized methods
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
private:
    void InitInnerProductFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitInnerProductBwd(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);

    shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data
                    , bwdd_weights_data, bwdw_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwdd_bottom_diff, bwdd_top_diff
                    , bwdw_top_diff, bwdw_weights_diff, bwdw_bias_diff;
    shared_ptr<inner_product_forward::primitive_desc> ipFwd_pd;
    shared_ptr<inner_product_backward_data::primitive_desc> ipBwdData_pd;
    shared_ptr<inner_product_backward_weights::primitive_desc> ipBwdWeights_pd;

    MKLDNNPrimitive<Dtype> ipFwd, ipBwdData, ipBwdWeights;
    shared_ptr<memory> fwd_top_data_memory, bwdd_bottom_diff_memory
                    , bwdw_weights_diff_memory,  bwdw_bias_diff_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, fwd_weights_data_primitive, fwd_bias_data_primitive
                    , bwdd_top_diff_primitive, bwdd_weights_data_primitive
                    , bwdw_top_diff_primitive, bwdw_bottom_data_primitive;
    int32_t w_, h_;

    /* In case of (iter_size > 1) we need additional buffers */
    shared_ptr<MKLDNNDiff<Dtype> > bwdw_weights_diff_iter, bwdw_bias_diff_iter;
    shared_ptr<memory> bwdw_weights_diff_memory_iter, bwdw_bias_diff_memory_iter;
    shared_ptr<Blob<Dtype> > bwdw_weights_diff_iter_blob, bwdw_bias_diff_iter_blob;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_weights_);
};


/**
 * @brief Normalize the input in a local region across feature maps.
 */

// =====  MKLDNNLRNLayer =======================================
template <typename Dtype>
class MKLDNNLRNLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype>  {
public:
    explicit MKLDNNLRNLayer(const LayerParameter& param);
    virtual ~MKLDNNLRNLayer() {}
protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);

    virtual inline const char* type() const { return "LRN"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
private:
    void InitLRNFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitLRNBwd(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);

    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;
    shared_ptr<lrn_forward::primitive_desc> lrnFwd_pd;
    shared_ptr<lrn_backward::primitive_desc> lrnBwd_pd;
    MKLDNNPrimitive<Dtype> lrnFwd;
    MKLDNNPrimitive<Dtype> lrnBwd;
    shared_ptr<memory::desc> bottom_md;

    int fl;
    float scale;
    shared_ptr<memory> buffer;
    MKLDNNPrimitive<Dtype> lrn_reorder;

    shared_ptr<memory> fwd_top_data_memory, bwd_bottom_diff_memory, scratch_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, bwd_top_diff_primitive;
    Dtype alpha_, beta_, k_;
    int size_, num_, width_, height_, channels_;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
};

// ===== MKLDNNPoolingLayer =======================================
template <typename Dtype>
class MKLDNNPoolingLayer : public MKLDNNLayer<Dtype>, public Layer<Dtype>  {
public:
    explicit MKLDNNPoolingLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(param), Layer<Dtype>(param)
            , fwd_bottom_data(), fwd_top_data()
            , bwd_top_diff(), bwd_bottom_diff()
            , poolingFwd_pd()
            , poolingBwd_pd()
            , indices_pd()
            , indices_memory(), fwd_top_data_memory(), bwd_bottom_diff_memory()
            , fwd_bottom_data_primitive(), bwd_top_diff_primitive()
            , num_(0), channels_(0), width_(0), height_(0), width_out_(0), height_out_(0)
            , kernel_w_(0), kernel_h_(0), stride_w_(0), stride_h_(0)
            , pad_t_(0),pad_b_(0), pad_l_(0), pad_r_(0)
            , global_pooling_(false)
            , force_exclude_padding_flag_(false)
            {
              PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
              PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
            }
    ~MKLDNNPoolingLayer() {}
protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Pooling"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }
    // MAX POOL layers can output an extra top blob for the mask;
    // others can only output the pooled inputs.
    virtual inline int MaxTopBlobs() const {
        return (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) ? 2 : 1;
    }
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down
                                ,const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                ,const vector<Blob<Dtype>*>& bottom);
    virtual void compute_output_shape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

private:
    void InitPoolingFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitPoolingBwd(const vector<Blob<Dtype>*>& bottom
                        , const vector<bool>& propagate_down
                        , const vector<Blob<Dtype>*>& top);

    shared_ptr<MKLDNNData<Dtype>> fwd_bottom_data, fwd_top_data;
    shared_ptr<MKLDNNDiff<Dtype>> bwd_top_diff, bwd_bottom_diff;
    shared_ptr<pooling_forward::primitive_desc> poolingFwd_pd;
    shared_ptr<pooling_backward::primitive_desc> poolingBwd_pd;
    MKLDNNPrimitive<Dtype> poolingFwd, poolingBwd;
    shared_ptr<memory::primitive_desc> indices_pd;
    shared_ptr<memory> indices_memory, fwd_top_data_memory, bwd_bottom_diff_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, bwd_top_diff_primitive;
    int32_t num_, channels_, width_, height_, width_out_, height_out_;
    int32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
    int32_t  pad_t_, pad_b_, pad_l_, pad_r_;
    Blob<uint32_t> max_idx_;
    bool global_pooling_;
    bool force_exclude_padding_flag_;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
};

// =====  MKLDNNReLULayer =======================================
template <typename Dtype>
class MKLDNNReLULayer : public MKLDNNLayer<Dtype> , public NeuronLayer<Dtype>  {
public:
    /**
    * @param param provides ReLUParameter relu_param,
    *     with ReLULayer options:
    *   - negative_slope (\b optional, default 0).
    *     the value @f$ \nu @f$ by which negative values are multiplied.
    */
  explicit MKLDNNReLULayer(const LayerParameter& param)
    : MKLDNNLayer<Dtype>(param), NeuronLayer<Dtype>(param)
    , fwd_top_data(), fwd_bottom_data()
    , bwd_top_diff(), bwd_bottom_diff()
    , reluFwd_pd(), reluBwd_pd()
    , fwd_top_data_memory(), bwd_bottom_diff_memory()
    , fwd_bottom_data_primitive(), bwd_top_diff_primitive()
    , shape_(0)
  {
    PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
    PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
  }
  ~MKLDNNReLULayer() {}

protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "ReLU"; }
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
private:
    void InitReLUFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitReLUBwd(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);

    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data, bwd_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;
    shared_ptr<relu_forward::primitive_desc> reluFwd_pd;
    shared_ptr<relu_backward::primitive_desc> reluBwd_pd;
    MKLDNNPrimitive<Dtype> reluFwd, reluBwd;
    shared_ptr<memory> fwd_top_data_memory, bwd_bottom_diff_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, bwd_top_diff_primitive, bwd_bottom_data_primitive;
    vector<int> shape_;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
};

// ===== MKLDNNConcatLayer ======================================
template <typename Dtype>
class MKLDNNConcatLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype> {
public:
    explicit MKLDNNConcatLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(param), Layer<Dtype>(param),
            concatFwd_pd(), fwd_output_memory(),
            bwd_reorder_input_memory(), bwd_reorder_output_memory(),
            fwd_top_data(), fwd_bottom_data(), split_dims() {
              PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
              PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
    }
protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "Concat"; }
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
private:
    void InitConcatFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitConcatBwd(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);

    shared_ptr<concat::primitive_desc> concatFwd_pd;
    shared_ptr<memory> fwd_output_memory;
    shared_ptr<primitive> bwd_reorder_input_memory;
    vector<shared_ptr<memory>> bwd_reorder_output_memory;
    vector<shared_ptr<memory>> bwd_bottom_memory_;
    vector<shared_ptr<primitive>> fwd_input_primitives_;
    vector<primitive::at> fwd_input_primitives_at_;
    MKLDNNPrimitive<Dtype> concatFwd;
    shared_ptr<MKLDNNData<Dtype> > fwd_top_data;
    vector<shared_ptr<MKLDNNData<Dtype> > > fwd_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwd_top_diff;
    vector<shared_ptr<MKLDNNDiff<Dtype> > > bwd_bottom_diff;
    vector<MKLDNNPrimitive<Dtype> > reorders;
    vector<int> split_dims;
    bool in_place_;

    int32_t num_concats_;
    vector<int> shape_;
    int concat_dimension;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
};

// ===== MKLDNNSplitLayer ======================================
template <typename Dtype>
class MKLDNNSplitLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype> {
public:
    explicit MKLDNNSplitLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(param), Layer<Dtype>(param),
              splitBwd_pd_(), bwd_bottom_diff_memory_()
            {
              PERFORMANCE_EVENT_ID_RESET(perf_id_bw_);
    }
    ~MKLDNNSplitLayer();

protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "Split"; }
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
private:
    void InitSplitFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitSplitBwd(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);

  private:
    vector<size_t> sizes_src_;
    vector<size_t> strides_src_;
    MKLDNNPrimitive<Dtype> splitBwd_;
    shared_ptr<sum::primitive_desc> splitBwd_pd_;
    shared_ptr<memory> bwd_bottom_diff_memory_;
    shared_ptr<MKLDNNDiff<Dtype> > bwd_bottom_diff_;
    vector<shared_ptr<primitive>> bwd_top_diff_primitives_;
    vector<primitive::at> bwd_top_diffs_primitives_at_;
    vector<shared_ptr<MKLDNNDiff<Dtype> > > bwd_top_diffs_;

    PERFORMANCE_EVENT_ID_DECL(perf_id_bw_);
};

// =====  MKLDNNEltwiseLayer =======================================
template <typename Dtype>
class MKLDNNEltwiseLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype>  {
public:
  explicit MKLDNNEltwiseLayer(const LayerParameter& param)
    : MKLDNNLayer<Dtype>(param), Layer<Dtype>(param)
    , fwd_top_data(), fwd_bottom_data()
    , eltwiseFwd_pd()
    , fwd_top_data_memory()
    , fwd_bottom_data_primitives_()
    , shape_(0)
    , num_bottoms_(0)
  {
    PERFORMANCE_EVENT_ID_RESET(perf_id_fw_);
  }
  ~MKLDNNEltwiseLayer() {}

protected:
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "Eltwise"; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
private:
    void InitEltwiseFwd(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    void InitEltwiseBwd(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down
                                , const vector<Blob<Dtype>*>& bottom);
   
    shared_ptr<MKLDNNData<Dtype> > fwd_top_data;
    vector<shared_ptr<MKLDNNData<Dtype> > > fwd_bottom_data;
    shared_ptr<sum::primitive_desc> eltwiseFwd_pd;
    MKLDNNPrimitive<Dtype> eltwiseFwd;

    shared_ptr<memory> fwd_top_data_memory;
    vector<shared_ptr<primitive>> fwd_bottom_data_primitives_;
    vector<primitive::at> fwd_bottom_data_primitives_at_;

    EltwiseParameter_EltwiseOp op_;
    vector<Dtype> coeffs_;
    Blob<int> max_idx_;
    vector<int> shape_;
    int32_t num_bottoms_;
    bool stable_prod_grad_;

    PERFORMANCE_EVENT_ID_DECL(perf_id_fw_);
};


}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_LAYERS_HPP_
