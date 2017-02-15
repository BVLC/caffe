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

using namespace mkldnn;

namespace caffe {

// =====  MKLDNNBatchNormLayer =======================================
template <typename Dtype>
class MKLDNNBatchNormLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype> {
public:
    explicit MKLDNNBatchNormLayer(const LayerParameter& param)
            : Layer<Dtype>(param)
            , fwd_top_data    ()
            , fwd_bottom_data ()
            , BatchNormFwd_pd()
            , output_memory(), scaleshift_memory()
            , mean_memory()
            , variance_memory()
            , input_primitive()
        {}

    ~MKLDNNBatchNormLayer() {}
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
    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<batch_normalization_forward::primitive_desc> BatchNormFwd_pd;

    MKLDNNPrimitive<Dtype> BatchNormFwd;
    shared_ptr<memory> output_memory, scaleshift_memory, mean_memory, variance_memory;
    shared_ptr<primitive> input_primitive;

    int32_t num_, width_, height_, channels_;
    Dtype eps_, moving_average_fraction_;
    bool use_weight_bias_, bias_term_, use_global_stats_;
};

// =====  MKLDNNConvolutionLayer =======================================
template <typename Dtype>
class MKLDNNConvolutionLayer : public MKLDNNLayer<Dtype> , public ConvolutionLayer<Dtype> {
public:
    explicit MKLDNNConvolutionLayer(const LayerParameter& param);
    virtual ~MKLDNNConvolutionLayer() {}
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
    void InitConvolution(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data;
    shared_ptr<convolution_forward::primitive_desc> convFwd_pd;
    MKLDNNPrimitive<Dtype> convFwd;
    shared_ptr<memory> output_memory;
    shared_ptr<primitive> input_primitive, weights_primitive, bias_primitive;
    int32_t width_, height_, width_out_, height_out_, kernel_w_, kernel_h_, stride_w_, stride_h_;
    int  pad_w_, pad_h_;
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
};


/**
 * @brief Normalize the input in a local region across feature maps.
 */

// =====  MKLDNNLRNLayer =======================================
template <typename Dtype>
class MKLDNNLRNLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype>  {
public:
    explicit MKLDNNLRNLayer(const LayerParameter& param)
        : MKLDNNLayer<Dtype>(), Layer<Dtype>(param)
        , fwd_top_data(), fwd_bottom_data ()
        , lrnFwd_pd()
        , output_memory(), scratch_(), input_primitive()
        , alpha_(0.), beta_(0.), k_(0.)
        , size_(0), num_(0), width_(0), height_(0), channels_(0)
        {}
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
    void InitLRN(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<lrn_forward::primitive_desc> lrnFwd_pd;
    MKLDNNPrimitive<Dtype> lrnFwd;
    shared_ptr<memory> output_memory, scratch_;
    shared_ptr<primitive> input_primitive;
    Dtype alpha_, beta_, k_;
    int size_, num_, width_, height_, channels_;
};

// ===== MKLDNNPoolingLayer =======================================
template <typename Dtype>
class MKLDNNPoolingLayer : public MKLDNNLayer<Dtype>, public Layer<Dtype>  {
public:
    explicit MKLDNNPoolingLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(), Layer<Dtype>(param)
            , fwd_bottom_data(), fwd_top_data()
            , poolingFwd_pd()
            , indices_pd()
            , indices_memory(), output_memory(), input_primitive()
            , num_(0), channels_(0), width_(0), height_(0), width_out_(0), height_out_(0)
            , kernel_w_(0), kernel_h_(0), stride_w_(0), stride_h_(0)
            , pad_w_(0), pad_h_(0)
            , global_pooling_(false)
            {}
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

private:
    void InitPooling(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data;
    shared_ptr<pooling_forward::primitive_desc> poolingFwd_pd;
    MKLDNNPrimitive<Dtype> poolingFwd;
    shared_ptr<memory::primitive_desc> indices_pd;
    shared_ptr<memory> indices_memory, output_memory;
    shared_ptr<primitive> input_primitive;
    int32_t num_, channels_, width_, height_, width_out_, height_out_;
    int32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
    int32_t  pad_w_, pad_h_;
    Blob<uint32_t> max_idx_;
    bool global_pooling_;
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
    : MKLDNNLayer<Dtype>(), NeuronLayer<Dtype>(param)
    , fwd_top_data(), fwd_bottom_data()
    , bwd_top_diff(), bwd_bottom_diff()
    , reluFwd_pd(), reluBwd_pd()
    , fwd_top_data_memory(), bwd_bottom_diff_memory()
    , fwd_bottom_data_primitive(), bwd_top_diff_primitive()
    , num_(0), width_(0), height_(0), channels_(0)
  {}
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

    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<MKLDNNDiff<Dtype> > bwd_top_diff, bwd_bottom_diff;
    shared_ptr<relu_forward::primitive_desc> reluFwd_pd;
    shared_ptr<relu_backward::primitive_desc> reluBwd_pd;
    MKLDNNPrimitive<Dtype> reluFwd, reluBwd;
    shared_ptr<memory> fwd_top_data_memory, bwd_bottom_diff_memory;
    shared_ptr<primitive> fwd_bottom_data_primitive, bwd_top_diff_primitive;
    int32_t num_, width_, height_, channels_;
};

// ===== MKLDNNConcatLayer ======================================
template <typename Dtype>
class MKLDNNConcatLayer : public MKLDNNLayer<Dtype> , public Layer<Dtype> {
public:
    explicit MKLDNNConcatLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(), Layer<Dtype>(param),
            concatFwd_pd(), output_memory(),
            fwd_top_data(), fwd_bottom_data(), split_channels() {
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
    void InitConcat(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    shared_ptr<concat::primitive_desc> concatFwd_pd;
    shared_ptr<memory> output_memory;
    vector<shared_ptr<primitive>> input_primitives_;
    vector<primitive::at> input_primitives_at_;
    MKLDNNPrimitive<Dtype> concatFwd;
    shared_ptr<MKLDNNData<Dtype> > fwd_top_data;
    vector<shared_ptr<MKLDNNData<Dtype> > > fwd_bottom_data;
    vector<int> split_channels;

    int32_t num_, width_, height_, channels_, num_concats_;
};
}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_LAYERS_HPP_
