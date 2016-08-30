#ifndef CAFFE_MKLDNN_LAYERS_HPP_
#define CAFFE_MKLDNN_LAYERS_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mkldnn_memory.hpp"
#include "mkldnn.hpp"

using namespace mkldnn;

namespace caffe {

// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine
{
public:
    static CpuEngine & Instance()
    {
        // I's thread-safe in C++11.
        static CpuEngine myInstance;
        return myInstance;
    }
    CpuEngine(CpuEngine const&) = delete;             // Copy construct
    CpuEngine(CpuEngine&&) = delete;                  // Move construct
    CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
    CpuEngine& operator=(CpuEngine &&) = delete;      // Move assign

    engine & get_engine() { return _cpu_engine; }
protected:
    CpuEngine() : _cpu_engine(engine::cpu, 0) {}
//    CpuEngine() : _cpu_engine(engine::cpu_lazy, 0) {}
    ~CpuEngine() {}
private:
    engine _cpu_engine;
};
// =====  MKLDNNConvolutionLayer =======================================
template <typename Dtype>
class MKLDNNConvolutionLayer : public ConvolutionLayer<Dtype>, public MKLDNNLayer<Dtype> {
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
    shared_ptr<convolution::primitive_desc> convFwd_pd;
    shared_ptr<convolution> convFwd;
    shared_ptr<memory> output_memory;
    shared_ptr<primitive> input_primitive, weights_primitive, bias_primitive;
    uint32_t width_, height_, width_out_, height_out_, kernel_w_, kernel_h_, stride_w_, stride_h_;
    int  pad_w_, pad_h_;
};

// =====  MKLDNNInnerProductLayer =======================================
template <typename Dtype>
class MKLDNNInnerProductLayer : public InnerProductLayer<Dtype>, public MKLDNNLayer<Dtype>  {
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
    void InitInnerProduct(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    shared_ptr<MKLDNNData<Dtype> > fwd_bottom_data, fwd_top_data, fwd_weights_data, fwd_bias_data;
    shared_ptr<inner_product::primitive_desc> ipFwd_pd;
    shared_ptr<inner_product> ipFwd;
    shared_ptr<memory> output_memory;
    shared_ptr<primitive> input_primitive, weights_primitive, bias_primitive;
    uint32_t w_, h_;
};


/**
 * @brief Normalize the input in a local region across feature maps.
 */

// =====  MKLDNNLRNLayer =======================================
template <typename Dtype>
class MKLDNNLRNLayer : public Layer<Dtype>, public MKLDNNLayer<Dtype>  {
public:
    explicit MKLDNNLRNLayer(const LayerParameter& param)
        : Layer<Dtype>(param), MKLDNNLayer<Dtype>()
        , fwd_top_data(NULL), fwd_bottom_data (NULL)
        , lrnFwd_pd(NULL), lrnFwd(NULL)
        , input_primitive(NULL), output_memory(NULL) , scratch_(NULL)
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
    shared_ptr<lrn::primitive_desc> lrnFwd_pd;
    shared_ptr<lrn> lrnFwd;
    shared_ptr<memory> output_memory, scratch_;
    shared_ptr<primitive> input_primitive;
    Dtype alpha_, beta_, k_;
    int size_, num_, width_, height_, channels_;
};

// ===== MKLDNNPoolingLayer =======================================
template <typename Dtype>
class MKLDNNPoolingLayer : public Layer<Dtype>, public MKLDNNLayer<Dtype>  {
public:
    explicit MKLDNNPoolingLayer(const LayerParameter& param)
            : Layer<Dtype>(param), MKLDNNLayer<Dtype>()
            , fwd_bottom_data(NULL), fwd_top_data(NULL)
            , poolingFwd_pd(NULL), poolingFwd(NULL)
            , indices_memory(NULL), input_primitive(NULL), output_memory(NULL)
            , indices_pd(NULL)
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

    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<pooling::primitive_desc> poolingFwd_pd;
    shared_ptr<pooling> poolingFwd;
    shared_ptr<memory::primitive_desc> indices_pd;
    shared_ptr<memory> indices_memory, output_memory;
    shared_ptr<primitive> input_primitive;
    uint32_t num_, channels_, width_, height_, width_out_, height_out_;
    uint32_t kernel_w_, kernel_h_, stride_w_, stride_h_;
    int32_t  pad_w_, pad_h_;
    Blob<uint32_t> max_idx_;
    bool global_pooling_;
};

// =====  MKLDNNReLULayer =======================================
template <typename Dtype>
class MKLDNNReLULayer : public NeuronLayer<Dtype>, public MKLDNNLayer<Dtype>  {
public:
    /**
    * @param param provides ReLUParameter relu_param,
    *     with ReLULayer options:
    *   - negative_slope (\b optional, default 0).
    *     the value @f$ \nu @f$ by which negative values are multiplied.
    */
    explicit MKLDNNReLULayer(const LayerParameter& param)
            : NeuronLayer<Dtype>(param), MKLDNNLayer<Dtype>()
            , fwd_top_data(NULL), fwd_bottom_data (NULL)
            , reluFwd_pd(NULL), reluFwd(NULL)
            , input_primitive(NULL), output_memory(NULL)
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
    void InitReLU(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    shared_ptr<MKLDNNData<Dtype> > fwd_top_data, fwd_bottom_data;
    shared_ptr<relu::primitive_desc> reluFwd_pd;
    shared_ptr<relu> reluFwd;
    shared_ptr<memory> output_memory;
    shared_ptr<primitive> input_primitive;
    uint32_t num_, width_, height_, channels_;
};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_LAYERS_HPP_
