#ifdef MKLDNN_SUPPORTED
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "mkl_service.h"

// TODO: Correct process case if there are no bias
// TODO: Exception handling - mkl-dnn produces exceptions on errors

namespace caffe {

template <typename Dtype>
MKLDNNConvolutionLayer<Dtype>::MKLDNNConvolutionLayer(const LayerParameter& param)
            : MKLDNNLayer<Dtype>(), ConvolutionLayer<Dtype>(param)
            , fwd_bottom_data(NULL), fwd_top_data(NULL), fwd_weights_data(NULL), fwd_bias_data(NULL)
            , convFwd_pd(NULL), output_memory(NULL)
            , input_primitive(NULL), weights_primitive(NULL), bias_primitive(NULL)
            , width_(0), height_(0), width_out_(0), height_out_(0), kernel_w_(0), kernel_h_(0)
            , stride_w_(0), stride_h_(0), pad_w_(0), pad_h_(0)
{
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::compute_output_shape()
{
    ConvolutionLayer<Dtype>::compute_output_shape();
    this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
        / this->stride_h_ + 1;
    this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
        / this->stride_w_ + 1;
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::init_properties(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    this->stride_w_ = this->stride_.cpu_data()[0];
    this->stride_h_ = this->stride_.cpu_data()[1];
    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->pad_w_ = this->pad_.cpu_data()[0];
    this->pad_h_ = this->pad_.cpu_data()[1];
    this->kernel_w_ = this->kernel_shape_.cpu_data()[0];
    this->kernel_h_  = this->kernel_shape_.cpu_data()[1];
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "<< MKLDNNConvolutionLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();
    ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
    init_properties(bottom, top);
    this->bottom_shape_ = &bottom[0]->shape();
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                            , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << " MKLDNNConvolutionLayer<Dtype>::Reshape: " << this->layer_param_.name();
    BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
    init_properties(bottom, top);
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::InitConvolution(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value)   NOT_IMPLEMENTED;

    int32_t g  = std::max(this->group_, 1);
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    int32_t ow = this->width_out_;
    int32_t oh = this->height_out_;
    int32_t oc = this->num_output_;

    int32_t kw = this->kernel_w_;
    int32_t kh = this->kernel_h_;

    memory::dims convolutionStrides {this->stride_h_, this->stride_w_};
    memory::dims padding {this->pad_h_, this->pad_w_};

    // ---- Initialize memory descriptors (fromat = any) to create convolution descriptor -------------
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt_any = memory::format::any;
    engine cpu_engine = CpuEngine::Instance().get_engine();

    memory::dims input_tz = {n, ic, ih, iw};
    memory::dims bias_tz = {oc};
    memory::dims output_tz = {n, oc, oh, ow};
    memory::dims weights_tz = ( g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};

    // ---- Memory descriptors for initializing of convolution primitive descriptor -------------
    memory::desc init_input_md({input_tz}, mpcsn, mfmt_any);
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt_any);
    memory::desc init_output_md({output_tz}, mpcsn, mfmt_any);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt_any);

    // ---- Initialize convolution primitive descriptor -------------
    shared_ptr<convolution_forward::desc> convFwd_desc;
    if (this->bias_term_) {
        convFwd_desc.reset(new convolution_forward::desc(prop_kind::forward, convolution_forward::direct
                                    , init_input_md, init_weights_md, init_bias_md, init_output_md
                                    , convolutionStrides, padding, padding, padding_kind::zero));
    } else {
        convFwd_desc.reset(new convolution_forward::desc(prop_kind::forward, convolution_forward::direct
                                    , init_input_md, init_weights_md, init_output_md
                                    , convolutionStrides, padding, padding, padding_kind::zero));
    }

    convFwd_pd.reset(new convolution_forward::primitive_desc(*convFwd_desc, cpu_engine));

    // ---- Create priv memory primitive descriptors stored as class members -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    shared_ptr<MemPD> prv_input_memory_pd(new MemPD(convFwd_pd->src_primitive_desc()));
    shared_ptr<MemPD> prv_bias_memory_pd(new MemPD(convFwd_pd->bias_primitive_desc()));
    shared_ptr<MemPD> prv_output_memory_pd(new MemPD(convFwd_pd->dst_primitive_desc()));
    shared_ptr<MemPD> prv_weights_memory_pd(new MemPD(convFwd_pd->weights_primitive_desc()));

    // ---- Create usr memory primitive descriptors -------------
    memory::format mfmt_nchw = memory::format::nchw;
    memory::format weights_mfmt = ( g!= 1) ? memory::format::goihw : memory::format::oihw;
    shared_ptr<MemPD> usr_input_memory_pd(new MemPD({{input_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_bias_memory_pd(new MemPD({{bias_tz}, mpcsn, memory::format::x}, cpu_engine));
    shared_ptr<MemPD> usr_output_memory_pd(new MemPD({{output_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_weights_memory_pd(new MemPD({{weights_tz}, mpcsn, weights_mfmt}, cpu_engine));

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_input_memory_pd, prv_input_memory_pd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_output_memory_pd, prv_output_memory_pd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    fwd_weights_data.reset(new MKLDNNData<Dtype>(usr_weights_memory_pd, prv_weights_memory_pd, this->blobs_[0].get(), this));
    weights_primitive = fwd_weights_data->create_input(false);

    if (this->bias_term_) {
        fwd_bias_data.reset(new MKLDNNData<Dtype>(usr_bias_memory_pd, prv_bias_memory_pd, this->blobs_[1].get(), this));
        bias_primitive = fwd_bias_data->create_input(false);
    }
    convFwd.reset(new convolution_forward(*convFwd_pd
                        , *input_primitive, *weights_primitive
                        , *bias_primitive, *output_memory));
    fwd_bottom_data->set_mkldnn_primitive(convFwd);
    fwd_top_data->set_mkldnn_primitive(convFwd);
    fwd_weights_data->set_mkldnn_primitive(convFwd);
    fwd_bias_data->set_mkldnn_primitive(convFwd);

    // Names are for debugging purposes only.
    fwd_bottom_data ->name = "fwd_bottom_data   @ " + this->layer_param_.name();
    fwd_top_data    ->name = "fwd_top_data      @ " + this->layer_param_.name();
    fwd_weights_data->name = "fwd_weights_data  @ " + this->layer_param_.name();
    fwd_bias_data   ->name = "fwd_bias_data     @ " + this->layer_param_.name();
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNConvolutionLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if( convFwd_pd == NULL)
        InitConvolution(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read(false);
    fwd_weights_data->sync_before_read(true);
    fwd_bias_data->sync_before_read(true);
    // update top that head at prv
    fwd_top_data->sync_before_write();

    convFwd.submit();
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MKLDNNConvolutionLayer);
#else

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                                , const vector<Blob<Dtype>*>& top)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                                , const vector<bool>& propagate_down
                                                , const vector<Blob<Dtype>*>& bottom)
{
    NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNConvolutionLayer);

}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
