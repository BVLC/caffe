#ifdef MKLDNN_SUPPORTED
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                            ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNPoolingLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

    Layer<Dtype>::LayerSetUp(bottom, top);
    PoolingParameter pool_param = this->layer_param_.pooling_param();

    if (pool_param.global_pooling()) {
        CHECK(!(pool_param.has_kernel_size() || pool_param.has_kernel_h() || pool_param.has_kernel_w()))
            << "With Global_pooling: true Filter size cannot specified";
    } else {
        CHECK(!pool_param.has_kernel_size() != !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
            << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
        CHECK(pool_param.has_kernel_size() ||(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
            << "For non-square filters both kernel_h and kernel_w are required.";
    }
    CHECK((!pool_param.has_pad() && pool_param.has_pad_h() && pool_param.has_pad_w())
            || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
        << "pad is pad OR pad_h and pad_w are required.";
    CHECK((!pool_param.has_stride() && pool_param.has_stride_h() && pool_param.has_stride_w())
            || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
        << "Stride is stride OR stride_h and stride_w are required.";

    global_pooling_ = pool_param.global_pooling();
    if (global_pooling_) {
        kernel_h_ = bottom[0]->height();
        kernel_w_ = bottom[0]->width();
    } else {
        if (pool_param.has_kernel_size()) {
            kernel_h_ = kernel_w_ = pool_param.kernel_size();
        } else {
            kernel_h_ = pool_param.kernel_h();
            kernel_w_ = pool_param.kernel_w();
        }
    }
    CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
    if (!pool_param.has_pad_h()) {
        pad_h_ = pad_w_ = pool_param.pad();
    } else {
        pad_h_ = pool_param.pad_h();
        pad_w_ = pool_param.pad_w();
    }
    if (!pool_param.has_stride_h()) {
        stride_h_ = stride_w_ = pool_param.stride();
    } else {
        stride_h_ = pool_param.stride_h();
        stride_w_ = pool_param.stride_w();
    }
    if (global_pooling_) {
        CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
            << "With Global_pooling: true; only pad = 0 and stride = 1";
    }
    if (pad_h_ != 0 || pad_w_ != 0) {
        CHECK(this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
        CHECK_LT(pad_h_, kernel_h_);
        CHECK_LT(pad_w_, kernel_w_);
    }

    height_out_ = static_cast<int>(ceil(static_cast<float>(
        bottom[0]->height() + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    width_out_ = static_cast<int>(ceil(static_cast<float>(
        bottom[0]->width() + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    if (pad_h_ || pad_w_) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        if ((height_out_ - 1) * stride_h_ >= bottom[0]->height() + pad_h_) {
          --height_out_;
        }
        if ((width_out_ - 1) * stride_w_ >= bottom[0]->height() + pad_w_) {
          --width_out_;
        }
        CHECK_LT((height_out_ - 1) * stride_h_, bottom[0]->height() + pad_h_);
        CHECK_LT((width_out_ - 1) * stride_w_, bottom[0]->height() + pad_w_);
    }
}

template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNPoolingLayer<Dtype>::Reshape: "  << this->layer_param_.name();

    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
        << "corresponding to (num, channels, height, width)";

    top[0]->Reshape(bottom[0]->num(), channels_, height_out_, width_out_);

    if (top.size() > 1) {
        (reinterpret_cast<Blob<uint32_t>* > (top[1]) )->Reshape(num_,
            channels_, height_out_, width_out_);
    }
    if (top.size() == 1) {
        max_idx_.Reshape(bottom[0]->num(), channels_, height_out_, width_out_);
    }
}

template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::InitPooling(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;

    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    pooling_forward::algorithm pooling_algorithm;
    switch (this->layer_param_.pooling_param().pool()) {
    case PoolingParameter_PoolMethod_MAX:
        pooling_algorithm = pooling_forward::algorithm::max;
        break;
    case PoolingParameter_PoolMethod_AVE:
        NOT_IMPLEMENTED;
        break;
    case PoolingParameter_PoolMethod_STOCHASTIC:
        NOT_IMPLEMENTED;
        break;
    default:
        LOG(FATAL) << "Unknown pooling method.";
    }

    int32_t n = this->num_;
    int32_t c = this->channels_;
    int32_t ih = this->height_;
    int32_t iw = this->width_;
    int32_t oh = this->height_out_;
    int32_t ow = this->width_out_;

    int32_t kh = this->kernel_h_;
    int32_t kw = this->kernel_w_;

    int32_t sh = this->stride_h_;
    int32_t sw = this->stride_w_;

    int32_t ph = this->pad_h_;
    int32_t pw = this->pad_w_;

    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    memory::dims input_tz = {n, c, ih, iw};
    memory::dims output_tz = {n, c, oh, ow};
    memory::format mfmt_nchw = memory::format::nchw;

    // ---- Initialize memory descriptors -------------
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc

    memory::format cmfmt = mfmt_nchw;
    if (bottom_data_is_prv) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
            = get_mkldnn_prv_descriptor<Dtype, false>(bottom[0]);
        cmfmt = static_cast<memory::format>(mem_descr->prv_memory_pd()->desc().data.format);
    }
    shared_ptr<memory::desc> input_md(new memory::desc({input_tz}, mpcsn, cmfmt));
    shared_ptr<memory::desc> output_md(new memory::desc({output_tz}, mpcsn, cmfmt));

    shared_ptr<MemPD> usr_input_mpd(new MemPD({{input_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> usr_output_mpd(new MemPD({{output_tz}, mpcsn, mfmt_nchw}, cpu_engine));
    shared_ptr<MemPD> prv_input_mpd(NULL);
    shared_ptr<MemPD> prv_output_mpd(NULL);
    if (bottom_data_is_prv) {
        prv_input_mpd.reset(new MemPD(*input_md, cpu_engine));
        prv_output_mpd.reset(new MemPD(*output_md, cpu_engine));
    }

    // ---- Initialize pooling primitive descriptor -------------
    pooling_forward::desc poolingFwd_desc(propagation, pooling_algorithm, *input_md,*output_md
                                        , {sh, sw}, {kh, kw}, {ph, pw}, {ph, pw}, padding_kind::zero);
    poolingFwd_pd.reset(new pooling_forward::primitive_desc(poolingFwd_desc, cpu_engine));

    // ---- Create priv memory  ---------------------

    // We'll output the mask to top[1] if it's of size >1.
    uint32_t* mask = NULL;  // suppress warnings about uninitalized variables
    // We'll output the mask to top[1] if it's of size >1.
    const bool use_top_mask = top.size() > 1;
    mask = (use_top_mask) ?  reinterpret_cast<uint32_t*>(top[1]->mutable_cpu_data())
            : max_idx_.mutable_cpu_data();

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_input_mpd, prv_input_mpd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_output_mpd, prv_output_mpd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    if ( propagation == prop_kind::forward_training ) {
        indices_pd.reset(new MemPD(poolingFwd_pd->workspace_primitive_desc()));
        indices_memory.reset(new memory(*indices_pd, reinterpret_cast<void *>(mask)));
        poolingFwd.reset(new pooling_forward(*poolingFwd_pd, *input_primitive, *output_memory, *indices_memory));
    } else {
        poolingFwd.reset(new pooling_forward(*poolingFwd_pd, *input_primitive, *output_memory));
    }
    fwd_bottom_data->set_mkldnn_primitive(poolingFwd);
    fwd_top_data->set_mkldnn_primitive(poolingFwd);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                            ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNPoolingLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    if (NULL == poolingFwd_pd)
        InitPooling(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read(false);
    // update top that head at prv
    fwd_top_data->sync_before_write();

    poolingFwd.submit();
}

template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            , const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }

#ifdef CPU_ONLY
STUB_GPU(MKLDNNPoolingLayer);
#else
template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom
                                            ,const vector<Blob<Dtype>*>& top)
{ NOT_IMPLEMENTED; }

template <typename Dtype>
void MKLDNNPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }
#endif

INSTANTIATE_CLASS(MKLDNNPoolingLayer);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
