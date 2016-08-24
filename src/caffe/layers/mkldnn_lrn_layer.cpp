#ifdef MKLDNN_SUPPORTED
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mkldnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
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
        NOT_IMPLEMENTED;
        break;
    default:
        LOG(FATAL) << "Unknown normalization region.";
    }
    if( lrnFwd_pd == NULL) {
        InitLRN(bottom, top);
    }
}

template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::InitLRN(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;
    auto propagation = this->phase_ == TEST ? prop_kind::forward_scoring : prop_kind::forward_training;

    lrn::algorithm  lrn_algorithm;
    switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        lrn_algorithm = lrn::algorithm::across_channels;
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        lrn_algorithm = lrn::algorithm::within_channel;
        break;
    default:
        LOG(FATAL) << "Unknown normalization region.";
    }

    uint32_t n  = this->num_;
    uint32_t iw = this->width_;
    uint32_t ih = this->height_;
    uint32_t ic = this->channels_;

    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::precision mpcsn = memory::precision::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, output_md;
    shared_ptr<memory::primitive_desc> usr_mpd(NULL), prv_mpd(NULL);
    if (bottom_data_is_prv) {
        CHECK_EQ((bottom[0]->get_prv_data_descriptor())->get_descr_type(), PrvMemDescr::PRV_DESCR_MKLDNN);
        shared_ptr<MKLDNNData<Dtype> > mem_descr
                = boost::static_pointer_cast<MKLDNNData<Dtype> >(bottom[0]->get_prv_data_descriptor());
        CHECK(mem_descr != NULL);
        input_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
        usr_mpd = mem_descr->usr_memory_pd();
        prv_mpd = mem_descr->prv_memory_pd();
    } else {
        input_md.reset(new memory::desc({{n, ic, ih, iw}}, mpcsn, memory::format::nchw));
        usr_mpd.reset(new memory::primitive_desc(*input_md, cpu_engine));
    }
    output_md = input_md;
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));
    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));

    // ---- Initialize LRN primitive descriptor -------------
    lrn::desc lrnFwd_desc(propagation, lrn_algorithm, *input_md
                            ,*output_md, alpha_, beta_, size_);
    lrnFwd_pd.reset(new lrn::primitive_desc(lrnFwd_desc, cpu_engine));

    memory::primitive_desc scratch_mpd(memory::desc(lrnFwd_pd->data.scratch_primitive_desc.memory_desc), cpu_engine);
    scratch_.reset(new memory(scratch_mpd));

    // ---- Create memory  ---------------------
    input_memory = fwd_bottom_data->create_input_memory(bottom[0]);
    if (fwd_top_data->conversion_needed())
        top[0]->set_prv_data_descriptor(fwd_top_data);
    output_memory = fwd_top_data->create_output_memory(top[0]);

    // ---- Create lrn --------------------
    lrnFwd.reset(new lrn(*lrnFwd_pd, *input_memory, *scratch_, *output_memory));
}


template <typename Dtype>
void MKLDNNLRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNLRNLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    // making reorders if needed.
    fwd_bottom_data->sync_blob_prv_data(bottom[0]);
    // update top that head at prv
    if (fwd_top_data->conversion_needed())
        top[0]->set_prv_data_descriptor(fwd_top_data);

    stream().submit({*lrnFwd}).wait();
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
