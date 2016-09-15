#ifdef MKLDNN_SUPPORTED
#include <algorithm>
#include <vector>
#include "caffe/filler.hpp"

#include "caffe/layers/mkldnn_layers.hpp"

namespace caffe {

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
    // Workaround. Checking count of parameters in order to handle
    // topology for reference BatchNorm layer which don't have scaling
    if (this->layer_param_.param_size() == 3) {
        this->blobs_.resize(3);
        use_weight_bias_ = false;
    }
    if (use_weight_bias_) {
        if ( bias_term_ ) {
            this->blobs_.resize(2);
        } else {
            this->blobs_.resize(1);
        }
        // Initialize scale and shift
        vector<int> scaleshift_shape(1);
        scaleshift_shape[0] = channels_;
        VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: channels_  = " << channels_;

        this->blobs_[0].reset(new Blob<Dtype>(scaleshift_shape));
        FillerParameter filler_param(this->layer_param_.batch_norm_param().filler());
        if (!this->layer_param_.batch_norm_param().has_filler()) {
            filler_param.set_type("constant");
            filler_param.set_value(1);
        }
        shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
        VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: scaleshift " << __LINE__ << ":" << this->layer_param_.name();
        filler->Fill(this->blobs_[0].get());

        if ( bias_term_ ) {
            this->blobs_[1].reset(new Blob<Dtype>(scaleshift_shape));
            FillerParameter bias_filler_param(this->layer_param_.batch_norm_param().bias_filler());
            if (!this->layer_param_.batch_norm_param().has_bias_filler()) {
                bias_filler_param.set_type("constant");
                bias_filler_param.set_value(0);
            }
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(bias_filler_param));
            VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::LayerSetUp: bias " << __LINE__ << ":" << this->layer_param_.name();
            bias_filler->Fill(this->blobs_[1].get());
        }
    }
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom
                                    ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::Reshape: " << this->layer_param_.name();

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

    top[0]->Reshape(this->num_, this->channels_, this->height_, this->width_);
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::InitBatchNorm(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    int32_t n  = this->num_;
    int32_t iw = this->width_;
    int32_t ih = this->height_;
    int32_t ic = this->channels_;

    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::precision mpcsn = memory::precision::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, output_md, scaleshift_md;
    shared_ptr<memory::primitive_desc> usr_mpd(NULL), prv_mpd(NULL), scaleshift_mpd(NULL);
    if (bottom_data_is_prv) {
        CHECK_EQ((bottom[0]->get_prv_data_descriptor())->get_descr_type()
                    ,PrvMemDescr::PRV_DESCR_MKLDNN);
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
    scaleshift_md.reset(new memory::desc({{2, ic}}, mpcsn, memory::format::nc));
    scaleshift_mpd.reset(new memory::primitive_desc(*scaleshift_md, cpu_engine));

    // ---- Initialize BatchNorm primitive descriptor -------------
    batch_normalization::desc BatchNormFwd_desc(prop_kind::forward, *input_md, *output_md, *scaleshift_md, eps_);
    BatchNormFwd_pd.reset(new batch_normalization::primitive_desc(BatchNormFwd_desc, cpu_engine));
    // ---- Create memory  ---------------------
    shared_ptr<memory::primitive_desc> ws_memory_pd(new memory::primitive_desc(BatchNormFwd_pd->data.workspace_primitive_desc));

    scaleshift_memory.reset(new memory(*scaleshift_mpd));
    ws_memory.reset(new memory(*ws_memory_pd));

    if (!use_weight_bias_) {
        Dtype* scaleShift_buffer_ = (Dtype *)(scaleshift_memory->get_data_handle());
        for (int i = 0; i < ic; i++) {
            scaleShift_buffer_[i] = 1.0;
            scaleShift_buffer_[channels_ + i] = 0;
        }
    }

    // ---  init primitive and prv_memory descriptors ----------------------
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, bottom[0], this));
    input_primitive = fwd_bottom_data->create_input(false);

    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd, top[0], this));
    output_memory = fwd_top_data->create_output_memory();

    // ---- Create BatchNorm --------------------
    BatchNormFwd.reset(new batch_normalization(*BatchNormFwd_pd, *input_primitive, *output_memory, *scaleshift_memory, *ws_memory));
    fwd_bottom_data->set_mkldnn_primitive(BatchNormFwd);
    fwd_top_data->set_mkldnn_primitive(BatchNormFwd);
}


template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNBatchNormLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();

    if( BatchNormFwd_pd == NULL)
        InitBatchNorm(bottom, top);
    // making reorders if needed.
    fwd_bottom_data->sync_before_read(false);
    // update top that head at prv
    fwd_top_data->sync_before_write();

    if (use_weight_bias_) {
        Dtype* scaleShift_buffer_ = (Dtype *)(scaleshift_memory->get_data_handle());
        // Fill ScaleShift buffer
        for (int i = 0; i < this->channels_; i++) {
            scaleShift_buffer_[i] = this->blobs_[0]->cpu_data()[i];
            scaleShift_buffer_[channels_ + i] = 0;
            if (bias_term_) {
                scaleShift_buffer_[channels_ + i] = this->blobs_[1]->cpu_data()[i];
            }
        }
    }

    BatchNormFwd.submit();
}

template <typename Dtype>
void MKLDNNBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }

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
