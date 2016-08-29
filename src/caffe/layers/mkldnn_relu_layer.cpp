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

    this->width_ = bottom[0]->width();
    this->height_ = bottom[0]->height();
    this->num_ = bottom[0]->num();
    this->channels_ = bottom[0]->channels();

    if( reluFwd_pd == NULL) {
        InitReLU(bottom, top);
    } else {
        VLOG(1) << " Reshape: second call";
    }
}



template <typename Dtype>
void MKLDNNReLULayer<Dtype>::InitReLU(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    if (std::is_same<Dtype, double>::value) NOT_IMPLEMENTED;
    uint32_t n  = this->num_;
    uint32_t iw = this->width_;
    uint32_t ih = this->height_;
    uint32_t ic = this->channels_;

    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    bool bottom_data_is_prv = (const_cast<Dtype*>(bottom[0]->prv_data()) != NULL);

    engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::precision mpcsn = memory::precision::f32;
    // ---- Initialize memory descriptors -------------
    shared_ptr<memory::desc> input_md, output_md;
    shared_ptr<memory::primitive_desc> usr_mpd(NULL), prv_mpd(NULL);
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
    fwd_bottom_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));
    fwd_top_data.reset(new MKLDNNData<Dtype>(usr_mpd, prv_mpd));

    // ---- Initialize relu primitive descriptor -------------
    relu::desc reluFwd_desc(prop_kind::forward, negative_slope, *input_md, *output_md);
    reluFwd_pd.reset(new relu::primitive_desc(reluFwd_desc, cpu_engine));
    // ---- Create memory  ---------------------
    input_primitive = fwd_bottom_data->create_input(bottom[0], false);
    if (fwd_top_data->conversion_needed())
        top[0]->set_prv_data_descriptor(fwd_top_data);
    output_memory = fwd_top_data->create_output_memory(top[0]);

    // ---- Create relu --------------------
    reluFwd.reset(new relu(*reluFwd_pd, *input_primitive, *output_memory));
    fwd_bottom_data->set_primitives(reluFwd, bottom[0]);
    fwd_top_data->set_mkldnn_primitive(reluFwd);
}


template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom
                                        ,const vector<Blob<Dtype>*>& top)
{
    VLOG(1) << "MKLDNNReLULayer<Dtype>::Forward_cpu: " << this->layer_param_.name();
    // making reorders if needed.
    fwd_bottom_data->sync_blob_prv_data(bottom[0], false);
    // update top that head at prv
    if (fwd_top_data->conversion_needed())
        top[0]->set_prv_data_descriptor(fwd_top_data);

    stream().submit({*reluFwd}).wait();
}

template <typename Dtype>
void MKLDNNReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top
                                            ,const vector<bool>& propagate_down
                                            ,const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }

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
