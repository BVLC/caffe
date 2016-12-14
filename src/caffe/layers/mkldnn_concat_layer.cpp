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
void MKLDNNConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // VLOG(1) << "MKLDNNConcatLayer<Dtype>::LayerSetUp: " << this->layer_param_.name();

  int dim_src = bottom[0]->shape().size();
  int dim_dst = dim_src;

  num_concats_ = bottom.size();
  channels_ = 0;

  for (auto i = 1; i < num_concats_; ++i) {
    CHECK_EQ(bottom[0]->num(), bottom[i]->num());
    CHECK_EQ(bottom[0]->height(), bottom[i]->height());
    CHECK_EQ(bottom[0]->width(), bottom[i]->width());
  }

  split_channels.reserve(num_concats_);
  for (auto i = 0; i < num_concats_; ++i) {
    CHECK_EQ(dim_src, bottom[i]->shape().size());

    split_channels[i] = bottom[i]->channels();
    channels_ += split_channels[i];
  }
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // VLOG(1) << "MKLDNNConcatLayer<Dtype>::Reshape: "  << this->layer_param_.name();

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::InitConcat(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (std::is_same<Dtype, double>::value)  NOT_IMPLEMENTED;

  engine cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type data_type = memory::data_type::f32;
  memory::format mfmt_any = memory::format::any;
  memory::format mfmt_nchw = memory::format::nchw;

  memory::dims output_tz = {num_, channels_, height_, width_};

  std::vector<memory::primitive_desc> srcs_mpd;
  std::vector<primitive::at> srcs;
  for (auto i = 0; i < num_concats_; i++) {
    fwd_bottom_data.push_back(boost::shared_ptr<MKLDNNData<Dtype> >());
    memory::dims input_tz = {num_, split_channels[i], height_, width_};
    memory::format src_mfmt = mfmt_nchw;
    shared_ptr<memory::primitive_desc> prv_src_mpd;
    shared_ptr<memory::primitive_desc> usr_src_mpd(
        new memory::primitive_desc({input_tz, data_type, mfmt_nchw}, cpu_engine));

    if (const_cast<Dtype*>(bottom[i]->prv_data()) != NULL) {
      shared_ptr<MKLDNNMemoryDescriptor<Dtype, false> > mem_descr
        = get_mkldnn_prv_descriptor<Dtype, false>(bottom[i]);
      src_mfmt = static_cast<memory::format>(
          mem_descr->prv_memory_pd()->desc().data.format);
      prv_src_mpd.reset(new memory::primitive_desc(
            {input_tz, data_type, src_mfmt}, cpu_engine));
    }

    srcs_mpd.push_back(memory::primitive_desc(
          {input_tz, data_type, src_mfmt}, cpu_engine));

    fwd_bottom_data[i].reset(new MKLDNNData<Dtype>(
          usr_src_mpd, prv_src_mpd, bottom[i], this));

    input_primitives_.push_back(fwd_bottom_data[i]->create_input(false));
    input_primitives_at_.push_back(*input_primitives_[i]);
  }

  shared_ptr<memory::primitive_desc> usr_dst_mpd(new memory::primitive_desc(
        {output_tz, data_type, mfmt_nchw}, cpu_engine));

  // FIXME: concat dimension
  auto concat_dimension = 1;
  concatFwd_pd.reset(new concat::primitive_desc(concat_dimension, srcs_mpd));

  shared_ptr<memory::primitive_desc> prv_dst_mpd(new memory::primitive_desc(
        concatFwd_pd->dst_primitive_desc()));

  fwd_top_data.reset(new MKLDNNData<Dtype>(usr_dst_mpd, prv_dst_mpd, top[0],
        this));
  output_memory = fwd_top_data->create_output_memory();

  concatFwd.reset(new concat(*concatFwd_pd, input_primitives_at_, *output_memory));

  for (auto i = 0; i < num_concats_; i++) {
    fwd_bottom_data[i]->set_mkldnn_primitive(concatFwd);
  }
  fwd_top_data->set_mkldnn_primitive(concatFwd);
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // VLOG(1) << "MKLDNNPoolingLayer<Dtype>::Forward_cpu: " << this->layer_param_.name();

  if (NULL == concatFwd_pd)
    InitConcat(bottom, top);
  for (auto i = 0; i < num_concats_; i++) {
    // making reorders if needed.
    fwd_bottom_data[i]->sync_before_read(false);
  }
  // update top that head at prv
  fwd_top_data->sync_before_write();

  concatFwd.submit();
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{ NOT_IMPLEMENTED; }

#ifdef CPU_ONLY
STUB_GPU(MKLDNNConcatLayer);
#else
template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MKLDNNConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#endif

INSTANTIATE_CLASS(MKLDNNConcatLayer);

} // namespace caffe

#endif
