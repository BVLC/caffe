#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void IgnoreOverlayLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  IgnoreOverlayParameter param = this->layer_param_.ignore_overlay_param();
  ignore_label_ = param.ignore_label();
}

template <typename Dtype>
void IgnoreOverlayLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //
  CHECK_EQ(bottom[1]->num(), num_) << "Incompatible inputs";
  CHECK_EQ(bottom[1]->channels(), channels_) << "Incompatible inputs";
  CHECK_EQ(bottom[1]->height(), height_) << "Incompatible inputs";
  CHECK_EQ(bottom[1]->width(), width_) << "Incompatible inputs";
  //
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void IgnoreOverlayLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_copy(bottom[1]->count(), bottom[1]->cpu_data(), top[0]->mutable_cpu_data());
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    const int value = bottom_data[i];
    if (value == ignore_label_) {
      top_data[i] = static_cast<Dtype>(value);
    }
  }
}

#ifdef CPU_ONLY
//STUB_GPU(IgnoreOverlayLayer);
#endif

INSTANTIATE_CLASS(IgnoreOverlayLayer);

}  // namespace caffe
