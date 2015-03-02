#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ReshapeParameter& reshape_param = this->layer_param_.reshape_param();
  channels_ = reshape_param.channels() > 0 ? reshape_param.channels() : bottom[0]->channels();
  height_ = reshape_param.height() > 0 ? reshape_param.height() : bottom[0]->height();
  width_ = reshape_param.width() > 0 ? reshape_param.width() : bottom[0]->width();
  count_ = bottom[0]->num() * channels_ * height_ * width_;
  CHECK_EQ(count_, bottom[0]->count());
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
  CHECK_EQ(bottom[0]->num(), top[0]->num());
  CHECK_EQ(channels_, top[0]->channels());
  CHECK_EQ(height_, top[0]->height());
  CHECK_EQ(width_, top[0]->width());
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

#ifdef CPU_ONLY
STUB_GPU(ReshapeLayer);
#endif

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
