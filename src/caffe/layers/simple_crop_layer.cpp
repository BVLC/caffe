#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// take the first bottom and crop its height and width equal to the second bottom
// removed rows/cols are take from the bottom/right respectively
template <typename Dtype>
void SimpleCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	crop_h_ = crop_w_ = 0;
}

template <typename Dtype>
void SimpleCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->height(), bottom[1]->height()) << "SimpleCropLayer cannot crop height of " <<
    bottom[0]->height() << " to " << bottom[1]->height();
  CHECK_GE(bottom[0]->width(), bottom[1]->width()) << "SimpleCropLayer cannot crop width of " <<
    bottom[0]->width() << " to " << bottom[1]->width();

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(),
      bottom[1]->width());
}

template <typename Dtype>
void SimpleCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = 0; h < top[0]->height(); ++h) {
        caffe_copy(top[0]->width(),
            bottom_data + bottom[0]->offset(n, c, crop_h_ + h, crop_w_),
            top_data + top[0]->offset(n, c, h));
      }
    }
  }
}

template <typename Dtype>
void SimpleCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, crop_h_ + h, crop_w_));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SimpleCropLayer);
#endif

INSTANTIATE_CLASS(SimpleCropLayer);
REGISTER_LAYER_CLASS(SimpleCrop);

}  // namespace caffe
