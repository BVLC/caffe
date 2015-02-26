#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  mask_channels_count_ = this->layer_param_.mask_param().mask_channels_count();
  CHECK_GE(mask_channels_count_, 0) <<
    "mask_channels_count should be >= 0";
}

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // Scale channels up, one set for each mask.
  int new_channels_ = channels_ * mask_channels_count_;
  top[0]->Reshape(num_, new_channels_, height_, width_);
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int picture_size = width_ * height_;
  int bottom_offset = picture_size;
  int top_offset = picture_size * mask_channels_count_;
  for (int i = 0; i < num_; ++i) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          Dtype true_mask_channel = bottom_data[h * width_ + w];
          for (int m = 0; m < mask_channels_count_; ++m) {
            if (m == true_mask_channel) {
              top_data[m * picture_size + h * width_ + w] = 1;
            } else {
              top_data[m * picture_size + h * width_ + w] = 0;
            }
          }
        }
      }
      bottom_data += bottom_offset;
      top_data += top_offset;
    }
  }
}
template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_data());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
