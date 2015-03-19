#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void AugmentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AugmentParameter augment_param = this->layer_param_.augment_param();
  if (augment_param.has_crop_size()) {
    crop_width_ = augment_param.crop_size();
    crop_height_ = augment_param.crop_size();
  } else if (this->layer_param_.augment_param().has_crop_width()) {
    crop_width_ = augment_param.crop_width();
    crop_height_ = augment_param.crop_height();
  } else {
    crop_width_ = bottom[0]->width();
    crop_height_ = bottom[0]->height();
  }
  CHECK_LT(0, crop_width_) <<
      "Crop width must be greater than 0.";
  CHECK_LT(0, crop_height_) <<
    "Crop height must be greater than 0.";
  CHECK_GE(bottom[0]->width(), crop_width_) <<
      "Crop width must be less than or equal to input width.";
  CHECK_GE(bottom[0]->height(), crop_height_) <<
    "Crop height must be less than or equal to input height.";
  mirror_ = this->layer_param_.augment_param().mirror();
}

template <typename Dtype>
void AugmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  num_ = bottom[0]->num();
  input_channels_ = bottom[0]->channels();
  input_height_ = bottom[0]->height();
  input_width_ = bottom[0]->width();
  label_channels_ = bottom[1]->channels();
  label_height_ = bottom[1]->height();
  label_width_ = bottom[1]->width();
  label_crop_height_ = (int)(label_height_ *
      (crop_height_ * 1.0 / input_height_));
  label_crop_width_ = (int)(label_width_ *
      (crop_width_ * 1.0 / input_width_));

  top[0]->Reshape(num_, input_channels_, crop_height_, crop_width_);
  top[1]->Reshape(num_, label_channels_, label_crop_height_, label_crop_width_);
  mirror_image_vec_.Reshape(num_, 1, 1, 1);
  h_shift_vec_.Reshape(num_, 1, 1, 1);
  w_shift_vec_.Reshape(num_, 1, 1, 1);
}

template <typename Dtype>
void AugmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* input_top_data = top[0]->mutable_cpu_data();
  const Dtype* input_bottom_data = bottom[0]->cpu_data();
  Dtype* label_top_data = top[1]->mutable_cpu_data();
  const Dtype* label_bottom_data = bottom[1]->cpu_data();
  float* mirror_image = mirror_image_vec_.mutable_cpu_data();
  float* h_shift = h_shift_vec_.mutable_cpu_data();
  float* w_shift = w_shift_vec_.mutable_cpu_data();

  if (mirror_) {
    caffe_rng_uniform(num_, 0.0f, 1.0f, mirror_image);
  }
  if (crop_width_ < input_width_) {
    // Need to crop width.
    // In 0-1 range for reuse with labels.
    caffe_rng_uniform(num_, 0.0f, 1.0f, w_shift);
  }
  if (crop_height_ < input_height_) {
    // Need to crop height.
    caffe_rng_uniform(num_, 0.0f, 1.0f, h_shift);
  }

  // Input crop/mirroring.
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < input_channels_; ++c) {
      for (int h = 0; h < crop_height_; ++h) {
        int h_on = h + (int)(h_shift[n] * (input_height_ - crop_height_ + 1));
        int w_end = crop_width_ - 1 +
            (int)(w_shift[n] * (input_width_ - crop_width_ + 1));
        for (int w = 0; w < crop_width_; ++w) {
          int w_on = w + (int)(w_shift[n] * (input_width_ - crop_width_ + 1));
          if (mirror_ && mirror_image[n] > .5) {
            input_top_data[h * crop_width_ + w] =
                input_bottom_data[h_on * input_width_ + w_end - w_on];
          } else {
            input_top_data[h * crop_width_ + w] =
                input_bottom_data[h_on * input_width_ + w_on];
          }
        }
      }
      input_bottom_data += input_width_ * input_height_;
      input_top_data += crop_width_ * crop_height_;
    }
  }

  // Label crop/mirroring.
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < label_channels_; ++c) {
      for (int h = 0; h < label_crop_height_; ++h) {
        int h_on = h + (int)(h_shift[n] *
            (label_height_ - label_crop_height_ + 1));
        int w_end = label_crop_width_ - 1 +
            (int)(w_shift[n] * (label_width_ - label_crop_width_ + 1));
        for (int w = 0; w < label_crop_width_; ++w) {
          int w_on = w + (int)(w_shift[n] *
                (label_width_ - label_crop_width_ + 1));
          if (mirror_ && mirror_image[n] > .5) {
            label_top_data[h * label_crop_width_ + w] =
                label_bottom_data[h_on * label_width_ + w_end - w_on];
          } else {
            label_top_data[h * label_crop_width_ + w] =
                label_bottom_data[h_on * label_width_ + w_on];
          }
        }
      }
      label_bottom_data += label_width_ * label_height_;
      label_top_data += label_crop_width_ * label_crop_height_;
    }
  }
}
template <typename Dtype>
void AugmentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_data());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AugmentLayer);
#endif

INSTANTIATE_CLASS(AugmentLayer);
REGISTER_LAYER_CLASS(Augment);

}  // namespace caffe
