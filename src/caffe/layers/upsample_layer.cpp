#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void UpsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UpsampleParameter upsample_param = this->layer_param_.upsample_param();
  CHECK((upsample_param.has_upsample_h() && upsample_param.has_upsample_w())
      || (!upsample_param.has_scale() && upsample_param.has_scale_h()
      && upsample_param.has_scale_w())
      || (!upsample_param.has_scale_h() && !upsample_param.has_scale_w()))
      << "upsample_h & upsample_w are required, else (DEPRECATED) "
      << "scale OR scale_h & scale_w are required.";

  if (upsample_param.has_upsample_h() && upsample_param.has_upsample_w()) {
    upsample_h_ = upsample_param.upsample_h();
    upsample_w_ = upsample_param.upsample_w();
    CHECK_GT(upsample_h_, 1);
    CHECK_GT(upsample_w_, 1);
  } else {
    LOG(INFO) << "Params 'pad_out_{}_' are deprecated. Please declare upsample"
        << " height and width useing the upsample_h, upsample_w parameters.";
    if (!upsample_param.has_scale_h()) {
      scale_h_ = scale_w_ = upsample_param.scale();
      CHECK_GT(scale_h_, 1);
    } else {
      scale_h_ = upsample_param.scale_h();
      scale_w_ = upsample_param.scale_w();
      CHECK_GT(scale_h_, 1);
      CHECK_GT(scale_w_, 1);
    }
    pad_out_h_ = upsample_param.pad_out_h();
    pad_out_w_ = upsample_param.pad_out_w();
    CHECK(!pad_out_h_ || scale_h_ == 2) 
        << "Output height padding compensation requires scale_h == 2, otherwise "
        << "the output size is ill-defined.";
    CHECK(!pad_out_w_ || scale_w_ == 2) 
        << "Output width padding compensation requires scale_w == 2, otherwise "
        << "the output size is ill-defined.";
    upsample_h_ = upsample_w_ = -1;  // flag to calculate in Reshape
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(4, bottom[1]->num_axes()) << "Input mask must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  if (bottom[0]->height() != bottom[1]->height())
    {
      LOG(ERROR) << "bottom[0]->height()=" << bottom[0]->height() << " / bottom[1]->height()=" << bottom[1]->height();
      CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    }
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  if (upsample_h_ <= 0 || upsample_w_ <= 0) {
    upsample_h_ = bottom[0]->height() * scale_h_ - int(pad_out_h_);
    upsample_w_ = bottom[0]->width() * scale_w_ - int(pad_out_w_);
  }
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), upsample_h_,
      upsample_w_);
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_mask_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // Initialize
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int i = 0; i < height_ * width_; ++i) {
        const int idx = static_cast<int>(bottom_mask_data[i]);
        if (idx >= upsample_h_ * upsample_w_) {
          // this can happen if the pooling layer that created the input mask
          // had an input with different size to top[0]
          LOG(FATAL) << "upsample top index " << idx << " out of range - "
            << "check scale settings match input pooling layer's "
            << "downsample setup";
        }
        top_data[idx] = bottom_data[i];
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      bottom_mask_data += bottom[1]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_mask_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    const int bottom_count = bottom[0]->count();
    caffe_set(bottom_count, Dtype(0), bottom_diff);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int i = 0; i < height_ * width_; ++i) {
          const int idx = static_cast<int>(bottom_mask_data[i]);
          if (idx >= height_ * width_ * scale_h_ * scale_w_) {
            // this can happen if the pooling layer that created
            // the input mask had an input with different size to top[0]
            LOG(FATAL) << "upsample top index " << idx << " out of range - "
              << "check scale settings match input pooling layer's downsample setup";
          }
          bottom_diff[i] = top_diff[idx];
        }
        // compute offset
        bottom_diff += bottom[0]->offset(0, 1);
        bottom_mask_data += bottom[1]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(UpsampleLayer);
#endif

INSTANTIATE_CLASS(UpsampleLayer);
REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe
