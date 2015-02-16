#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  PReLUParameter prelu_param = this->layer_param().prelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = prelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, channels));
    }
    caffe_set<Dtype>(this->blobs_[0]->count(),
      (Dtype)(prelu_param.init_value()), this->blobs_[0]->mutable_cpu_data());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
      << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
      << "Nagative slope size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(1, 1, 1, bottom[0]->count() / bottom[0]->num());
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void PReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int hw = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  if (channel_shared_) {
    // Channel shared variant
    const Dtype slope = *slope_data;
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
      + slope * std::min(bottom_data[i], Dtype(0));
    }
  } else {
    for (int i = 0; i < count; ++i) {
      int c = (i / hw) % channels;
      top_data[i] = std::max(bottom_data[i], Dtype(0))
      + slope_data[c] * std::min(bottom_data[i], Dtype(0));
    }
  }
}

template <typename Dtype>
void PReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int hw = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  if (propagate_down[0]) {
    const Dtype* slope_data = this->blobs_[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (channel_shared_) {
      // Channel shared variant
      const Dtype slope = *slope_data;
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope * (bottom_data[i] <= 0));
      }
    } else {
      for (int i = 0; i < count; ++i) {
        int c = (i / hw) % channels;
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
      }
    }
  }
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), slope_diff);
    if (channel_shared_) {
      // Channel shared variant
      Dtype slope = 0;
      for (int i = 0; i < count; ++i) {
        slope += top_diff[i] * bottom_data[i] * (bottom_data[i] <= 0);
      }
      *slope_diff = slope;
    } else {
      for (int i = 0; i < count; ++i) {
        int c = (i / hw) % channels;
        slope_diff[c] += top_diff[i] * bottom_data[i] * (bottom_data[i] <= 0);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PReLULayer);
#endif

INSTANTIATE_CLASS(PReLULayer);
REGISTER_LAYER_CLASS(PReLU);

}  // namespace caffe
