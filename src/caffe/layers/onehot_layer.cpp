#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_labels_ = this->layer_param_.onehot_param().num_labels();
  CHECK_GE(num_labels_, 1);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_GE(num_labels_, channels_)
      << "num_labels must be less than or equal to the number of channels.";
  top[0]->Reshape(num_, num_labels_, height_, width_);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
	const int channel_offset = height_ * width_;
	const Dtype* bottom_data = bottom[0]->cpu_data(n, 0, h, w);
	Dtype* top_data = top[0]->mutable_cpu_data(n, 0, h, w);
	for (int c = 0; c < channels_; ++c) {
	  const int label = bottom_data[c * channel_offset];
	  CHECK_LT(label, num_labels_);
	  CHECK_GE(label, -1);
	  // Ignore -1 label (it is just used as filler)
	  if (label >= 0) {
	    top_data[label * channel_offset] = Dtype(1);
	  }
	}
      }
    }
  }
}

INSTANTIATE_CLASS(OneHotLayer);

}  // namespace caffe
