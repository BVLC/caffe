#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int rand_count = bottom[0]->num() * bottom[0]->channels();
  const int channel_size = bottom[0]->width() * bottom[0]->height();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(rand_count, 1. - threshold_, mask);
    for (int i = 0; i < rand_count; i++) {
      const unsigned int m = mask[i];
      const Dtype* bottom_channel_data = bottom_data + channel_size * i;
      Dtype* top_channel_data = top_data + channel_size * i;
      for (int j = 0; j < channel_size; j++) {
        top_channel_data[j] = bottom_channel_data[j] * m * scale_;
      }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int rand_count = bottom[0]->num() * bottom[0]->channels();
      const int channel_size = bottom[0]->width() * bottom[0]->height();
      for (int i = 0; i < rand_count; i++) {
        const unsigned int m = mask[i];
        const Dtype* top_diff_data = top_diff + channel_size * i;
        Dtype* bottom_diff_data = bottom_diff + channel_size * i;
        for (int j = 0; j < channel_size; j++) {
          bottom_diff_data[j] = top_diff_data[j] * m * scale_;
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SpatialDropoutLayer);
#endif

INSTANTIATE_CLASS(SpatialDropoutLayer);
REGISTER_LAYER_CLASS(SpatialDropout);

}  // namespace caffe
