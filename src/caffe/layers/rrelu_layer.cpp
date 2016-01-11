#include <algorithm>
#include <vector>

#include "caffe/layers/rrelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void RReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void RReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

  template <typename Dtype>
void RReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope_lower = this->layer_param_.rrelu_param().negative_slope_lower();
  Dtype negative_slope_upper = this->layer_param_.rrelu_param().negative_slope_upper();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    Dtype *mask = rand_vec_.mutable_cpu_data();
    caffe_rng_uniform(count,negative_slope_lower,negative_slope_upper,mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
        + Dtype(1) / mask[i] * std::min(bottom_data[i], Dtype(0));
  }
  }
  else
  {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
        + Dtype(2) / (negative_slope_lower+negative_slope_upper) * std::min(bottom_data[i], Dtype(0));
    }
  }
}

template <typename Dtype>
void RReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope_lower = this->layer_param_.rrelu_param().negative_slope_lower();
    Dtype negative_slope_upper = this->layer_param_.rrelu_param().negative_slope_upper();
    if (this->phase_ == TRAIN) {
      Dtype *mask = rand_vec_.mutable_cpu_data();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
                                        + Dtype(1) / mask[i] * (bottom_data[i] <= 0));

      }
    }
    else
    {
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
                                        + Dtype(2) / (negative_slope_lower+negative_slope_upper) * (bottom_data[i] <= 0));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(RReLULayer);
#endif

INSTANTIATE_CLASS(RReLULayer);
REGISTER_LAYER_CLASS(RReLU);

}  // namespace caffe
