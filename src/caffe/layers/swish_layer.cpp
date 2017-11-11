#include <cmath>
#include <vector>

#include "caffe/layers/swish_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
inline Dtype swish(Dtype x, Dtype sigmoid_beta_x) {
  return x * sigmoid_beta_x;
}

template <typename Dtype>
void SwishLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  this->sigmoid_beta_x_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SwishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* sigmoid_beta_x_data = this->sigmoid_beta_x_.mutable_cpu_data();
  Dtype beta = this->layer_param_.swish_param().beta();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    sigmoid_beta_x_data[i] = sigmoid(beta * bottom_data[i]);
    top_data[i] = swish(bottom_data[i], sigmoid_beta_x_data[i]);
  }
}

template <typename Dtype>
void SwishLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* sigmoid_beta_x_data = this->sigmoid_beta_x_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype beta = this->layer_param_.swish_param().beta();
    for (int i = 0; i < count; ++i) {
      const Dtype swish_x = top_data[i];
      bottom_diff[i] = top_diff[i] * (beta * swish_x + sigmoid_beta_x_data[i]
          * (1. - beta * swish_x));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwishLayer);
#endif

INSTANTIATE_CLASS(SwishLayer);

}  // namespace caffe
