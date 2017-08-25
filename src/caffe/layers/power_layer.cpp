#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  diff_scale_ = power_  * scale_;
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_set(count, value, top_data);
    return;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_powx(count, top_data, power_, top_data);
  }
}


#ifdef CPU_ONLY
STUB_GPU(PowerLayer);
#endif

INSTANTIATE_CLASS(PowerLayer);
REGISTER_LAYER_CLASS(Power);

}  // namespace caffe
