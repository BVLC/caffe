#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_gpu_set(count, value, top_data);
    return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_gpu_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_gpu_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_gpu_powx(count, top_data, power_, top_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PowerLayer);


}  // namespace caffe
