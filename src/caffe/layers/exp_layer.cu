#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (inner_scale_ == Dtype(1)) {
    caffe_gpu_exp(count, bottom_data, top_data);
  } else {
    caffe_gpu_scale(count, inner_scale_, bottom_data, top_data);
    caffe_gpu_exp(count, top_data, top_data);
  }
  if (outer_scale_ != Dtype(1)) {
    caffe_gpu_scal(count, outer_scale_, top_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpLayer);


}  // namespace caffe
