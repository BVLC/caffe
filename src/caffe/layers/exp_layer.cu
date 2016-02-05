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

template <typename Dtype>
void ExpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_mul(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != Dtype(1)) {
    caffe_gpu_scal(count, inner_scale_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpLayer);


}  // namespace caffe
