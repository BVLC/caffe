#include <vector>

#include "caffe/layers/log_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (input_scale_ == Dtype(1) && input_shift_ == Dtype(0)) {
    caffe_gpu_log(count, bottom_data, top_data);
  } else {
    caffe_copy(count, bottom_data, top_data);
    if (input_scale_ != Dtype(1)) {
      caffe_gpu_scal(count, input_scale_, top_data);
    }
    if (input_shift_ != Dtype(0)) {
      caffe_gpu_add_scalar(count, input_shift_, top_data);
    }
    caffe_gpu_log(count, top_data, top_data);
  }
  if (base_scale_ != Dtype(1)) {
    caffe_gpu_scal(count, base_scale_, top_data);
  }
}

template <typename Dtype>
void LogLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, bottom_data, bottom_diff);
    if (input_scale_ != Dtype(1)) {
      caffe_gpu_scal(count, input_scale_, bottom_diff);
    }
    if (input_shift_ != Dtype(0)) {
      caffe_gpu_add_scalar(count, input_shift_, bottom_diff);
    }
    caffe_gpu_powx(count, bottom_diff, Dtype(-1), bottom_diff);
    if (backward_num_scale_ != Dtype(1)) {
      caffe_gpu_scal(count, backward_num_scale_, bottom_diff);
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(LogLayer);

}  // namespace caffe
