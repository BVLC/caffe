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


INSTANTIATE_LAYER_GPU_FUNCS(LogLayer);

}  // namespace caffe
