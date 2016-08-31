#include <vector>

#include "caffe/layers/reverse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* src = bottom[0]->gpu_data();

  const int count = top[0]->count();
  const int axis_count = top[0]->count(axis_);
  const int copy_amount
        = (axis_ + 1 == top[0]->num_axes()) ? 1 : top[0]->count(axis_ + 1);
  const int num_fix = (axis_ > 0) ? count / axis_count : 1;
  const int sub_iter_max = top[0]->shape(axis_);

  for (int fix = 0; fix < num_fix; ++fix) {
    Dtype* target = top[0]->mutable_gpu_data()
            + (fix + 1) * copy_amount * sub_iter_max - copy_amount;
    for (int i = 0; i < sub_iter_max; ++i) {
      caffe_copy(copy_amount, src, target);
      src += copy_amount;     // normal order
      target -= copy_amount;  // reverse order
    }
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  Dtype* target = bottom[0]->mutable_gpu_diff();

  const int count = top[0]->count();
  const int axis_count = top[0]->count(axis_);
  const int copy_amount
        = (axis_ + 1 == top[0]->num_axes()) ? 1 : top[0]->count(axis_ + 1);
  const int num_fix = (axis_ > 0) ? count / axis_count : 1;
  const int sub_iter_max = top[0]->shape(axis_);

  for (int fix = 0; fix < num_fix; ++fix) {
    const Dtype* src = top[0]->gpu_diff()
            + (fix + 1) * copy_amount * sub_iter_max - copy_amount;
    for (int i = 0; i < sub_iter_max; ++i) {
      caffe_copy(copy_amount, src, target);
      target += copy_amount;  // normal order
      src -= copy_amount;     // reverse order
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseLayer);


}  // namespace caffe
