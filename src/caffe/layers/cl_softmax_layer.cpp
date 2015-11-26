#ifdef USE_OCL
#include <string>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

extern "C" const char _cl_softmax_layer_start;
extern "C" const char _cl_softmax_layer_end;

namespace caffe {


template<typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  ClState& state = Caffe::cl_state();
  state.submit_program("softmax", &_cl_softmax_layer_start,
      &_cl_softmax_layer_end);

  ClKernel kernel_channel_max = state.get_kernel("kernel_channel_max");
  kernel_channel_max.set_arg(0, num);
  kernel_channel_max.set_arg(1, channels);
  kernel_channel_max.set_arg(2, spatial_dim);
  kernel_channel_max.set_arg(3, top_data);
  kernel_channel_max.set_arg(4, scale_data);
  kernel_channel_max.enqueue(num * spatial_dim);

  // subtract
  ClKernel kernel_channel_subtract = state.get_kernel(
      "kernel_channel_subtract");
  kernel_channel_subtract.set_arg(0, num);
  kernel_channel_subtract.set_arg(1, channels);
  kernel_channel_subtract.set_arg(2, spatial_dim);
  kernel_channel_subtract.set_arg(3, top_data);
  kernel_channel_subtract.set_arg(4, scale_data);
  kernel_channel_subtract.enqueue(num * spatial_dim);

  // exponentiate
  ClKernel kernel_exp = state.get_kernel("kernel_exp");
  kernel_exp.set_arg(0, num * channels * spatial_dim);
  kernel_exp.set_arg(1, top_data);
  kernel_exp.set_arg(2, top_data);
  kernel_exp.enqueue(num * channels * spatial_dim);

  // sum after exp
  ClKernel kernel_channel_sum = state.get_kernel("kernel_channel_sum");
  kernel_channel_sum.set_arg(0, num);
  kernel_channel_sum.set_arg(1, channels);
  kernel_channel_sum.set_arg(2, spatial_dim);
  kernel_channel_sum.set_arg(3, top_data);
  kernel_channel_sum.set_arg(4, scale_data);
  kernel_channel_sum.enqueue(num * spatial_dim);

  // divide
  ClKernel kernel_channel_div = state.get_kernel("kernel_channel_div");
  kernel_channel_div.set_arg(0, num);
  kernel_channel_div.set_arg(1, channels);
  kernel_channel_div.set_arg(2, spatial_dim);
  kernel_channel_div.set_arg(3, top_data);
  kernel_channel_div.set_arg(4, scale_data);
  kernel_channel_div.enqueue(num * spatial_dim);
}

template<typename Dtype>
void SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = top[0]->num();
  int channels = top[0]->channels();
  int spatial_dim = top[0]->height() * top[0]->width();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);

  ClState& state = Caffe::cl_state();
  state.submit_program("softmax", &_cl_softmax_layer_start,
      &_cl_softmax_layer_end);

  ClKernel kernel_channel_dot = state.get_kernel("kernel_channel_dot");
  kernel_channel_dot.set_arg(0, num);
  kernel_channel_dot.set_arg(1, channels);
  kernel_channel_dot.set_arg(2, spatial_dim);
  kernel_channel_dot.set_arg(3, top_diff);
  kernel_channel_dot.set_arg(4, top_data);
  kernel_channel_dot.set_arg(5, scale_data);
  kernel_channel_dot.enqueue(num * spatial_dim);

  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff

  ClKernel kernel_channel_subtract = state.get_kernel(
      "kernel_channel_subtract");

  kernel_channel_subtract.set_arg(0, num);
  kernel_channel_subtract.set_arg(1, channels);
  kernel_channel_subtract.set_arg(2, spatial_dim);
  kernel_channel_subtract.set_arg(3, bottom_diff);
  kernel_channel_subtract.set_arg(4, scale_data);
  kernel_channel_subtract.enqueue(num * spatial_dim);

  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);

}  // namespace caffe
#endif  // USE_OCL
