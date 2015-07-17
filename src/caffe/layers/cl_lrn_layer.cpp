#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

extern "C" const char _cl_lrn_layer_start;
extern "C" const char _cl_lrn_layer_end;

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  cl_uint argIdx = 0;
  int n_threads = num_ * height_ * width_;

  ClState& state = Caffe::cl_state();
  state.submit_program("lrn", &_cl_lrn_layer_start, &_cl_lrn_layer_end);

  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  ClKernel kernel_fill_scale = state.get_kernel("LRNFillScale");
  kernel_fill_scale.set_arg(argIdx++, n_threads);
  kernel_fill_scale.set_arg(argIdx++, bottom_data);
  kernel_fill_scale.set_arg(argIdx++, num_);
  kernel_fill_scale.set_arg(argIdx++, channels_);
  kernel_fill_scale.set_arg(argIdx++, height_);
  kernel_fill_scale.set_arg(argIdx++, width_);
  kernel_fill_scale.set_arg(argIdx++, size_);
  kernel_fill_scale.set_arg(argIdx++, alpha_ / size_);
  kernel_fill_scale.set_arg(argIdx++, k_);
  kernel_fill_scale.set_arg(argIdx++, scale_data);
  kernel_fill_scale.enqueue(n_threads);

  argIdx = 0;
  n_threads = bottom[0]->count();
  ClKernel kernel_compute_output = state.get_kernel("LRNComputeOutput");
  kernel_compute_output.set_arg(argIdx++, n_threads);
  kernel_compute_output.set_arg(argIdx++, bottom_data);
  kernel_compute_output.set_arg(argIdx++, scale_data);
  kernel_compute_output.set_arg(argIdx++, -beta_);
  kernel_compute_output.set_arg(argIdx++, top_data);
  kernel_compute_output.enqueue(n_threads);
}

template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      CrossChannelForward_gpu(bottom, top);
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      WithinChannelForward(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  cl_uint argIdx = 0;
  cl_int err;
  int n_threads = num_ * height_ * width_;

  ClState& state = Caffe::cl_state();
  state.submit_program("lrn", &_cl_lrn_layer_start, &_cl_lrn_layer_end);

  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  ClKernel kernel = state.get_kernel("LRNComputeDiff");
  kernel.set_arg(argIdx++, n_threads);
  kernel.set_arg(argIdx++, bottom[0]->gpu_data());
  kernel.set_arg(argIdx++, top[0]->gpu_data());
  kernel.set_arg(argIdx++, scale_.gpu_data());
  kernel.set_arg(argIdx++, top[0]->gpu_diff());
  kernel.set_arg(argIdx++, num_);
  kernel.set_arg(argIdx++, channels_);
  kernel.set_arg(argIdx++, height_);
  kernel.set_arg(argIdx++, width_);
  kernel.set_arg(argIdx++, size_);
  kernel.set_arg(argIdx++, -beta_);
  kernel.set_arg(argIdx++, Dtype(2. * alpha_ * beta_ / size_));
  kernel.set_arg(argIdx++, bottom[0]->mutable_gpu_diff());
  kernel.enqueue(n_threads);
}

template <typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      CrossChannelBackward_gpu(top, propagate_down, bottom);
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      WithinChannelBackward(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL) << "Unknown normalization region.";
  }
}

// float instantiation
template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);

// double instantiation
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);

INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
#endif  // USE_OCL
