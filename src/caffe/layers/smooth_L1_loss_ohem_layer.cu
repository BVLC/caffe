// --------------------------------------------------------
// R-FCN
// Written by Yi Li, 2016.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/smooth_l1_loss_ohem_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  template <typename Dtype>
  __global__ void SmoothL1ForwardGPU(const int n, const Dtype* in, Dtype* out) {
    // f(x) = 0.5 * x^2    if |x| < 1
    //        |x| - 0.5    otherwise
    CUDA_KERNEL_LOOP(index, n) {
      Dtype val = in[index];
      Dtype abs_val = abs(val);
      if (abs_val < 1) {
        out[index] = 0.5 * val * val;
      } else {
        out[index] = abs_val - 0.5;
      }
    }
  }

  template <typename Dtype>
  __global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      Dtype sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += data[(n * channels + c) * spatial_dim + s];
      }
      channel_sum[index] = sum;
    }
  }

  template <typename Dtype>
  void SmoothL1LossOHEMLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
    if (has_weights_) {
      caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w * (b0 - b1)
    }
    SmoothL1ForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, diff_.gpu_data(),
      errors_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;


    Dtype loss;
    caffe_gpu_asum(count, errors_.gpu_data(), &loss);
    int spatial_dim = diff_.height() * diff_.width();

    Dtype pre_fixed_normalizer =
      this->layer_param_.loss_param().pre_fixed_normalizer();
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
      pre_fixed_normalizer);

    // Output per-instance loss
    if (top.size() >= 2) {
      kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(top[1]->count()),
        CAFFE_CUDA_NUM_THREADS >> > (outer_num_, bottom[0]->channels(),
        inner_num_, errors_.gpu_data(), top[1]->mutable_gpu_data());
    }
  }

  template <typename Dtype>
  __global__ void SmoothL1BackwardGPU(
    const int n, const Dtype* in, Dtype* out) {
    // f'(x) = x         if |x| < 1
    //       = sign(x)   otherwise
    CUDA_KERNEL_LOOP(index, n) {
      Dtype val = in[index];
      Dtype abs_val = abs(val);
      if (abs_val < 1) {
        out[index] = val;
      } else {
        out[index] = (Dtype(0) < val) - (val < Dtype(0));
      }
    }
  }

  template <typename Dtype>
  void SmoothL1LossOHEMLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    int count = diff_.count();
    SmoothL1BackwardGPU<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, diff_.gpu_data(),
      diff_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    for (int i = 0; i < 2; ++i) {
      if (propagate_down[i]) {
        const Dtype sign = (i == 0) ? 1 : -1;
        int spatial_dim = diff_.height() * diff_.width();

        Dtype pre_fixed_normalizer =
          this->layer_param_.loss_param().pre_fixed_normalizer();
        Dtype normalizer = get_normalizer(normalization_, pre_fixed_normalizer);
        Dtype alpha = sign * top[0]->cpu_diff()[0] / normalizer;

        caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
      }
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossOHEMLayer);

}  // namespace caffe
