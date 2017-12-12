#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim, Dtype epsilon,
                                const Dtype* data, Dtype* norm_data) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    norm_data[index] = sum + epsilon;
  }
}

template <typename Dtype>
__global__ void kernel_channel_scale(const int num, const int channels, const int spatial_dim,
                                     const Dtype* data, const Dtype* norm_data,
                                     Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    output_data[index] = data[index] * norm_data[n * spatial_dim + s];
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) const {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Blob<Dtype> squared;
  squared.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* square_data = squared.mutable_gpu_data();
  Blob<Dtype> norm;
  if (top.size() != 2) {
    norm.Reshape(bottom[0]->num(), 1,
                   bottom[0]->height(), bottom[0]->width());
  }
  Dtype* norm_data = (top.size() == 2) ? top[1]->mutable_gpu_data() : norm.mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (normalize_type_ == "L2") {
    caffe_gpu_powx(num*channels*spatial_dim, bottom_data, Dtype(2), square_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-12, square_data, norm_data);
    caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(-0.5), norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data);
  }
  else if (normalize_type_ == "L1") {
    caffe_gpu_abs(num*channels*spatial_dim, bottom_data, square_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(num*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, 1e-6, square_data, norm_data);
    caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(-1), norm_data);
    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, bottom_data, norm_data, top_data);
  }
  else {
    NOT_IMPLEMENTED;
  }
}



INSTANTIATE_LAYER_GPU_FUNCS_CONST(NormalizeLayer);


}  // namespace caffe
