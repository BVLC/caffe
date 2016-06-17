#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/attention_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftAttentionForward(const int nthreads, const int num,
    const int channels, const int spatial_dim, const Dtype* a_data,
    const Dtype* alpha_data, const Dtype* beta_data, Dtype* z_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels;
    int c = index % channels;
    Dtype sum = 0;
    Dtype beta = beta_data[n];
    for (int s = 0; s < spatial_dim; ++s) {
      sum += beta * a_data[(n * channels + c) * spatial_dim + s]
          * alpha_data[n * spatial_dim + s];
    }
    z_data[n * channels + c] = sum;
  }
}

template <typename Dtype>
__global__ void SoftAttentionGradientA(const int nthreads, const int num,
    const int channels, const int spatial_dim, const Dtype* alpha_data,
    const Dtype* beta_data, const Dtype* z_diff, Dtype* a_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / spatial_dim / channels;
    int c = (index / spatial_dim) % channels;
    int s = index % spatial_dim;
    Dtype beta = beta_data[n];
    a_diff[(n * channels + c) * spatial_dim + s] = beta *
        alpha_data[n * spatial_dim + s] * z_diff[n * channels + c];
  }
}

template <typename Dtype>
__global__ void SoftAttentionGradientAlpha(const int nthreads, const int num,
    const int channels, const int spatial_dim, const Dtype* a_data,
    const Dtype* beta_data, const Dtype* z_diff, Dtype* alpha_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    Dtype beta = beta_data[n];
    for (int c = 0; c < channels; ++c) {
      sum += beta * a_data[(n * channels + c) * spatial_dim + s]
          * z_diff[n * channels + c];
    }
    alpha_diff[n * spatial_dim + s] = sum;
  }
}

template <typename Dtype>
__global__ void SoftAttentionGradientBeta(const int nthreads, const int num,
    const int channels, const int spatial_dim, const Dtype* z_data,
    const Dtype* beta_data,const Dtype* z_diff, Dtype* beta_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index;
    Dtype sum = 0;
    Dtype beta = beta_data[n];
    for (int c = 0; c < channels; ++c) {
      sum += z_data[n * channels + c] / beta * z_diff[n * channels + c];
    }
    beta_diff[n] = sum;
  }
}

template <typename Dtype>
void SoftAttentionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* a_data = bottom[0]->gpu_data();
  const Dtype* alpha_data = bottom[1]->gpu_data();
  const Dtype* beta_data = bottom[2]->gpu_data();
  Dtype* z_data = top[0]->mutable_gpu_data();
  int nthreads = num_ * channels_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftAttentionForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_, channels_, spatial_dim_, a_data,
      alpha_data, beta_data, z_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SoftAttentionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] && !propagate_down[1] && !propagate_down[2]) {
    return;
  }
  const Dtype* a_data = bottom[0]->gpu_data();
  const Dtype* alpha_data = bottom[1]->gpu_data();
  const Dtype* beta_data = bottom[2]->gpu_data();
  const Dtype* z_data = top[0]->gpu_data();
  const Dtype* z_diff = top[0]->gpu_diff();
  Dtype* a_diff = bottom[0]->mutable_gpu_diff();
  Dtype* alpha_diff = bottom[1]->mutable_gpu_diff();
  Dtype* beta_diff = bottom[2]->mutable_gpu_diff();
  if (propagate_down[0]) {
    int nthreads = num_ * channels_ * spatial_dim_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftAttentionGradientA<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_, channels_, spatial_dim_,
        alpha_data, beta_data, z_diff, a_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  if (propagate_down[1]) {
    int nthreads = num_ * spatial_dim_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftAttentionGradientAlpha<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_, channels_, spatial_dim_,
        a_data, beta_data, z_diff, alpha_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  if (propagate_down[2]) {
    int nthreads = num_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftAttentionGradientBeta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, num_, channels_, spatial_dim_,
        z_data, beta_data, z_diff, beta_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftAttentionLayer);

}  // namespace caffe
