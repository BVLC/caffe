#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/util/math_functions.hpp"

namespace caffeine {

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    Dtype* scale) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values 
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < size + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* in,
    const Dtype* scale, const Dtype negative_beta, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
  LRNFillScale<<<CAFFEINE_GET_BLOCKS(n_threads), CAFFEINE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_, channels_, height_, width_, size_,
      alpha_ / size_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  LRNComputeOutput<<<CAFFEINE_GET_BLOCKS(n_threads), CAFFEINE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
Dtype LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
  return Dtype(0.);
}

INSTANTIATE_CLASS(LRNLayer);

}  // namespace caffeine
