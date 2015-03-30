#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/common.cuh"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;
    int h_im = h * stride_h - pad_h;
    int w_im = w * stride_w - pad_w;
    data_im += ((n * channels + c_im) * height + h_im) * width + w_im; // patch start
    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
	if (h_im + i * hole_h >= 0 && h_im + i * hole_h < height && w_im + j * hole_w >= 0 && w_im + j * hole_w < width) {
	  *data_col = data_im[(i * hole_h) * width + j * hole_w];
	}
	else {
	  *data_col = 0.;
	}
	data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_col) {
  // We are going to launch num * channels * height_col * width_col kernels, each
  // kernel responsible for copying a single block from a single image.
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = num * channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im,
      num, channels, height, width,
      kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, hole_h, hole_w,
      height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    float* data_col);
template void im2col_gpu<double>(const double* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;
    int h_im = h * stride_h - pad_h;
    int w_im = w * stride_w - pad_w;
    data_im += ((n * channels + c_im) * height + h_im) * width + w_im;
    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
	if (h_im + i * hole_h >= 0 && h_im + i * hole_h < height && w_im + j * hole_w >= 0 && w_im + j * hole_w < width) {
	  atomicAdd(&data_im[(i * hole_h) * width + j * hole_w], *data_col);
	}
	data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_im) {
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = num * channels * height_col * width_col;
  // We will use atomicAdd in the kernel to accumulate the values.
  caffe_gpu_set(num * channels * height * width, Dtype(0), data_im);
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col,
      num, channels, height, width,
      kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, hole_h, hole_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    float* data_im);
template void col2im_gpu<double>(const double* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    double* data_im);

}  // namespace caffe
