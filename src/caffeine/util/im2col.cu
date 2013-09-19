#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffeine/common.hpp"
#include "caffeine/util/im2col.hpp"

namespace caffeine {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
  const int height, const int width, const int ksize,
  const int stride, const int height_col, const int width_col, Dtype* data_col) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride;
    int w_in = w_out * stride;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        *data_col = data_im[i * width + j];
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  im2col_gpu_kernel<<<CAFFEINE_GET_BLOCKS(num_kernels), CAFFEINE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, ksize, stride, height_col, width_col,
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    double* data_col);

/*
template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int stride,
    Dtype* data_im) {
  memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  int height_col = (height - ksize) / stride + 1;
  int width_col = (width - ksize) / stride + 1;
  int channels_col = channels * ksize * ksize;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        data_im[(c_im * height + h * stride + h_offset) * width + w * stride 
            + w_offset] += data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    double* data_im);
*/
}  // namespace caffeine
