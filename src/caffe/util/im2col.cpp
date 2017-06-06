/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
#if 0
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
#else
  int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;

    const int hc0 = h_offset * dilation_h - pad_h;
    const int wc0 = w_offset * dilation_w - pad_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h + hc0;

      const int row_offset = (c * height_col + h) * width_col;
      const int srow_offset = (c_im * height + h_pad) * width;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w + wc0;
        if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width)))
          data_col[row_offset + w] = data_im[srow_offset + w_pad];
        else {
          data_col[row_offset + w] = 0.;
        }
      }
    }
  }
#endif
}

template <typename Dtype>
void im3d2col_cpu(const Dtype* data_im, const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // LOG(ERROR) << "image size: " << depth << ", " << height << ", " << width;
  // LOG(ERROR) << "kernel size: " << kernel_d << ", " << kernel_h << ", " << kernel_w;

  // Implicit dilated kernel size
  long dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  long dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  long dil_kernel_d = (kernel_d - 1) * dilation_d + 1;
  long height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  long width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  long depth_col = (depth + 2 * pad_d - dil_kernel_d) / stride_d + 1;
  long channels_col = channels * kernel_h * kernel_w * kernel_d;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (long c = 0; c < channels_col; ++c) {
    long w_offset = c % kernel_w;
    long h_offset = (c / kernel_w) % kernel_h;
    long d_offset = (c / kernel_w / kernel_h) % kernel_d;
    long c_im = c / kernel_h / kernel_w / kernel_d;
    for (int d = 0; d < depth_col; ++d) {
      long d_pad = d * stride_d - pad_d + d_offset * dilation_d;
      for (long h = 0; h < height_col; ++h) {
        long h_pad = h * stride_h - pad_h + h_offset * dilation_h;
        for (long w = 0; w < width_col; ++w) {
          long w_pad = w * stride_w - pad_w + w_offset * dilation_w;
          if (((unsigned long)h_pad < (unsigned long)height) &&
              ((unsigned long)w_pad < (unsigned long)width) &&
              ((unsigned long)d_pad < (unsigned long)depth)) {
            data_col[((c * depth_col + d) * height_col + h) * width_col + w] =
              data_im[((c_im * depth + d_pad) * height + h_pad) * width + w_pad];
          } else {
            data_col[((c * depth_col + d) * height_col + h) * width_col + w] = 0.;
          }
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);
template void im3d2col_cpu<float>(const float* data_im, const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    float* data_col);
template void im3d2col_cpu<double>(const double* data_im, const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    double* data_col);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_output) {
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(num_spatial_axes, 0);
  vector<int> d_iter(num_spatial_axes, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
#if 0
  caffe_set(height * width * channels, Dtype(0), data_im);
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
#else
  int dil_patch_h = (kernel_h - 1) * dilation_h + 1;
  int dil_patch_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  long chunk_len = kernel_h * kernel_w;

  caffe_set(height * width * channels, Dtype(0), data_im);

  #ifdef _OPENMP
  #pragma omp parallel for if (channels > 1)
  #endif 
  for (int idx = 0; idx < channels; ++idx) {
    for (int inner_idx = 0; inner_idx < chunk_len; ++inner_idx) {
      int c = idx * chunk_len + inner_idx;
      int w_offset = c % kernel_w;
      int h_offset = (c / kernel_w) % kernel_h;
      int c_im = c / kernel_h / kernel_w;

      const int hc0 = h_offset * dilation_h - pad_h;
      const int wc0 = w_offset * dilation_w - pad_w;
      for (int h = 0; h < height_col; ++h) {
        for (int w = 0; w < width_col; ++w) {
          int h_pad = h * stride_h + hc0;
          const int srow_offset = (c_im * height + h_pad) * width;
          const int row_offset = (c * height_col + h) * width_col;
          int w_pad = w * stride_w + wc0;
          if ((((unsigned)h_pad) < ((unsigned)height)) && (((unsigned)w_pad) < ((unsigned)width))) {
            data_im[srow_offset + w_pad] += data_col[row_offset + w];
          }
        }
      }
    }
  }
#endif
}

template <typename Dtype>
void col2im3d_cpu(const Dtype* data_col, const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  // Implicit dilated patch
  long dil_patch_h = (kernel_h - 1) * dilation_h + 1;
  long dil_patch_w = (kernel_w - 1) * dilation_w + 1;
  long dil_patch_d = (kernel_d - 1) * dilation_d + 1;
  long height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  long width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  long depth_col = (depth + 2 * pad_d - dil_patch_d) / stride_d + 1;
  long num_kernels = channels * height * width * depth;
  long chunk_len = kernel_h * kernel_w * kernel_d;

  caffe_set(num_kernels, Dtype(0), data_im);

  #ifdef _OPENMP
  #pragma omp parallel for if (channels > 1)
  #endif
  for (long c_im = 0; c_im < channels; ++c_im) {
    for (long c = c_im * chunk_len; c < chunk_len * (c_im + 1); ++c) {
      long w_offset = c % kernel_w;
      long h_offset = (c / kernel_w) % kernel_h;
      long d_offset = (c / kernel_w / kernel_h) % kernel_d;
 
      long dc0 = d_offset * dilation_d - pad_d;
      long hc0 = h_offset * dilation_h - pad_h;
      long wc0 = w_offset * dilation_w - pad_w;
      for (long d = 0; d < depth_col; ++d) {
        long d_pad = d * stride_d + dc0;
        for (long h = 0; h < height_col; ++h) {
          long h_pad = h * stride_h + hc0;
          for (long w = 0; w < width_col; ++w) {
            long w_pad = w * stride_w + wc0;

            if (((unsigned long)h_pad < (unsigned long)height) &&
                ((unsigned long)w_pad < (unsigned long)width) &&
                ((unsigned long)d_pad < (unsigned long)depth)) {
              data_im[((c_im * depth + d_pad) * height + h_pad) * width + w_pad] +=
                data_col[((c * depth_col + d) * height_col + h) * width_col + w];
            }
          }
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);
template void col2im3d_cpu<float>(const float* data_col, const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im3d_cpu<double>(const double* data_col, const int channels,
    const int depth, const int height, const int width,
    const int kernel_d, const int kernel_h, const int kernel_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int stride_d, const int stride_h, const int stride_w,
    const int dilation_d, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);


}  // namespace caffe
