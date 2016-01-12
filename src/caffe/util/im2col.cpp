#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype *data_im, int channels, int input_h, int input_w,
    int kernel_h, int kernel_w, int pad_h, int pad_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    Dtype *data_col) {
  const int output_h = (input_h + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (input_w + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = input_h * input_w;
  for (; channels--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (((unsigned) input_row) >= ((unsigned) input_h)) {
            for (int output_cols = output_w; output_cols; output_cols--)
              *(data_col++) = 0;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (((unsigned) input_col) < ((unsigned) input_w))
                *(data_col++) = data_im[input_row * input_w + input_col];
              else
                *(data_col++) = 0;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename Dtype>
void col2im_cpu(
    const Dtype *data_col, int channels, int input_h, int input_w,
    int kernel_h, int kernel_w, int pad_h, int pad_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    Dtype *data_im) {
  const int output_h = (input_h + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (input_w + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = input_h * input_w;
  caffe_set(channels * input_h * input_w, Dtype(0), data_im);
  for (; channels--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (((unsigned) input_row) >= ((unsigned) input_h)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (((unsigned) input_col) < ((unsigned) input_w))
                data_im[input_row * input_w + input_col] += *data_col;
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, int channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, int channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, double* data_col);
template void col2im_cpu<float>(const float* data_col, int channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, int channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, double* data_im);

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
