#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int_tp channels,
    const int_tp height, const int_tp width, const int_tp kernel_h, const int_tp kernel_w,
    const int_tp pad_h, const int_tp pad_w,
    const int_tp stride_h, const int_tp stride_w,
    Dtype* data_col) {
  int_tp height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int_tp channels_col = channels * kernel_h * kernel_w;
  for (int_tp c = 0; c < channels_col; ++c) {
    int_tp w_offset = c % kernel_w;
    int_tp h_offset = (c / kernel_w) % kernel_h;
    int_tp c_im = c / kernel_h / kernel_w;
    for (int_tp h = 0; h < height_col; ++h) {
      for (int_tp w = 0; w < width_col; ++w) {
        int_tp h_pad = h * stride_h - pad_h + h_offset;
        int_tp w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int_tp channels,
    const int_tp height, const int_tp width, const int_tp kernel_h, const int_tp kernel_w,
    const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
    const int_tp stride_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int_tp channels,
    const int_tp height, const int_tp width, const int_tp kernel_h, const int_tp kernel_w,
    const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
    const int_tp stride_w, double* data_col);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int_tp num_spatial_axes, const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    Dtype* data_output) {
  if (!im2col) {
    int_tp im_size = im_shape[0];
    for (int_tp i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int_tp kernel_size = 1;
  for (int_tp i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int_tp channels_col = col_shape[0];
  vector<int_tp> d_offset(num_spatial_axes, 0);
  vector<int_tp> d_iter(num_spatial_axes, 0);
  for (int_tp c = 0; c < channels_col; ++c) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int_tp offset = c;
    for (int_tp d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int_tp index_col = c;
      int_tp index_im = c / kernel_size;
      bool is_padding = false;
      for (int_tp d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int_tp d = d_iter[d_i];
        const int_tp d_pad = d * stride[d_i] - pad[d_i] + d_offset[d_i];
        is_padding |= d_pad < 0 || d_pad >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_pad;
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
      for (int_tp d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int_tp d_max = col_shape[d_i + 1];
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
  }  // for (int_tp c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int_tp num_spatial_axes,
    const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int_tp num_spatial_axes,
    const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int_tp num_spatial_axes,
    const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int_tp channels,
    const int_tp height, const int_tp width, const int_tp patch_h, const int_tp patch_w,
    const int_tp pad_h, const int_tp pad_w,
    const int_tp stride_h, const int_tp stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int_tp height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int_tp channels_col = channels * patch_h * patch_w;
  for (int_tp c = 0; c < channels_col; ++c) {
    int_tp w_offset = c % patch_w;
    int_tp h_offset = (c / patch_w) % patch_h;
    int_tp c_im = c / patch_h / patch_w;
    for (int_tp h = 0; h < height_col; ++h) {
      for (int_tp w = 0; w < width_col; ++w) {
        int_tp h_pad = h * stride_h - pad_h + h_offset;
        int_tp w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int_tp channels,
    const int_tp height, const int_tp width, const int_tp patch_h, const int_tp patch_w,
    const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
    const int_tp stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int_tp channels,
    const int_tp height, const int_tp width, const int_tp patch_h, const int_tp patch_w,
    const int_tp pad_h, const int_tp pad_w, const int_tp stride_h,
    const int_tp stride_w, double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int_tp num_spatial_axes,
    const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int_tp num_spatial_axes,
    const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int_tp num_spatial_axes,
    const int_tp* im_shape, const int_tp* col_shape,
    const int_tp* kernel_shape, const int_tp* pad, const int_tp* stride,
    double* data_im);


}  // namespace caffe
