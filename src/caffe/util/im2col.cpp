#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, 
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_col) {
  // effective kernel if we expand the holes (trous)
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels_col; ++c) {
      int w_offset = (c % kernel_w)  * hole_w;
      int h_offset = ((c / kernel_w) % kernel_h) * hole_h;
      int c_im = c / kernel_w / kernel_h;
      for (int h = 0; h < height_col; ++h) {
	const int h_im = h * stride_h + h_offset - pad_h;
	for (int w = 0; w < width_col; ++w) {
	  const int w_im = w * stride_w + w_offset - pad_w;
	  data_col[((n * channels_col + c) * height_col + h) * width_col + w] =
	    (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) ?
	    data_im[((n * channels + c_im) * height + h_im) * width + w_im] : 
	    0.; // zero-pad
	}
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_im) {
  caffe_set(num * channels * height * width, Dtype(0), data_im);
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels_col; ++c) {
      int w_offset = (c % kernel_w)  * hole_w;
      int h_offset = ((c / kernel_w) % kernel_h) * hole_h;
      int c_im = c / kernel_w / kernel_h;
      for (int h = 0; h < height_col; ++h) {
	const int h_im = h * stride_h + h_offset - pad_h;
	for (int w = 0; w < width_col; ++w) {
	  const int w_im = w * stride_w + w_offset - pad_w;
	  if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
	    data_im[((n * channels + c_im) * height + h_im) * width + w_im] += 
	      data_col[((n * channels_col + c) * height_col + h) * width_col + w];
	  }
	}
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    double* data_im);

}  // namespace caffe
