#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int hole_h, const int hole_w,
    Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
