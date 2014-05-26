// Copyright 2014 BVLC and contributors.

#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im);

template <typename Dtype>
inline void im2col(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      im2col_cpu(data_im, channels, height, width, ksize, pad, stride,
                 data_col);
    case Caffe::GPU:
      im2col_gpu(data_im, channels, height, width, ksize, pad, stride,
                 data_col);
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
inline void col2im(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      im2col_cpu(data_col, channels, height, width, psize, pad, stride,
                 data_im);
    case Caffe::GPU:
      im2col_gpu(data_col, channels, height, width, psize, pad, stride,
                 data_im);
    default:
      LOG(FATAL) << "Unknown caffe mode.";
  }
}

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
