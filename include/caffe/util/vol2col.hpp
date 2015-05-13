#ifndef _CAFFE_UTIL_VOL2COL_HPP_
#define _CAFFE_UTIL_VOL2COL_HPP_

namespace caffe {

template <typename Dtype>
void vol2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int depth,
    const int kernel_h, const int kernel_w, const int kernel_d,
    const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    Dtype* data_col);

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int depth, const int patch_h,
    const int patch_w, const int patch_d, const int pad_h, const int pad_w,
    const int pad_d, const int stride_h, const int stride_w, const int stride_d,
    Dtype* data_im);

template <typename Dtype>
void vol2col_gpu(const Dtype* data_im, const int channels, const int height,
    const int width, const int depth, const int kernel_h, const int kernel_w,
    const int kernel_d, const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    Dtype* data_col);

template <typename Dtype>
void col2vol_gpu(const Dtype* data_col, const int channels, const int height,
    const int width, const int depth, const int patch_h, const int patch_w,
    const int patch_d, const int pad_h, const int pad_w, const int pad_d,
    const int stride_h, const int stride_w, const int stride_d,
    Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_VOL2COL_HPP_
