#ifndef _CAFFEINE_UTIL__IM2COL_HPP_
#define _CAFFEINE_UTIL_IM2COL_HPP_

namespace caffeine {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int stride,
    Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int psize, const int stride,
    Dtype* data_im);



}  // namespace caffeine

#endif  // CAFFEINE_UTIL_IM2COL_HPP_
