
#ifndef CAFFE_LOOP_CONVOLUTION_HPP_
#define CAFFE_LOOP_CONVOLUTION_HPP_

namespace caffe {

/*
 * @brief 
 */
template <typename Dtype>
void conv_filter(
    const Dtype* in, const Dtype* kernel, Dtype* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);

/*
 * @brief 
 */
template <typename Dtype>
void conv_weight(
    const Dtype* in, Dtype* kernel, const Dtype* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);

/*
 * @brief 
 */
template <typename Dtype>
void conv_image(
    Dtype* in, const Dtype* kernel, const Dtype* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w, int pad_h,
    int pad_w, int stride_h, int stride_w);

// end of namespace caffe
}

#endif
