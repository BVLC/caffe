
// Loop convolution engine for convnet
#ifndef CAFFE_LOOP_CONVOLUTION_HPP_
#define CAFFE_LOOP_CONVOLUTION_HPP_

namespace caffe {

/**
 * @brief Forwardprop to output maps
 *
 * The shapes of arrays of `bottom`, `weight` and `top` should be 
 * (channels, bottom_h, bottom_w), (num_output, channels, weight_h, weight_w)
 * and (num_output, out_h, out_w) respectively and they should be arranged
 * in C-order (row-major). Spatial shape of top map will be calculated from
 * other arguments.
 *
 * @param[in] bottom Bottom maps 
 * @param[in] weight Convolutional weight
 * @param[in,out] top Top maps accumulated
 *
 * Note that data bottom `top` will not be initialized with 0 in this
 * function for the purpose of enabling us to accumulate output
 * by multiple call of this function.
 */
template <typename Dtype>
void conv_top(
    const Dtype* bottom, const Dtype* weight, Dtype* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);

/**
 * @brief Backprop to weight parameter
 *
 * @param[in] bottom Bottom maps
 * @param[in,out] weight Gradient w.r.t. weight accumulated
 * @param[in] top Gradient w.r.t. top layer map
 * @see conv_top
 */
template <typename Dtype>
void conv_weight(
    const Dtype* bottom, Dtype* weight, const Dtype* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);

/**
 * @brief Backprop to bottom maps
 *
 * @param[in,out] bottom Gradient w.r.t. bottom map accumulated
 * @param[in] weight Convolutional weight
 * @param[in] top Gradient w.r.t. top map
 * @see conv_top
 */
template <typename Dtype>
void conv_bottom(
    Dtype* bottom, const Dtype* weight, const Dtype* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w, int pad_h,
    int pad_w, int stride_h, int stride_w);

// end of namespace caffe
}

#endif
