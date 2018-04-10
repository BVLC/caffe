#ifndef CAFFE_CAFFE_CONV_LAYER_HPP_
#define CAFFE_CAFFE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template<typename Dtype, typename MItype, typename MOtype>
class CaffeConvolutionLayer
    : public BaseConvolutionLayer<Dtype, MItype, MOtype> {
 public:
  explicit CaffeConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype, MItype, MOtype>(param) {
  }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output,
                        bool skip_im2col = false,
                        const QuantizerValues* const input_quant = nullptr,
                        const QuantizerValues* const weights_quant = nullptr,
                        const QuantizerValues* const output_quant = nullptr);
  void forward_cpu_bias(Dtype* output, const Dtype* bias,
                        const QuantizerValues* const output_quant = nullptr,
                        const QuantizerValues* const bias_quant = nullptr);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
                         Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(vptr<const Dtype> col_input,
                        vptr<const Dtype> weights, vptr<Dtype> output,
                        bool skip_im2col = false,
                        const QuantizerValues* const input_quant = nullptr,
                        const QuantizerValues* const weights_quant = nullptr,
                        const QuantizerValues* const output_quant = nullptr);
  void forward_gpu_bias(vptr<Dtype> output,
                        vptr<const Dtype> bias,
                        const QuantizerValues* const output_quant = nullptr,
                        const QuantizerValues* const bias_quant = nullptr);
  void backward_gpu_gemm(vptr<const Dtype> input, vptr<const Dtype> weights,
                         vptr<Dtype> col_output);
  void weight_gpu_gemm(vptr<const Dtype> col_input, vptr<const Dtype> output,
                       vptr<Dtype> weights);
  void backward_gpu_bias(vptr<Dtype> bias, vptr<const Dtype> input);
#endif

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff,
                              const QuantizerValues* const data_quant) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      im2col_cpu(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1],
          this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0],
          this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0],
          this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0],
          this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0],
          this->dilation_.cpu_data()[1], col_buff, data_quant);
    } else {
      im2col_nd_cpu(data, this->num_spatial_axes_,
          this->conv_input_shape_.cpu_data(),
          this->col_buffer_shape_.data(),
          this->kernel_shape_.cpu_data(),
          this->pad_.cpu_data(), this->stride_.cpu_data(),
          this->dilation_.cpu_data(), col_buff, data_quant);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1],
          this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0],
          this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, this->num_spatial_axes_,
          this->conv_input_shape_.cpu_data(),
          this->col_buffer_shape_.data(), this->kernel_shape_.cpu_data(),
          this->pad_.cpu_data(), this->stride_.cpu_data(),
          this->dilation_.cpu_data(), data);
    }
  }

#ifndef CPU_ONLY
  inline void conv_im2col_gpu(vptr<const Dtype> data, vptr<Dtype> col_buff,
                              const QuantizerValues* const data_quant) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      this->device_->im2col(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1],
          this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0],
          this->dilation_.cpu_data()[1], col_buff, data_quant);
    } else {
      this->device_->im2col_nd(data, this->num_spatial_axes_,
          this->num_kernels_im2col_,
          this->conv_input_shape_.gpu_data(),
          this->col_buffer_.gpu_shape(),
          this->kernel_shape_.gpu_data(), this->pad_.gpu_data(),
          this->stride_.gpu_data(), this->dilation_.gpu_data(), col_buff,
          data_quant);
    }
  }

  inline void conv_col2im_gpu(vptr<const Dtype> col_buff, vptr<Dtype> data) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      this->device_->col2im(col_buff, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1],
          this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], data);
    } else {
      this->device_->col2im_nd(col_buff, this->num_spatial_axes_,
          this->num_kernels_col2im_, this->conv_input_shape_.gpu_data(),
          this->col_buffer_.gpu_shape(),
          this->kernel_shape_.gpu_data(), this->pad_.gpu_data(),
          this->stride_.gpu_data(), this->dilation_.gpu_data(), data);
    }
  }
#endif  // !CPU_ONLY
};


}  // namespace caffe

#endif  // CAFFE_CAFFE_CONV_LAYER_HPP_
