#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/caffe_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::forward_cpu_gemm(
                                                   const Dtype* input,
                                                   const Dtype* weights,
                                                   Dtype* output,
                                                   bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  for (int_tp g = 0; g < this->group_; ++g) {
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          this->conv_out_channels_ / this->group_, this->conv_out_spatial_dim_,
          this->kernel_dim_, (Dtype) 1., weights + this->weight_offset_ * g,
          col_buff + this->col_offset_ * g, (Dtype) 0.,
          output + this->output_offset_ * g);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::forward_cpu_bias(
                                                   Dtype* output,
                                                   const Dtype* bias) {
  caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
                        this->out_spatial_dim_, 1, (Dtype) 1., bias,
                        this->bias_multiplier_.cpu_data(), (Dtype) 1., output);
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::backward_cpu_gemm(
                                                    const Dtype* output,
                                                    const Dtype* weights,
                                                    Dtype* input) {
  Dtype* col_buff = this->col_buffer_.mutable_cpu_data();
  if (this->is_1x1_) {
    col_buff = input;
  }
  for (int_tp g = 0; g < this->group_; ++g) {
    caffe_gemm<Dtype>(CblasTrans, CblasNoTrans, this->kernel_dim_,
          this->conv_out_spatial_dim_, this->conv_out_channels_ / this->group_,
          (Dtype) 1., weights + this->weight_offset_ * g,
          output + this->output_offset_ * g, (Dtype) 0.,
          col_buff + this->col_offset_ * g);
  }
  if (!this->is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::weight_cpu_gemm(
                                                  const Dtype* input,
                                                  const Dtype* output,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    col_buff = this->col_buffer_.cpu_data();
  }
  for (int_tp g = 0; g < this->group_; ++g) {
    caffe_gemm<Dtype>(CblasNoTrans, CblasTrans,
          this->conv_out_channels_ / this->group_,
          this->kernel_dim_, this->conv_out_spatial_dim_, (Dtype) 1.,
          output + this->output_offset_ * g,
          col_buff + this->col_offset_ * g, (Dtype) 1.,
          weights + this->weight_offset_ * g);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::backward_cpu_bias(
                                                    Dtype* bias,
                                                    const Dtype* input) {
  caffe_gemv<Dtype>(CblasNoTrans, this->num_output_,
                        this->out_spatial_dim_, 1., input,
                        this->bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::forward_gpu_gemm(
                                                   vptr<const Dtype> input,
                                                   const uint_tp input_off,
                                                   vptr<const Dtype> weights,
                                                   vptr<Dtype> output,
                                                   const uint_tp output_off,
                                                   bool skip_im2col) {
  vptr<const Dtype> col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input + input_off,
                      this->col_buffer()->mutable_gpu_data());
    }
    col_buff = this->col_buffer()->gpu_data();
  }
  for (int_tp g = 0; g < this->group_; ++g) {
    this->device_->template gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, this->conv_out_channels_ / this->group_,
        this->conv_out_spatial_dim_, this->kernel_dim_, (Dtype) 1.,
        weights + this->weight_offset_ * g,
        col_buff + (this->is_1x1_ ? input_off : 0)
        + this->col_offset_ * g, (Dtype) 0.,
        output + output_off + this->output_offset_ * g);
  }
  this->unlock_col_buffer();
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::forward_gpu_bias(
                                                   vptr<Dtype> output,
                                                   const uint_tp output_off,
                                                   vptr<const Dtype> bias) {
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            this->num_output_,
                            this->out_spatial_dim_, 1, (Dtype) 1., bias,
                            this->bias_multiplier_.gpu_data(), (Dtype) 1.,
                            output + output_off);
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::backward_gpu_gemm(
                                                    vptr<const Dtype> output,
                                                    const uint_tp output_off,
                                                    vptr<const Dtype> weights,
                                                    vptr<Dtype> input,
                                                    const uint_tp input_off) {
  vptr<Dtype> col_buff = this->col_buffer()->mutable_gpu_data();
  if (this->is_1x1_) {
    col_buff = input;
  }
  for (int_tp g = 0; g < this->group_; ++g) {
    this->device_->template gemm<Dtype>(
        CblasTrans, CblasNoTrans, this->kernel_dim_,
        this->conv_out_spatial_dim_,
        this->conv_out_channels_ / this->group_, (Dtype) 1.,
        weights + this->weight_offset_ * g,
        output + output_off + this->output_offset_ * g, (Dtype) 0.,
        col_buff + (this->is_1x1_ ? input_off : 0) + this->col_offset_ * g);
  }
  if (!this->is_1x1_) {
    conv_col2im_gpu(col_buff, input + input_off);
  }
  this->unlock_col_buffer();
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::weight_gpu_gemm(
                                                  vptr<const Dtype> input,
                                                  const uint_tp input_off,
                                                  vptr<const Dtype> output,
                                                  const uint_tp output_off,
                                                  vptr<Dtype> weights) {
  vptr<const Dtype> col_buff = input;
  if (!this->is_1x1_) {
    conv_im2col_gpu(input + input_off, this->col_buffer()->mutable_gpu_data());
    col_buff = this->col_buffer()->gpu_data();
  }
  for (int_tp g = 0; g < this->group_; ++g) {
    this->device_->template gemm<Dtype>(
        CblasNoTrans, CblasTrans, this->conv_out_channels_ / this->group_,
        this->kernel_dim_,
        this->conv_out_spatial_dim_, (Dtype) 1.,
        output + output_off + this->output_offset_ * g,
        col_buff + (this->is_1x1_ ? input_off : 0)
        + this->col_offset_ * g, (Dtype) 1.,
        weights + this->weight_offset_ * g);
  }
  this->unlock_col_buffer();
}

template<typename Dtype, typename MItype, typename MOtype>
void CaffeConvolutionLayer<Dtype, MItype, MOtype>::backward_gpu_bias(
                                                    vptr<Dtype> bias,
                                                    vptr<const Dtype> input,
                                                    const uint_tp input_off) {
  this->device_->template gemv<Dtype>(CblasNoTrans, this->num_output_,
                 this->out_spatial_dim_, 1., input + input_off,
                 this->bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS_3T_GUARDED(CaffeConvolutionLayer, (half_fp), (half_fp),
                             (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(CaffeConvolutionLayer, (float), (float),
                             (float));
INSTANTIATE_CLASS_3T_GUARDED(CaffeConvolutionLayer, (double), (double),
                             (double));

}  // namespace caffe
