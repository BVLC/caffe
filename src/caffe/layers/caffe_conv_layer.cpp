
template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::forward_cpu_gemm(
                                                   const Dtype* input,
                                                   const Dtype* weights,
                                                   Dtype* output,
                                                   bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int_tp g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          conv_out_channels_ / group_, conv_out_spatial_dim_,
                          kernel_dim_, (Dtype) 1., weights + weight_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype) 0.,
                          output + output_offset_ * g);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::forward_cpu_bias(
                                                   Dtype* output,
                                                   const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype) 1., bias,
                        bias_multiplier_.cpu_data(), (Dtype) 1., output);
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::backward_cpu_gemm(
                                                    const Dtype* output,
                                                    const Dtype* weights,
                                                    Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int_tp g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                          conv_out_spatial_dim_, conv_out_channels_ / group_,
                          (Dtype) 1., weights + weight_offset_ * g,
                          output + output_offset_ * g, (Dtype) 0.,
                          col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::weight_cpu_gemm(
                                                  const Dtype* input,
                                                  const Dtype* output,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int_tp g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                          kernel_dim_, conv_out_spatial_dim_, (Dtype) 1.,
                          output + output_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype) 1.,
                          weights + weight_offset_ * g);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::backward_cpu_bias(
                                                    Dtype* bias,
                                                    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input,
                        bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::forward_gpu_gemm(
                                                   vptr<const Dtype> input,
                                                   const uint_tp input_off,
                                                   vptr<const Dtype> weights,
                                                   vptr<Dtype> output,
                                                   const uint_tp output_off,
                                                   bool skip_im2col) {
  vptr<const Dtype> col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input + input_off, col_buffer()->mutable_gpu_data());
    }
    col_buff = col_buffer()->gpu_data();
  }
  for (int_tp g = 0; g < group_; ++g) {
    this->device_->template gemm<Dtype>(
        CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_,
        conv_out_spatial_dim_, kernel_dim_, (Dtype) 1.,
        weights + weight_offset_ * g,
        col_buff + (is_1x1_ ? input_off : 0) + col_offset_ * g, (Dtype) 0.,
        output + output_off + output_offset_ * g);
  }
  unlock_col_buffer();
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::forward_gpu_bias(
                                                   vptr<Dtype> output,
                                                   const uint_tp output_off,
                                                   vptr<const Dtype> bias) {
  this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype) 1., bias,
                        bias_multiplier_.gpu_data(), (Dtype) 1.,
                        output + output_off);
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::backward_gpu_gemm(
                                                    vptr<const Dtype> output,
                                                    const uint_tp output_off,
                                                    vptr<const Dtype> weights,
                                                    vptr<Dtype> input,
                                                    const uint_tp input_off) {
  vptr<Dtype> col_buff = col_buffer()->mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int_tp g = 0; g < group_; ++g) {
    this->device_->template gemm<Dtype>(
        CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_,
        conv_out_channels_ / group_, (Dtype) 1., weights + weight_offset_ * g,
        output + output_off + output_offset_ * g, (Dtype) 0.,
        col_buff + (is_1x1_ ? input_off : 0) + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input + input_off);
  }
  unlock_col_buffer();
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::weight_gpu_gemm(
                                                  vptr<const Dtype> input,
                                                  const uint_tp input_off,
                                                  vptr<const Dtype> output,
                                                  const uint_tp output_off,
                                                  vptr<Dtype> weights) {
  vptr<const Dtype> col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input + input_off, col_buffer()->mutable_gpu_data());
    col_buff = col_buffer()->gpu_data();
  }
  for (int_tp g = 0; g < group_; ++g) {
    this->device_->template gemm<Dtype>(
        CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_,
        conv_out_spatial_dim_, (Dtype) 1.,
        output + output_off + output_offset_ * g,
        col_buff + (is_1x1_ ? input_off : 0) + col_offset_ * g, (Dtype) 1.,
        weights + weight_offset_ * g);
  }
  unlock_col_buffer();
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::backward_gpu_bias(
                                                    vptr<Dtype> bias,
                                                    vptr<const Dtype> input,
                                                    const uint_tp input_off) {
  this->device_->template gemv<Dtype>(CblasNoTrans, num_output_,
                 out_spatial_dim_, 1., input + input_off,
                 bias_multiplier_.gpu_data(), 1., bias);
}

template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Blob<Dtype> >
          BaseConvolutionLayer<Dtype, MItype, MOtype>::col_buffer() {
  if (col_buffer_lock_id_ == -1) {
    shared_col_buffer_ = this->device_->template Buffer<Dtype>(
                                       col_buffer_shape_, &col_buffer_lock_id_);
  }
  return shared_col_buffer_;
}

template<typename Dtype, typename MItype, typename MOtype>
void BaseConvolutionLayer<Dtype, MItype, MOtype>::unlock_col_buffer() {
  if (not (col_buffer_lock_id_ == -1)) {
    shared_col_buffer_ = nullptr;
    this->device_->unlock_buffer(&col_buffer_lock_id_);
  }
}


#endif  // !CPU_ONLY
