#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
  << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
  << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
          && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
  << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
          && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
  << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
  && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
  << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
            conv_out_channels_, conv_in_channels_ / group_,
            kernel_h_, kernel_w_, this->device_context_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_context_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
              this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
  " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
    << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
    << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
    << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (Caffe::mode() == Caffe::Brew::CPU) {
    if (reverse_dimensions()) {
      col_buffer_.Reshape(1, kernel_dim_, height_, width_);
    } else {
      col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
    }
  } else {
    // Shared column buffer per device-queue across all layers on that device
    for (int i = 0; i < this->device_context_->num_queues(); ++i) {
      if (reverse_dimensions()) {
        shared_ptr< Blob<Dtype> > buffer =
            this->device_context_->template Buffer<Dtype>(i);
        buffer->Reshape(1, kernel_dim_, height_, width_);
      } else {
        shared_ptr< Blob<Dtype> > buffer =
            this->device_context_->template Buffer<Dtype>(i);
        buffer->Reshape(1, kernel_dim_, height_out_, width_out_);
      }
    }
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bool reshaped = bias_multiplier_.Reshape(bias_multiplier_shape);
    // This will trigger a memory copy if in GPU mode,
    // which may not be necessary.
    // Thus omit to set the values if not necessary.
    if (reshaped) {
      caffe_set(bias_multiplier_.count(), Dtype(1),
                bias_multiplier_.mutable_cpu_data());
    }
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
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
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          conv_out_channels_ / group_, conv_out_spatial_dim_,
                          kernel_dim_ / group_, (Dtype) 1.,
                          weights + weight_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype) 0.,
                          output + output_offset_ * g);
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
                                                   const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        height_out_ * width_out_, 1, (Dtype) 1., bias,
                        bias_multiplier_.cpu_data(), (Dtype) 1., output);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
                                                    const Dtype* weights,
                                                    Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
                          conv_out_spatial_dim_, conv_out_channels_ / group_,
                          (Dtype) 1., weights + weight_offset_ * g,
                          output + output_offset_ * g, (Dtype) 0.,
                          col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
                                                  const Dtype* output,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                          kernel_dim_ / group_, conv_out_spatial_dim_,
                          (Dtype) 1., output + output_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype) 1.,
                          weights + weight_offset_ * g);
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
                                                    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
                        input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
                                                   const int input_off,
                                                   const Dtype* weights,
                                                   Dtype* output,
                                                   const int output_off,
                                                   bool skip_im2col) {
  const Dtype* col_buff = input;
  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (!is_1x1_) {
      if (!skip_im2col) {
        conv_im2col_gpu(input + input_off, col_buffer()->mutable_gpu_data());
      }
      col_buff = col_buffer()->gpu_data();
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_,
          conv_out_spatial_dim_, kernel_dim_ / group_, (Dtype) 1.,
          weights + weight_offset_ * g,
          col_buff + (is_1x1_ ? input_off : 0) + col_offset_ * g, (Dtype) 0.,
          output + output_off + output_offset_ * g);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (!is_1x1_) {
      if (!skip_im2col) {
        greentea_conv_im2col_gpu(input, input_off,
                                 col_buffer()->mutable_gpu_data(), 0);
      }
      col_buff = col_buffer()->gpu_data();
    }
    for (int g = 0; g < group_; ++g) {
      greentea_gpu_gemm<Dtype>(this->device_context_->id(), CblasNoTrans,
                               CblasNoTrans, conv_out_channels_ / group_,
                               conv_out_spatial_dim_, kernel_dim_ / group_,
                               (Dtype) 1., (cl_mem) weights, weight_offset_ * g,
                               (cl_mem) col_buff,
                               (is_1x1_ ? input_off : 0) + col_offset_ * g,
                               (Dtype) 0., (cl_mem) output,
                               output_off + output_offset_ * g);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
                                                   const int output_off,
                                                   const Dtype* bias) {
  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          height_out_ * width_out_, 1, (Dtype) 1., bias,
                          bias_multiplier_.gpu_data(), (Dtype) 1.,
                          output + output_off);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_gemm<Dtype>(this->device_context_->id(), CblasNoTrans,
                             CblasNoTrans, num_output_,
                             height_out_ * width_out_, 1, (Dtype) 1.,
                             (cl_mem) bias, 0,
                             (cl_mem) (bias_multiplier_.gpu_data()), 0,
                             (Dtype) 1., (cl_mem) output, output_off);
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
                                                    const int output_off,
                                                    const Dtype* weights,
                                                    Dtype* input,
                                                    const int input_off) {
  Dtype* col_buff = col_buffer()->mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(
          CblasTrans, CblasNoTrans, kernel_dim_ / group_, conv_out_spatial_dim_,
          conv_out_channels_ / group_, (Dtype) 1., weights + weight_offset_ * g,
          output + output_off + output_offset_ * g, (Dtype) 0.,
          col_buff + (is_1x1_ ? input_off : 0) + col_offset_ * g);
    }
    if (!is_1x1_) {
      conv_col2im_gpu(col_buff, input + input_off);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    for (int g = 0; g < group_; ++g) {
      greentea_gpu_gemm<Dtype>(this->device_context_->id(), CblasTrans,
                               CblasNoTrans, kernel_dim_ / group_,
                               conv_out_spatial_dim_,
                               conv_out_channels_ / group_, (Dtype) 1.,
                               (cl_mem) weights, weight_offset_ * g,
                               (cl_mem) output, output_off + output_offset_ * g,
                               (Dtype) 0., (cl_mem) col_buff,
                               (is_1x1_ ? input_off : 0) + col_offset_ * g);
    }
    if (!is_1x1_) {
      greentea_conv_col2im_gpu(col_buff, 0, input, input_off);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
                                                  const int input_off,
                                                  const Dtype* output,
                                                  const int output_off,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (!is_1x1_) {
      conv_im2col_gpu(input + input_off, col_buffer()->mutable_gpu_data());
      col_buff = col_buffer()->gpu_data();
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(
          CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
          kernel_dim_ / group_, conv_out_spatial_dim_, (Dtype) 1.,
          output + output_off + output_offset_ * g,
          col_buff + (is_1x1_ ? input_off : 0) + col_offset_ * g, (Dtype) 1.,
          weights + weight_offset_ * g);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (!is_1x1_) {
      greentea_conv_im2col_gpu(input, input_off,
                               col_buffer()->mutable_gpu_data(), 0);
      col_buff = col_buffer()->gpu_data();
    }
    for (int g = 0; g < group_; ++g) {
      greentea_gpu_gemm<Dtype>(this->device_context_->id(), CblasNoTrans,
                               CblasTrans, conv_out_channels_ / group_,
                               kernel_dim_ / group_, conv_out_spatial_dim_,
                               (Dtype) 1., (cl_mem) output,
                               output_off + output_offset_ * g,
                               (cl_mem) col_buff,
                               (is_1x1_ ? input_off : 0) + col_offset_ * g,
                               (Dtype) 1., (cl_mem) weights,
                               weight_offset_ * g);
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
                                                    const Dtype* input,
                                                    const int input_off) {
  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_,
                          1., input + input_off, bias_multiplier_.gpu_data(),
                          1., bias);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_gemv<Dtype>(this->device_context_->id(), CblasNoTrans,
                             num_output_, height_out_ * width_out_, 1.,
                             (cl_mem) input, input_off,
                             (cl_mem) (bias_multiplier_.gpu_data()), 0, 1.,
                             (cl_mem) bias, 0);
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
shared_ptr< Blob<Dtype> > BaseConvolutionLayer<Dtype>::col_buffer() {
    return this->device_context_->
        template Buffer<Dtype>(this->device_context_->current_queue_id());
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
