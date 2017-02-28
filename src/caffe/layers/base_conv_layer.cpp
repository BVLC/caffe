#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  use_colbuffer_ = true;

  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int_tp first_spatial_axis = channel_axis_ + 1;
  const int_tp num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int_tp> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int_tp> spatial_dim_blob_shape(
      1, std::max(num_spatial_axes_, (int_tp) 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int_tp* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
      << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
      << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int_tp num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
    << "kernel_size must be specified once, or once per spatial dimension "
    << "(kernel_size specified " << num_kernel_dims << " times; "
    << num_spatial_axes_ << " spatial dims);";
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
      conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0)<< "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int_tp* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
      << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
      << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int_tp num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
        num_stride_dims == num_spatial_axes_)
    << "stride must be specified once, or once per spatial dimension "
    << "(stride specified " << num_stride_dims << " times; "
    << num_spatial_axes_ << " spatial dims);";
    const int_tp kDefaultStride = 1;
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
      conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int_tp* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
      << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
      << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int_tp num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
        num_pad_dims == num_spatial_axes_)
    << "pad must be specified once, or once per spatial dimension "
    << "(pad specified " << num_pad_dims << " times; "
    << num_spatial_axes_ << " spatial dims);";
    const int_tp kDefaultPad = 0;
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
      conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }

  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int_tp* dilation_data = dilation_.mutable_cpu_data();
  const int_tp num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }

  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1
        && pad_data[i] == 0;
    if (!is_1x1_) {
      break;
    }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
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
  vector<int_tp> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int_tp> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape, this->device_);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
      << weight_shaped_blob.shape_string() << "; instead, shape was "
      << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape, this->device_);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
      << bias_shaped_blob.shape_string() << "; instead, shape was "
      << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
              this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const int_tp first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
    << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
    << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int_tp bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int_tp> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int_tp> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int_tp* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int_tp i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.

  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }

  col_buffer_.Reshape(col_buffer_shape_);
  if (Caffe::mode() == Caffe::Brew::GPU && use_colbuffer_) {
    // Shared column buffer per device-queue across all layers on that device
    for (int_tp i = 0; i < this->device_->num_queues(); ++i) {
      shared_ptr<Blob<Dtype> > buffer = this->device_
          ->template Buffer<Dtype>(i);
      buffer->Reshape(col_buffer_shape_);
    }
  }

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int_tp> bias_multiplier_shape(1, out_spatial_dim_);
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
  for (int_tp g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          conv_out_channels_ / group_, conv_out_spatial_dim_,
                          kernel_dim_, (Dtype) 1., weights + weight_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype) 0.,
                          output + output_offset_ * g);
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
                                                   const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype) 1., bias,
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

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
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

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
                                                    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input,
                        bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
                                                   const int_tp input_off,
                                                   const Dtype* weights,
                                                   Dtype* output,
                                                   const int_tp output_off,
                                                   bool skip_im2col) {
  const Dtype* col_buff = input;
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (!is_1x1_) {
      if (!skip_im2col) {
        conv_im2col_gpu(input + input_off, col_buffer()->mutable_gpu_data());
      }
      col_buff = col_buffer()->gpu_data();
    }
    for (int_tp g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(
          CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_,
          conv_out_spatial_dim_, kernel_dim_, (Dtype) 1.,
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
    for (int_tp g = 0; g < group_; ++g) {
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasNoTrans, conv_out_channels_ / group_,
                               conv_out_spatial_dim_, kernel_dim_,
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
                                                   const int_tp output_off,
                                                   const Dtype* bias) {
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                          out_spatial_dim_, 1, (Dtype) 1., bias,
                          bias_multiplier_.gpu_data(), (Dtype) 1.,
                          output + output_off);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                             CblasNoTrans, num_output_, out_spatial_dim_, 1,
                             (Dtype) 1., (cl_mem) bias, 0,
                             (cl_mem) (bias_multiplier_.gpu_data()), 0,
                             (Dtype) 1., (cl_mem) output, output_off);
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
                                                    const int_tp output_off,
                                                    const Dtype* weights,
                                                    Dtype* input,
                                                    const int_tp input_off) {
  Dtype* col_buff = col_buffer()->mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int_tp g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(
          CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_,
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
    for (int_tp g = 0; g < group_; ++g) {
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasTrans,
                               CblasNoTrans, kernel_dim_, conv_out_spatial_dim_,
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
                                                  const int_tp input_off,
                                                  const Dtype* output,
                                                  const int_tp output_off,
                                                  Dtype* weights) {
  const Dtype* col_buff = input;
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (!is_1x1_) {
      conv_im2col_gpu(input + input_off, col_buffer()->mutable_gpu_data());
      col_buff = col_buffer()->gpu_data();
    }
    for (int_tp g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(
          CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_,
          conv_out_spatial_dim_, (Dtype) 1.,
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
    for (int_tp g = 0; g < group_; ++g) {
      greentea_gpu_gemm<Dtype>(this->device_->id(), CblasNoTrans,
                               CblasTrans, conv_out_channels_ / group_,
                               kernel_dim_, conv_out_spatial_dim_, (Dtype) 1.,
                               (cl_mem) output, output_off + output_offset_ * g,
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
                                                    const int_tp input_off) {
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                          input + input_off, bias_multiplier_.gpu_data(), 1.,
                          bias);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    greentea_gpu_gemv<Dtype>(this->device_->id(), CblasNoTrans,
                             num_output_, out_spatial_dim_, 1., (cl_mem) input,
                             input_off, (cl_mem) (bias_multiplier_.gpu_data()),
                             0, 1., (cl_mem) bias, 0);
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
shared_ptr<Blob<Dtype> > BaseConvolutionLayer<Dtype>::col_buffer() {
  return this->device_->template Buffer<Dtype>(
      this->device_->current_queue_id());
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
