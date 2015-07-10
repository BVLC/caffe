#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/vision_layers.hpp"
#include "assert.h"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
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
    DLOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}


template <typename Dtype>
size_t BaseConvolutionLayer<Dtype>::getChannelNumPixels() {
  return height_*width_;
}

template <typename Dtype>
size_t BaseConvolutionLayer<Dtype>::getImageNumPixels() {
  return channels_*getChannelNumPixels();
}

template <typename Dtype>
size_t BaseConvolutionLayer<Dtype>::getChannelColLength() {
  return height_out_*width_out_;
}

template <typename Dtype>
size_t BaseConvolutionLayer<Dtype>::getImageColLength() {
  return kernel_dim_*getChannelColLength();
}


template <typename Dtype>
bool BaseConvolutionLayer<Dtype>::setupMaskIM2COL() {

  if ( height_*width_*channels_*kernel_h_*kernel_w_*height_out_*width_out_ <= 0 ) {
    LOG(WARNING)<<"skipping because at least one value is zero";
    return false;
  }

  DLOG(INFO)<<"num_        = "<<num_;
  DLOG(INFO)<<"height_     = "<<height_;
  DLOG(INFO)<<"width_      = "<<width_;
  DLOG(INFO)<<"channels_   = "<<channels_;
  DLOG(INFO)<<"kernel_h_   = "<<kernel_h_;
  DLOG(INFO)<<"kernel_w_   = "<<kernel_w_;
  DLOG(INFO)<<"stride_h_   = "<<kernel_h_;
  DLOG(INFO)<<"stride_w_   = "<<kernel_w_;
  DLOG(INFO)<<"height_out_ = "<<height_out_;
  DLOG(INFO)<<"width_out_  = "<<width_out_;

  index_mask_.Reshape(1, 1, height_, width_);
  im2col_mask_.Reshape(1, channels_*kernel_h_*kernel_w_, height_out_, width_out_);
  col2im_mask_.Reshape(1, 1, height_, width_);

  for( int pixel = 0; pixel < height_*width_; pixel++ ) {
    index_mask_.mutable_cpu_data()[pixel] = pixel;
  }

  //iSNAPSHOT("index mask", index_mask_.cpu_data(), height_*width_);
  DLOG(INFO)<<"call im2col_cpu()";
  im2col_cpu(index_mask_.cpu_data(), channels_, height_,
            width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
            stride_h_, stride_w_, im2col_mask_.mutable_cpu_data());

  return true;
}

template <typename Dtype>
bool BaseConvolutionLayer<Dtype>::setupMaskCOL2IM() {

  DLOG(ERROR)<<"Waiting for implementation";
  return false;
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
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
  if (reverse_dimensions()) {
    col_buffer_.Reshape(num_, kernel_dim_, height_, width_);
  } else {
    col_buffer_.Reshape(num_, kernel_dim_, height_out_, width_out_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, num_* height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

  this->setupMaskIM2COL();
  //this->setupMaskCOL2IM();
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#if defined(USE_CUDA)

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {

  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      TIME("forward_gpu_gemm()->conv_im2col_gpu()",
          {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
          });
    }
    TIME("forward_gpu_gemm()->col_buffer_.gpu_data()",
        {
    col_buff = col_buffer_.gpu_data();
        });
  }

  TIME("2nd step()",
      {
  for (int g = 0; g < group_; ++g) {
    TIME("forward_gpu_gemm()->caffe_gpu_gemm()",
        {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
        });
  }
    });
}
#endif

#if defined(USE_OPENCL)

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(
    const Dtype* input, const size_t input_offset,
    const Dtype* weights, const size_t weights_offset,
    Dtype* output, const size_t output_offset,
    bool skip_im2col) {

  const Dtype* col_buff     = input;
  size_t col_buffer_offset  = input_offset;

  if (!is_1x1_) {
    if (!skip_im2col) {
      //conv_im2col_gpu((const Dtype*) input + input_offset, col_buffer_.mutable_gpu_data());
      conv_im2col_gpu(input, input_offset, col_buffer_.mutable_gpu_data(), 0);
      col_buffer_offset = 0;

    }
    TIME("forward_gpu_gemm()->col_buffer_.gpu_data()",
        {
    col_buff = col_buffer_.gpu_data();
        });
  }

  TIME("2nd step()",
      {
  for (int g = 0; g < group_; ++g) {
    TIME("forward_gpu_gemm()->caffe_gpu_gemm()", {

        /*
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_ / group_,
            (Dtype)1., (Dtype*) weights + weights_offset + weight_offset_ * g,
            (Dtype*) col_buff + col_offset_ * g,
            (Dtype)0., (Dtype*) output + output_offset + output_offset_ * g);
        */

        size_t M = conv_out_channels_ /group_;
        size_t N = conv_out_spatial_dim_;
        size_t K = kernel_dim_ / group_;

        DLOG(INFO)<<"MNK = "<<M<<" x "<<N<<" x "<<K;
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M, N, K,
            (Dtype)1., (Dtype*) weights, weights_offset + weight_offset_ * g,
            col_buff, col_buffer_offset + col_offset_ * g,
            (Dtype)0., (Dtype*) output, output_offset + output_offset_ * g);

    });
  }
    });
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {

  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      TIME("forward_gpu_gemm()->conv_im2col_gpu()",
          {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
          });
    }
    TIME("forward_gpu_gemm()->col_buffer_.gpu_data()",
        {
    col_buff = col_buffer_.gpu_data();
        });
  }

  TIME("2nd step()",
      {
  for (int g = 0; g < group_; ++g) {
    TIME("forward_gpu_gemm()->caffe_gpu_gemm()", {

        /*
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_ / group_,
            (Dtype) 1.,
            (Dtype*) weights + weight_offset_ * g,
            (Dtype*) col_buff + col_offset_ * g,
            (Dtype) 0.,
            (Dtype*) output +  output_offset_ * g);
        */
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
            group_, conv_out_spatial_dim_, kernel_dim_ / group_,
            (Dtype)1., (Dtype*) weights, weight_offset_ * g,
            col_buff, col_offset_ * g,
            (Dtype)0., (Dtype*) output, output_offset_ * g);
    });
  }
    });
}

#endif

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  TIME("BIAS1 caffe_gpu_gemm()",
      {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
      });
}

#if defined(USE_OPENCL)
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output, const size_t output_offset,
    const Dtype* bias) {
  TIME("BIAS2 caffe_gpu_gemm()",
      {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
              height_out_ * width_out_, 1, (Dtype)1., bias, 0.0, bias_multiplier_.gpu_data(), 0.0,
              (Dtype)1., output, output_offset);

      });
}
#endif

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
#if defined(USE_CUDA)
  	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
#endif
#if defined(USE_OPENCL)
    TIME("backward_gpu_gemm()->caffe_gpu_gemm()", {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
            conv_out_spatial_dim_, conv_out_channels_ / group_,
            (Dtype)1.,
            (Dtype*) weights + weight_offset_ * g,
            (Dtype*) output + output_offset_ * g,
            (Dtype)0.,
            (Dtype*) col_buff + col_offset_ * g);

        /*
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
            conv_out_spatial_dim_, conv_out_channels_ / group_,
            (Dtype)1.,
            weights, weight_offset_ * g,
            output, output_offset_ * g,
            (Dtype)0.,
            col_buff, col_offset_ * g);
            */
    });
#endif
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

#if defined(USE_OPENCL)

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(
    const Dtype* output, const size_t output_offset,
    const Dtype* weights, const size_t weights_offset,
    Dtype* input, const size_t input_offset) {

  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  size_t col_buffer_offset = 0;
  if (is_1x1_) {
    col_buff = input;
    col_buffer_offset = input_offset;
  }
  for (int g = 0; g < group_; ++g) {
    TIME("backward_gpu_gemm()->caffe_gpu_gemm()", {
        if ( is_1x1_ ) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
              conv_out_spatial_dim_, conv_out_channels_ / group_,
              (Dtype)1.,
              (Dtype*) weights, weights_offset + weight_offset_ * g,
              (Dtype*) output, output_offset + output_offset_ * g,
              (Dtype)0.,
              (Dtype*) col_buff, col_buffer_offset + col_offset_ * g);
        } else {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
              conv_out_spatial_dim_, conv_out_channels_ / group_,
              (Dtype)1.,
              (Dtype*) weights, weights_offset + weight_offset_ * g,
              (Dtype*) output, output_offset + output_offset_ * g,
              (Dtype)0.,
              (Dtype*) col_buff, col_buffer_offset + col_offset_ * g);
        }
    });
  }
  if (!is_1x1_) {
    //conv_col2im_gpu(col_buff, (Dtype*) input + input_offset);
    conv_col2im_gpu(col_buff, col_buffer_offset, input, input_offset);
  }
}
#endif

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
#if defined(USE_CUDA)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
#endif
#if defined(USE_OPENCL)
    TIME("weight_gpu_gemm()->caffe_gpu_gemm()", {

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
            kernel_dim_ / group_, conv_out_spatial_dim_,
            (Dtype)1.,
            (Dtype*) output + output_offset_ * g,
            (Dtype*) col_buff +  col_offset_ * g,
            (Dtype)1.,
            (Dtype*) weights + weight_offset_ * g);

        /*
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
            kernel_dim_ / group_, conv_out_spatial_dim_,
            (Dtype)1.,
            output, output_offset_ * g,
            col_buff, col_offset_ * g,
            (Dtype)1.,
            weights, weight_offset_ * g);
            */
    });
#endif
  }
}

#if defined(USE_OPENCL)

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(
    const Dtype* input, const size_t input_offset,
    const Dtype* output, const size_t output_offset,
    Dtype* weights, const size_t weights_offset) {

  const Dtype* col_buff   = input;
  size_t col_buff_offset  = input_offset;

  if (!is_1x1_) {
    //conv_im2col_gpu(input + input_offset, col_buffer_.mutable_gpu_data());
    conv_im2col_gpu(input, input_offset, col_buffer_.mutable_gpu_data(), 0);
    col_buff = col_buffer_.gpu_data();
    col_buff_offset = 0;
  }
  for (int g = 0; g < group_; ++g) {
    TIME("weight_gpu_gemm()->caffe_gpu_gemm()", {
        /*
        if ( !is_1x1_ ) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
              kernel_dim_ / group_, conv_out_spatial_dim_,
              (Dtype)1.,
              output, output_offset + output_offset_ * g ,
              (Dtype*) col_buff, col_buff_offset + col_offset_ * g,
              (Dtype)1.,
              (Dtype*) weights, weights_offset + weight_offset_ * g);
        } else {
        */
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
              kernel_dim_ / group_, conv_out_spatial_dim_,
              (Dtype)1.,
              output, output_offset + output_offset_ * g,
              (Dtype*) col_buff, col_buff_offset + col_offset_ * g,
              (Dtype)1.,
              (Dtype*) weights, weights_offset + weight_offset_ * g);
        //}
    });
  }
}

#endif

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  TIME("backward_gpu_bias()->caffe_gpu_gemv()", {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
  });
}

#if defined(USE_OPENCL)

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(
    Dtype* bias, const size_t bias_offset,
    const Dtype* input, const size_t input_offset) {
  TIME("backward_gpu_bias()->caffe_gpu_gemv()", {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, input_offset, bias_multiplier_.gpu_data(), 0, 1., bias, bias_offset);
  });
}

#endif

//#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
