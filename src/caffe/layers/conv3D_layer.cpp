#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void Convolution3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 5 axes, "
            << "corresponding to (num, channels, height, width, depth)";
    // Configure the kernel size, padding, stride, and inputs.
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();

    CHECK(conv_param.has_kernel_size()) 
            << "Pad dimension is required. Set to be equal in all dimensions.";
    CHECK_GT(conv_param.kernel_size(), 0) << "Filter dimensions cannot be zero.";
    kernel_h_ = kernel_w_ = kernel_d_ = conv_param.kernel_size();

    CHECK(conv_param.has_pad())
            << "Pad dimension is required. Set to be equal in all dimensions.";
    pad_h_ = pad_w_ = pad_d_ = conv_param.pad();    

    CHECK(conv_param.has_stride())
            << "Stride dimension is required. Set to be equal in all dimensions.";
    CHECK_GT(conv_param.kernel_size(), 0) << "Stride dimensions cannot be zero.";
    stride_h_ = stride_w_ = stride_d_ = conv_param.stride();

    // Special case: im2col is the identity for 1x1 convolution with stride 1
    // and no padding, so flag for skipping the buffer and transformation.
    // is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1 && kernel_h_ == 1
    //     && stride_h_ == 1 && stride_w_ == 1 && stride_d_ &&
    //     pad_h_ == 0 && pad_w_ == 0 && pad_d_ ==;
    is_1x1_ = false;

    // Configure output channels and groups.
    channels_ = bottom[0]->shape(1);
    // channels_ = bottom[0]->channels();
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
        // output channels x input channels per-group x 
        // kernel depth x kernel height x kernel width
        vector<int> weights_shape(5);
        weights_shape[0] = conv_out_channels_;
        weights_shape[1] = conv_in_channels_ / group_;
        weights_shape[2] = kernel_h_;
        weights_shape[3] = kernel_w_;
        weights_shape[4] = kernel_d_;
        this->blobs_[0].reset(new Blob<Dtype>(weights_shape));
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
void Convolution3DLayer<Dtype>::compute_output_shape() {
    this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
            / this->stride_h_ + 1;
    this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
            / this->stride_w_ + 1;
    this->depth_out_ = (this->depth_ + 2 * this->pad_d_ - this->kernel_d_)
            / this->stride_d_ + 1;
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 5 axes, "
            << "corresponding to (num, channels, height, width, depth)";
    
    vector<int> shape = bottom[0]->shape();
    num_ = shape[0];
    CHECK_EQ(shape[1], channels_) << "Input size incompatible with"
        " convolution kernel.";
    height_ = shape[2];
    width_ = shape[3];
    depth_ = shape[4];

    // TODO: generalize to handle inputs of different shapes.
    for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
        vector<int> shape = bottom[bottom_id]->shape();
        CHECK_EQ(num_, shape[0]) << "Inputs must have same num.";
        CHECK_EQ(channels_ , shape[1]) << "Inputs must have same channels";
        CHECK_EQ(height_, shape[2]) << "Inputs must have same height";
        CHECK_EQ(width_,  shape[3]) << "Inputs must have same width";
        CHECK_EQ(depth_, shape[4]) << "Inputs must have same depth";
    }
    // Shape the tops.
    compute_output_shape();
    for (int top_id = 0; top_id < top.size(); ++top_id) {
        vector<int> out_shape(5);
        out_shape[0] = num_;
        out_shape[1] = num_output_;
        out_shape[2] = height_out_; 
        out_shape[3] = width_out_;
        out_shape[4] = depth_out_;
        top[top_id]->Reshape(out_shape);
    }
    if (reverse_dimensions()) {
        conv_in_height_ = height_out_;
        conv_in_width_ = width_out_;
        conv_in_depth_ = depth_out_;
        conv_out_spatial_dim_ = height_ * width_ * depth_;
    } else {
        conv_in_height_ = height_;
        conv_in_width_ = width_;
        conv_in_depth_ = depth_;
        conv_out_spatial_dim_ = height_out_ * width_out_ * depth_out_;
    }
    kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_ * kernel_d_;
    weight_offset_ = conv_out_channels_ * 
            kernel_dim_ / group_ / group_ / group_;
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // The im2col result buffer will only hold one image at a time to avoid
    // overly large memory usage. In the special case of 1x1 convolution
    // it goes lazily unused to save memory.
    if (reverse_dimensions()) {
        vector<int> back_shape(5);
        back_shape[0] = 1;
        back_shape[1] = kernel_dim_;
        back_shape[2] = height_; 
        back_shape[3] = width_;
        back_shape[4] = depth_;
        col_buffer_.Reshape(back_shape);
    } else {
        vector<int> forw_shape(5);
        forw_shape[0] = 1;
        forw_shape[1] = kernel_dim_;
        forw_shape[2] = height_out_; 
        forw_shape[3] = width_out_;
        forw_shape[4] = depth_out_;
        col_buffer_.Reshape(forw_shape);
    }
    // Set up the all ones "bias multiplier" for adding biases by BLAS
    if (bias_term_) {
        vector<int> bias_multiplier_shape(1, height_out_ * 
                width_out_ * depth_out_);
        bias_multiplier_.Reshape(bias_multiplier_shape);
        caffe_set(bias_multiplier_.count(), Dtype(1),
                bias_multiplier_.mutable_cpu_data());
    }
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(vector<int>(1,n)), weight,
          top_data + top[i]->offset(vector<int>(1,n)));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(vector<int>(1,n)), bias);
      }
    }
  }
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + 
            top[i]->offset(vector<int>(1,n)));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(vector<int>(1,n)),
              top_diff + top[i]->offset(vector<int>(1,n)), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(vector<int>(1,n)), 
              weight, bottom_diff + bottom[i]->offset(vector<int>(1,n)));
        }
      }
    }
  }
}

// MODIFIED BASE_CONV_LAYER_FUNCTIONS

template <typename Dtype>
void Convolution3DLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
        const Dtype* weights, Dtype* output, bool skip_im2col) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
        if (!skip_im2col) {
            conv_vol2col_cpu(input, col_buffer_.mutable_cpu_data());
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
void Convolution3DLayer<Dtype>::forward_cpu_bias(Dtype* output,
        const Dtype* bias) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            height_out_ * width_out_ * depth_out_, 1, (Dtype)1., bias, 
            bias_multiplier_.cpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
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
        conv_col2vol_cpu(col_buff, input);
    }
}

template <typename Dtype>
void Convolution3DLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
        const Dtype* output, Dtype* weights) {
    const Dtype* col_buff = input;
    if (!is_1x1_) {
        conv_vol2col_cpu(input, col_buffer_.mutable_cpu_data());
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
void Convolution3DLayer<Dtype>::backward_cpu_bias(Dtype* bias,
        const Dtype* input) {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_ * 
        depth_out_, 1., input, bias_multiplier_.cpu_data(), 1., bias);
}

// #ifndef CPU_ONLY

// template <typename Dtype>
// void Convolution3DLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
//     const Dtype* weights, Dtype* output, bool skip_im2col) {
//   const Dtype* col_buff = input;
//   if (!is_1x1_) {
//     if (!skip_im2col) {
//       conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
//     }
//     col_buff = col_buffer_.gpu_data();
//   }
//   for (int g = 0; g < group_; ++g) {
//     caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
//         group_, conv_out_spatial_dim_, kernel_dim_ / group_,
//         (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
//         (Dtype)0., output + output_offset_ * g);
//   }
// }

// template <typename Dtype>
// void Convolution3DLayer<Dtype>::forward_gpu_bias(Dtype* output,
//     const Dtype* bias) {
//   caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
//       height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
//       (Dtype)1., output);
// }

// template <typename Dtype>
// void Convolution3DLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
//     const Dtype* weights, Dtype* input) {
//   Dtype* col_buff = col_buffer_.mutable_gpu_data();
//   if (is_1x1_) {
//     col_buff = input;
//   }
//   for (int g = 0; g < group_; ++g) {
//     caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
//         conv_out_spatial_dim_, conv_out_channels_ / group_,
//         (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
//         (Dtype)0., col_buff + col_offset_ * g);
//   }
//   if (!is_1x1_) {
//     conv_col2im_gpu(col_buff, input);
//   }
// }

// template <typename Dtype>
// void Convolution3DLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
//     const Dtype* output, Dtype* weights) {
//   const Dtype* col_buff = input;
//   if (!is_1x1_) {
//     conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
//     col_buff = col_buffer_.gpu_data();
//   }
//   for (int g = 0; g < group_; ++g) {
//     caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
//         kernel_dim_ / group_, conv_out_spatial_dim_,
//         (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
//         (Dtype)1., weights + weight_offset_ * g);
//   }
// }

// template <typename Dtype>
// void Convolution3DLayer<Dtype>::backward_gpu_bias(Dtype* bias,
//     const Dtype* input) {
//   caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
//       input, bias_multiplier_.gpu_data(), 1., bias);
// }

// #endif  // !CPU_ONLY

#ifdef CPU_ONLY
STUB_GPU(Convolution3DLayer);
#endif

INSTANTIATE_CLASS(Convolution3DLayer);
REGISTER_LAYER_CLASS(Convolution3D);

}  // namespace caffe
