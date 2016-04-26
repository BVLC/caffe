#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/conv_spatial_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int_tp input_dim = this->input_shape(i + 1);
    const int_tp output_dim = (input_dim + 2 * pad_data[i]
        - kernel_shape_data[i]) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  tuned_ = 0;
  // Calculate variables used for kernel generation
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  kernel_h_ = kernel_shape_data[0];
  kernel_w_ = kernel_shape_data[1];
  height_ = bottom[0]->shape(this->channel_axis_ + 1);
  width_ = bottom[0]->shape(this->channel_axis_ + 2);
  const int_tp* pad_data = this->pad_.cpu_data();
  pad_h_ = pad_data[0];
  pad_w_ = pad_data[1];
  const int_tp* stride_data = this->stride_.cpu_data();
  stride_h_ = stride_data[0];
  stride_w_ = stride_data[1];

  output_h_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  output_w_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  padded_width_ = width_ + 2 * pad_w_;
  padded_height_ = height_ + 2 * pad_h_;
#ifndef CPU_ONLY
#ifdef USE_GREENTEA
  if (std::is_same<Dtype, float>::value) {
    M_ = this->num_output_ / this->group_;
    this->num_ = bottom[0]->count(0, this->channel_axis_);
    SetUp(bottom, top, Caffe::GetDefaultDevice()->backend());
  }
#endif
#endif
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

  // Shape the tops.
  vector<int_tp> top_shape(bottom[0]->shape().begin(),
                           bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }

  for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  CHECK_EQ(2, this->num_spatial_axes_)
    << "ConvolutionSpatial input must have 2 spatial axes "
    << "(e.g., height and width). ";

  const int_tp height_out = top[0]->shape(this->channel_axis_ + 1);
  const int_tp width_out = top[0]->shape(this->channel_axis_ + 2);
  M_ = this->num_output_ / this->group_;
  K_ = this->channels_ * kernel_h_ * kernel_w_ / this->group_;
  N_ = height_out * width_out;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.
  spatial_col_buffer_.Reshape(this->num_, this->channels_, height_ + 2 * pad_h_,
                      width_ + 2 * pad_w_);
  swizzled_weights_.Reshape(this->num_output_, this->channels_,
                            kernel_h_ + 2 * pad_h_, kernel_w_ + 2 * pad_w_);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (this->bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayerSpatial);
#endif

INSTANTIATE_CLASS(ConvolutionLayerSpatial);

}  // namespace caffe
