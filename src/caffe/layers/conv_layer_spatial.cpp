#ifndef USE_OCL
#define CPU_ONLY 1
#endif
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::compute_output_shape() {

  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }

}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  tuned_ = 0;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

  // Shape the tops.
  vector<int> top_shape(bottom[0]->shape().begin(),
  bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
 
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  CHECK_EQ(2, this->num_spatial_axes_)
      << "ConvolutionSpatial input must have 2 spatial axes "
      << "(e.g., height and width). ";
 
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];

//  // Prepare the matrix multiplication computation.
//  // Each input will be convolved as a single GEMM.
  M_ = this->num_output_ / this->group_;
  K_ = this->channels_ * kernel_h * kernel_w / this->group_;
  N_ = height_out * width_out;
//  // The im2col result buffer will only hold one image at a time to avoid
//  // overly large memory usage.
  col_buffer_.Reshape(this->num_, this->channels_, height + 2 * pad_h,
      width + 2 * pad_w);
  swizzled_weights_.Reshape(this->num_output_, this->channels_, kernel_h + 2 * pad_h,
      kernel_w + 2 * pad_w);
//  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (this->bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset = M_ * K_;  // number of filter parameters in a group
    int col_offset = K_ * N_;  // number of values in an input region / column
    int top_offset = M_ * N_;  // number of values in an output region / column
    for (int n = 0; n < this->num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_cpu(bottom_data + n * this->bottom_dim_, this->channels_, height, width,
          kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, col_data);
      // Take inner products for groups.
      for (int g = 0; g < this->group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype) 1., weight + weight_offset * g, col_data + col_offset * g,
            (Dtype) 0., top_data + n * this->top_dim_ + top_offset * g);
      }
      // Add bias.
      if (this->bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_, N_, 1,
            (Dtype) 1., this->blobs_[1]->cpu_data(),
            bias_multiplier_.cpu_data(), (Dtype) 1.,
            top_data + n * this->top_dim_);
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, N_, 1.,
            top_diff + n * this->top_dim_, bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_cpu_data();
      Dtype* col_diff = col_buffer_.mutable_cpu_diff();
      const Dtype* bottom_data = (bottom)[i]->cpu_data();
      Dtype* bottom_diff = (bottom)[i]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_cpu(bottom_data + n * this->bottom_dim_, this->channels_, height,
            width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype) 1., top_diff + n * this->top_dim_ + top_offset * g,
                col_data + col_offset * g, (Dtype) 1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
          for (int g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype) 1., weight + weight_offset * g,
                top_diff + n * this->top_dim_ + top_offset * g, (Dtype) 0.,
                col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_cpu(col_diff, this->channels_, height, width, kernel_h, kernel_w,
              pad_h, pad_w, stride_h, stride_w,
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
