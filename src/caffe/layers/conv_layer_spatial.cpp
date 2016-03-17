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

  const int_tp height = bottom[0]->shape(this->channel_axis_ + 1);
  const int_tp width = bottom[0]->shape(this->channel_axis_ + 2);
  const int_tp height_out = top[0]->shape(this->channel_axis_ + 1);
  const int_tp width_out = top[0]->shape(this->channel_axis_ + 2);
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp pad_h = pad_data[0];
  const int_tp pad_w = pad_data[1];
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp kernel_h = kernel_shape_data[0];
  const int_tp kernel_w = kernel_shape_data[1];

//  // Prepare the matrix multiplication computation.
//  // Each input will be convolved as a single GEMM.
  M_ = this->num_output_ / this->group_;
  K_ = this->channels_ * kernel_h * kernel_w / this->group_;
  N_ = height_out * width_out;
//  // The im2col result buffer will only hold one image at a time to avoid
//  // overly large memory usage.
  col_buffer_.Reshape(this->num_, this->channels_, height + 2 * pad_h,
                      width + 2 * pad_w);
  swizzled_weights_.Reshape(this->num_output_, this->channels_,
                            kernel_h + 2 * pad_h, kernel_w + 2 * pad_w);
//  // Set up the all ones "bias multiplier" for adding biases by BLAS
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
  const int_tp height = bottom[0]->shape(this->channel_axis_ + 1);
  const int_tp width = bottom[0]->shape(this->channel_axis_ + 2);
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp pad_h = pad_data[0];
  const int_tp pad_w = pad_data[1];
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp kernel_h = kernel_shape_data[0];
  const int_tp kernel_w = kernel_shape_data[1];
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp stride_h = stride_data[0];
  const int_tp stride_w = stride_data[1];

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
  const int_tp weight_offset = M_ * K_;
  const int_tp col_offset = K_ * N_;
  const int_tp top_offset = M_ * N_;
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, N_, 1.,
                              top_diff + n * this->top_dim_,
                              bias_multiplier_.cpu_data(), 1., bias_diff);
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
      for (int_tp n = 0; n < this->num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_cpu(bottom_data + n * this->bottom_dim_, this->channels_, height,
                   width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                   1, 1, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int_tp g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(
                CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype) 1.,
                top_diff + n * this->top_dim_ + top_offset * g,
                col_data + col_offset * g, (Dtype) 1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
          for (int_tp g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(
                CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype) 1.,
                weight + weight_offset * g,
                top_diff + n * this->top_dim_ + top_offset * g, (Dtype) 0.,
                col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_cpu(col_diff, this->channels_, height, width, kernel_h,
                     kernel_w, pad_h, pad_w, stride_h, stride_w, 1, 1,
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
