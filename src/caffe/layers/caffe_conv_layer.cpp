#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CaffeConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>&
    bottom, vector<Blob<Dtype>*>* top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Figure out the dimensions for individual gemms.
  M_ = this->num_output_ / this->group_;
  K_ = this->channels_ * this->kernel_h_ * this->kernel_w_ / this->group_;
  N_ = this->height_out_ * this->width_out_;
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(1, this->channels_ * this->kernel_h_ * this->kernel_w_,
      this->height_out_, this->width_out_);
  // Set up the all ones "bias multiplier" for adding bias using blas
  if (this->bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void CaffeConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    Dtype* col_data = col_buffer_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < this->num_; ++n) {
      // First, im2col
      im2col_cpu(bottom_data + bottom[i]->offset(n), this->channels_,
          this->height_, this->width_, this->kernel_h_, this->kernel_w_,
          this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_,
          col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < this->group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (this->bias_term_) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
            bias_multiplier_.cpu_data(),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
}

template <typename Dtype>
void CaffeConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
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
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_cpu_data();
      Dtype* col_diff = col_buffer_.mutable_cpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_cpu(bottom_data + (*bottom)[i]->offset(n), this->channels_,
            this->height_, this->width_, this->kernel_h_, this->kernel_w_,
            this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_,
            col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary
        if (propagate_down[i]) {
          for (int g = 0; g < this->group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight + weight_offset * g,
                top_diff + top[i]->offset(n) + top_offset * g,
                (Dtype)0., col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_cpu(col_diff, this->channels_, this->height_, this->width_,
              this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
              this->stride_h_, this->stride_w_, bottom_diff +
              (*bottom)[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CaffeConvolutionLayer);
#endif

INSTANTIATE_CLASS(CaffeConvolutionLayer);

}  // namespace caffe

