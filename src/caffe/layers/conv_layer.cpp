#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#if defined(USE_OPENCL)

/// @brief refer to CPU forward -- the BLAS implementation is the same.
template<typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  DLOG(INFO)<<"in call ConvolutionLayer<Dtype>::Forward_gpu()";
  //TIME("ConvolutionLayer->Forward_gpu()", {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* bottom_data;
    Dtype* top_data;

    bottom_data = bottom[i]->gpu_data();
    top_data = top[i]->mutable_gpu_data();

    if ( this->group_ == 1 ) {

      const Dtype* cb_ptr = bottom_data;


      if (! this->is_1x1_) {
          // all images at once using im2col_perf kernel
          im2col_group_gpu(bottom_data, this->getImageNumPixels(), this->num_, this->channels_, this->height_, this->width_, this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_, this->col_buffer_.mutable_gpu_data(), this->getImageColLength());

          // all images at once using mask
          //im2col_group_gpu(bottom_data, this->im2col_mask_.gpu_data(), this->num_, this->channels_, this->height_, this->width_, this->kernel_h_, this->kernel_w_, this->height_out_, this->width_out_, this->col_buffer_.mutable_gpu_data());
          cb_ptr = this->col_buffer_.gpu_data();
      } else {
        cb_ptr = bottom_data;
      }

      size_t M = this->conv_out_channels_ /this->group_;
      size_t N = this->num_*this->conv_out_spatial_dim_;
      size_t K = this->kernel_dim_ / this->group_;

      caffe_gpu_group_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M, N, K,
          1, this->num_, 1,
          (Dtype)1.,
          (Dtype*) weight,
          cb_ptr,
          (Dtype)0.,
          (Dtype*) top_data);

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();

        M = this->num_output_;
        N = this->height_out_ * this->width_out_ * this->num_;
        K = 1;
        caffe_gpu_group_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M, N, K,
            1,this->num_,1,
            (Dtype)1., bias,
            this->bias_multiplier_.gpu_data(),
          (Dtype)1., top_data);
      }
    } else {
      for (int n = 0; n < this->num_; ++n) {

            //this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight, top_data + top[i]->offset(n));
            this->forward_gpu_gemm(bottom_data, bottom[i]->offset(n), weight, 0, top_data, top[i]->offset(n));

            if (this->bias_term_) {
              const Dtype* bias = this->blobs_[1]->gpu_data();
              //this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
              this->forward_gpu_bias(top_data, top[i]->offset(n), bias);
            }
            OpenCLManager::CurrentPlatform()->CurrentDevice().getNextCommandQueue();
      }

      OpenCLManager::CurrentPlatform()->CurrentDevice().setCommandQueueIDX(0);
      OpenCLManager::CurrentPlatform()->CurrentDevice().waitForCommandQueues();
    }
  }
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  TIME("ConvolutionLayer->Backward_gpu()", {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff  = this->blobs_[0]->mutable_gpu_diff();

  if (this->param_propagate_down_[0]) {
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), this->blobs_[1]->mutable_gpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

      for (int n = 0; n < this->num_; ++n) {
        //this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
        this->backward_gpu_bias(bias_diff, 0, top_diff, top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data  = bottom[i]->gpu_data();
      Dtype* bottom_diff        = bottom[i]->mutable_gpu_diff();

      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          //this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n), top_diff + top[i]->offset(n), weight_diff);
          this->weight_gpu_gemm(bottom_data, bottom[i]->offset(n), top_diff, top[i]->offset(n), weight_diff, 0.0);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          //this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight, bottom_diff + bottom[i]->offset(n));
          this->backward_gpu_gemm(top_diff, top[i]->offset(n), weight, 0, bottom_diff, bottom[i]->offset(n));
        }
      }
    }
    OpenCLManager::CurrentPlatform()->CurrentDevice().getNextCommandQueue();

  }
  OpenCLManager::CurrentPlatform()->CurrentDevice().setCommandQueueIDX(0);
  OpenCLManager::CurrentPlatform()->CurrentDevice().waitForCommandQueues();
  });
}

#endif // USE_OPENCL

#if defined(CPU_ONLY) && ! defined(USE_OPENCL)
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);
//REGISTER_LAYER_CLASS(Convolution);

}  // namespace caffe
