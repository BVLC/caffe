// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  stride_ = this->layer_param_.convolution_param().stride();
  group_ = this->layer_param_.convolution_param().group();
  pad_ = this->layer_param_.convolution_param().pad();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(channels_ % group_, 0);
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  col_buffer_.Reshape(
      1, channels_ * kernel_size_ * kernel_size_, height_out, width_out);
  // Set the parameters
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  // Figure out the dimensions for individual gemms.
  M_ = num_output_ / group_;
  K_ = (channels_ * kernel_size_ * kernel_size_) / group_;
  N_ = height_out * width_out;
  (*top)[0]->Reshape(bottom[0]->num(), num_output_, height_out, width_out);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_size_, kernel_size_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Set up the bias filler
  if (bias_term_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
  // openmp
  num_of_threads_ = 1;
#ifdef _OPENMP
  num_of_threads_ = omp_get_max_threads();
  if (num_of_threads_ < 1) {
     LOG(WARNING) << "Conv layer: omp_get_max_threads() =" << num_of_threads_;
     num_of_threads_ = 1;
  }
#endif
  // LOG(INFO) << "Conv layer: num threads_=" << num_of_threads_;
  col_buffer_mt_.resize(num_of_threads_ *
      channels_ * kernel_size_ * kernel_size_ * height_out * width_out);
  weight_diff_mt_.resize(num_of_threads_ *
     num_output_ * (channels_ / group_)* kernel_size_ * kernel_size_);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu_task(
      const Dtype* bottom_data, Dtype* top_data, const Dtype* weight, int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out  = (width_  + 2 * pad_ - kernel_size_) / stride_ + 1;

  int tid = 0;
#ifdef _OPENMP
  int max_threads = omp_get_num_threads();
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Forward_cpu: omp_thread_num() =" << tid
               << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif
  int col_data_buffer_size = channels_ * kernel_size_ * kernel_size_ *
                             height_out * width_out;

  Dtype* col_data  = & col_buffer_mt_[ tid* col_data_buffer_size];
  int input_data_size= channels_* height_* width_;
  int bottom_offset= n * input_data_size;
  memset(col_data, 0., (col_data_buffer_size * sizeof(Dtype)));
  // First, im2col
  im2col_cpu(bottom_data + bottom_offset, channels_, height_,
             width_, kernel_size_, pad_, stride_, col_data);
  // Second, innerproduct with groups
  int top_offset_n= n* (num_output_ * height_out * width_out);
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + top_offset_n  + top_offset * g);
  }
  // third, add bias
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
         N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
         reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
         (Dtype)1., top_data + top_offset_n);
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
              vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data= (*top)[0]->mutable_cpu_data();
#pragma omp parallel for  //  shared(bottom,top)
  for (int n = 0; n < num_; ++n) {
    Forward_cpu_task(bottom_data, top_data, weight, n);
  }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu_task(
      const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff,
      const Dtype* weight, const bool propagate_down, int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int height_out = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  int width_out  = (width_  + 2 * pad_ - kernel_size_) / stride_ + 1;

  int tid = 0;
  //  tid = n%num_of_threads_;
#ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Backward_cpu: omp_thread_num() =" << tid
               << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif
  Dtype* col_data = & col_buffer_mt_[ tid *
         (channels_ * kernel_size_ * kernel_size_ * height_out * width_out)];
  Dtype* weight_diff_data= & weight_diff_mt_[tid *
         (num_output_ * (channels_ / group_) *  kernel_size_ * kernel_size_)];
  // since we saved memory in the forward pass by not storing all col data,
  // we will need to recompute them.
  int bottom_offset = channels_ * height_ * width_;
  im2col_cpu(bottom_data + bottom_offset * n, channels_, height_,
             width_, kernel_size_, pad_, stride_, col_data);
  //  gradient w.r.t. weight. Note that we will accumulate diffs.
  int top_offset_n =  num_output_ * height_out * width_out;
  for (int g = 0; g < group_ ; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top_offset_n *n  + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff_data + weight_offset * g);
  }
  // gradient w.r.t. bottom data, if necessary
  if (propagate_down) {
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
        (Dtype)1., weight + weight_offset * g,
        top_diff + top_offset_n *n + top_offset * g,
        (Dtype)0., col_data + col_offset * g);
    }
    // col2im back to the data
    col2im_cpu(col_data, channels_, height_, width_, kernel_size_, pad_,
               stride_, bottom_diff + bottom_offset * n);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0., sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
          bias_diff);
    }
  }  //  end of bias_term_
  //  clean weight_diff_buffers before back propagation
  memset(& weight_diff_mt_[0], 0., (num_of_threads_ * num_output_ *
         (channels_/ group_)* kernel_size_ * kernel_size_ * sizeof(Dtype)));

  //  do back propagation
#pragma omp parallel for
  for (int n = 0; n < num_; ++n) {
    Backward_cpu_task(top_diff, bottom_data, bottom_diff, weight,
                      propagate_down, n);
  }
//  #pragma omp barrier
  //  merge weights_diff_buffers--------------------
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int weight_diff_size = num_output_ *
                        (channels_ / group_) * kernel_size_*kernel_size_;
  memset(weight_diff, 0., ( weight_diff_size*sizeof(Dtype)));
  int j = 0;
  for (int tid = 0; tid < num_of_threads_; tid++) {
#pragma simd
    for (int i = 0; i < weight_diff_size; i++, j++) {
      weight_diff[i] += weight_diff_mt_[j];
    }
  }
}

/*
template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  if (num_of_threads_ > 0)
     Forward_cpu_omp(bottom,top);
// single thread version
 else {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  for (int n = 0; n < num_; ++n) {
    // First, im2col
    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
                      width_, kernel_size_, pad_, stride_, col_data);
    // Second, innerproduct with groups
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + (*top)[0]->offset(n) + top_offset * g);
    }
    // third, add bias
    if (bias_term_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
 }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (num_of_threads_ > 0)
   Backward_cpu_omp(top,propagate_down, bottom);
 else {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  Dtype* col_diff = col_buffer_.mutable_cpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
          1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
          bias_diff);
    }
  }
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < num_; ++n) {
    // since we saved memory in the forward pass by not storing all col data,
    // we will need to recompute them.
    im2col_cpu(bottom_data + (*bottom)[0]->offset(n), channels_, height_,
                      width_, kernel_size_, pad_, stride_, col_data);
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
    }
    // gradient w.r.t. bottom data, if necessary
    if (propagate_down) {
      for (int g = 0; g < group_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top[0]->offset(n) + top_offset * g,
          (Dtype)0., col_diff + col_offset * g);
      }
      // col2im back to the data
      col2im_cpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
          stride_, bottom_diff + (*bottom)[0]->offset(n));
    }
  }
 }
}
*/
INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
