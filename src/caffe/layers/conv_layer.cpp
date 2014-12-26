
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>  // for max
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
  is_1x1_ = (kernel_w_ == 1) && (kernel_h_ == 1) &&
             (stride_h_ == 1) && (stride_w_ == 1) &&
              (pad_h_ == 0) && (pad_w_ == 0);
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
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
    // outputChannels x inputChannels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        num_output_, channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases:
    // 1 x 1 x 1 x output channels
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
  height_out_ =
      (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = num_output_ / group_;
  K_ = channels_ * kernel_h_ * kernel_w_ / group_;
  N_ = height_out_ * width_out_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_.Reshape(
      1, channels_ * kernel_h_ * kernel_w_, height_out_, width_out_);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  // ---- openmp ------------------------------------------
  num_of_threads_ = 1;
#ifdef _OPENMP
  num_of_threads_ = omp_get_max_threads();
  if (num_of_threads_ < 1) {
     LOG(WARNING) << "Conv layer: omp_get_max_threads() =" << num_of_threads_;
     num_of_threads_ = 1;
  }
#endif
  col_buffer_mt_.resize(num_of_threads_ *
      channels_ * kernel_h_ * kernel_w_ * height_out_ * width_out_);
  weight_diff_mt_.resize(num_of_threads_ *
     num_output_ * (channels_ / group_) * kernel_h_* kernel_w_);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu_task(
      const Dtype* bottom_data, Dtype* top_data,
      Dtype* col_buff, const Dtype* weight, int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;

  Dtype* col_data = NULL;

  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Forward_cpu: omp_thread_num() =" << tid
               << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif

  int col_data_buffer_size = channels_ * kernel_h_ * kernel_w_ *
                             height_out_ * width_out_;
  int input_data_size= channels_* height_* width_;
  int bottom_offset= n * input_data_size;
  if (!is_1x1_) {
    col_data = & col_buffer_mt_[ tid* col_data_buffer_size];
    // im2col transformation: unroll input regions for filtering
    // into column matrix for multiplication.
    im2col_cpu(bottom_data + bottom_offset, channels_, height_,
        width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
        col_data);
  } else {  // special case for 1x1 convolution
    col_data = col_buff + bottom_offset;
  }
  // Take inner products for groups.
  int top_offset_n= n* (num_output_ * height_out_ * width_out_);
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
        (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
        (Dtype)0., top_data + top_offset_n  + top_offset * g);
  }
  // Add bias.
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
         N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
         reinterpret_cast<const Dtype*>(bias_multiplier_.cpu_data()),
         (Dtype)1., top_data + top_offset_n);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(
       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    Dtype* col_buff = bottom[i]->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for  //  shared(bottom,top)
#endif
    for (int n = 0; n < num_; ++n) {
      Forward_cpu_task(bottom_data, top_data, col_buff, weight, n);
    }
  }
}

// BACKWARD ===================================================================

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu_bottom_diff_task(
      const Dtype* top_diff, Dtype* bottom_diff,
       const Dtype* weight, int i,  int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int height_out = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  int width_out  = (width_  + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Backward_cpu: omp_thread_num() =" << tid
               << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif
  Dtype* col_data = NULL;
  int bottom_offset = channels_ * height_ * width_;
  if (!is_1x1_) {
    col_data = & col_buffer_mt_[tid *
         (channels_*kernel_h_*kernel_w_*height_out*width_out)];
  } else {
    col_data = bottom_diff + bottom_offset * n;
  }
  int top_offset_n =  num_output_ * height_out * width_out;
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
        (Dtype)1., weight + weight_offset * g,
        top_diff + top_offset_n *n + top_offset * g,
        (Dtype)0., col_data + col_offset * g);
  }
  // col2im back to the data
  if (!is_1x1_) {
    col2im_cpu(col_data, channels_, height_, width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_,
        stride_h_, stride_w_, bottom_diff + bottom_offset*n);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu_weight_diff_task(
        const Dtype* top_diff, const vector<Blob<Dtype>*>& bottom,
         int i, int n) {
  int height_out = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  int width_out  = (width_  + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int tid = 0;
  const Dtype* bottom_data = bottom[i]->cpu_data();
#ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Backward_cpu: omp_thread_num() =" << tid
               << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif
  Dtype* col_data = & col_buffer_mt_[tid *
         (channels_*kernel_h_*kernel_w_*height_out*width_out)];
  Dtype* weight_diff_data= & weight_diff_mt_[tid *
         (num_output_ * (channels_ / group_) * kernel_h_*kernel_w_)];
  // since we saved memory in the forward pass by not storing all col data,
  // we will need to recompute them.
  if (!is_1x1_) {
    im2col_cpu(bottom_data + bottom[i]->offset(n), channels_, height_,
                width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                stride_h_, stride_w_, col_data);
  } else {
    col_data = bottom[i]->mutable_cpu_data() + bottom[i]->offset(n);
  }
  //  gradient w.r.t. weight. Note that we will accumulate diffs.
  int top_offset_n =  num_output_ * height_out * width_out;

  if (this->param_propagate_down_[0]) {
    for (int g = 0; g < group_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
          (Dtype)1., top_diff + top_offset_n *n  + top_offset * g,
           col_data + col_offset * g, (Dtype)1.,
           weight_diff_data + weight_offset * g);
    }
  }
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_memset(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  // ---- compute weight_diff -----------------------------
  int weight_diff_size = num_output_ * (channels_ / group_) *
         kernel_h_ * kernel_w_;
  caffe_memset(weight_diff_size * sizeof(Dtype), 0., weight_diff);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->cpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.cpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
    // const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

    // ---- compute weight_diff -----------------------------
      caffe_memset(num_of_threads_ * weight_diff_size *
              sizeof(Dtype), 0., & weight_diff_mt_[0]);
      if (this->param_propagate_down_[0]) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int n = 0; n < num_; ++n) {
          Backward_cpu_weight_diff_task(top_diff, bottom, i, n);
        }
      // sum weight_diff over all threads
        for (int t = 0; t < num_of_threads_ ; ++t) {
          for (int j = 0; j < weight_diff_size ; ++j) {
            weight_diff[j] += weight_diff_mt_[t * weight_diff_size + j];
          }
        }
      }
      // ------- back propagate top_diff to bottom_diff -------
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->cpu_data();
        }
        // back-prop by gemm
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int n = 0; n < num_; ++n) {
          Backward_cpu_bottom_diff_task(top_diff, bottom_diff, weight, i, n);
        }
      }  // end of propagate down
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif


INSTANTIATE_CLASS(ConvolutionLayer);
}  // namespace caffe
