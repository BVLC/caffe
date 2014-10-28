
#include <cufft.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>  // for max
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT<Dtype>() {
  if (fft_on_ )
    fft_clean();
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
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

  // ---- fft ---------------------------------------------
  fft_on_ = true;
  if (fft_on_)
    fft_setup();
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_setup() {
  fft_height_ = height_ + std::max(2*pad_h_, (kernel_h_ - 1));
  fft_width_  = width_  + std::max(2*pad_w_, (kernel_w_- 1));

  if ((fft_height_ % 16) > 0)
    fft_height_ = fft_height_ + (16 - (fft_height_ % 16));
  if ((fft_width_ % 16) > 0)
    fft_width_ = fft_width_ + (16 - (fft_width_ % 16));

  fft_map_real_size_ = fft_height_ * fft_width_;
  fft_map_complex_size_ = fft_height_ * (fft_width_/2 +1);
  height_out_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  width_out_  = (width_  + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  map_size_ = height_ * width_;
  map_out_size_ = height_out_ * width_out_;

  // allocate buffers
  fft_clean();
  switch (Caffe::mode()) {
    case Caffe::CPU:
      fft_cpu_setup();
      break;
    case Caffe::GPU:
      fft_gpu_setup();
      break;
  }
}
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_cpu_setup() {
  if (fft_cpu_initialized_)
    fft_cpu_clean();

  // allocate buffers for fft
  int num_weights = num_output_ * (channels_ / group_);

  fft_weights_complex_ = (std::complex<Dtype> *)  caffe_cpu_fft_malloc<Dtype>(
        num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype> ));
  fft_map_in_real_ = reinterpret_cast<Dtype *> (caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * fft_map_real_size_ * sizeof(Dtype)));
  fft_map_in_complex_ = (std::complex<Dtype> *) caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * fft_map_complex_size_ * sizeof(std::complex<Dtype>));
  fft_map_out_complex_ = (std::complex<Dtype>*) caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * std::max(num_output_, channels_) *
        fft_map_complex_size_ * sizeof(std::complex<Dtype>));
  fft_map_out_real_ = reinterpret_cast<Dtype *> (caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * std::max(num_output_, channels_)*
        fft_map_real_size_ * sizeof(Dtype)));

  //  fftw plans
    fft_handle_ = caffe_cpu_fft_plan_dft_r2c_2d<Dtype>(fft_height_, fft_width_,
        fft_map_in_real_, fft_map_in_complex_, FFTW_ESTIMATE);
  ifft_handle_ = caffe_cpu_fft_plan_dft_c2r_2d<Dtype>(fft_height_, fft_width_,
        fft_map_out_complex_, fft_map_out_real_, FFTW_ESTIMATE);

  // plan for fft_many
  int in_N[2];
  in_N[0] = fft_height_;
  in_N[1] = fft_width_;
  int in_stride = 1;
  int out_N[2];
  out_N[0] = fft_height_;
  out_N[1] = fft_width_/2 +1;
  int out_stride = 1;
  int out_dist = fft_height_ * (fft_width_/2 +1);

  //--- in place fft-----------------------------------------------------
  int in_dist_inplace = fft_height_ * 2*(fft_width_/2+1);
  int in_N_inplace[2];
  in_N_inplace[0] = fft_height_;
  in_N_inplace[1] = 2 * (fft_width_/2+1);
  fft_many_handle_ = caffe_cpu_fft_plan_many_dft_r2c<Dtype>(2, in_N,
         num_weights, reinterpret_cast<Dtype*>(fft_weights_complex_),
          in_N_inplace, in_stride, in_dist_inplace, fft_weights_complex_,
           out_N, out_stride, out_dist, FFTW_ESTIMATE);

  // for openMP
#ifdef _OPENMP
  caffe_cpu_fft_init_threads();
  caffe_cpu_fft_plan_with_nthreads(num_of_threads_);
#endif
  fft_cpu_initialized_ = true;
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_clean() {
  if ( fft_cpu_initialized_)
    fft_cpu_clean();
  if ( fft_gpu_initialized_)
    fft_gpu_clean();
}

// free FFT buffers--------------------------------------------------
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_cpu_clean() {
  if (fft_cpu_initialized_) {
    caffe_cpu_fft_free<Dtype>(fft_map_in_real_);
    caffe_cpu_fft_free<Dtype>(fft_map_in_complex_);
    caffe_cpu_fft_free<Dtype>(fft_weights_complex_);
    caffe_cpu_fft_free<Dtype>(fft_map_out_complex_);
    caffe_cpu_fft_free<Dtype>(fft_map_out_real_);
    caffe_cpu_fft_destroy_plan<Dtype>(fft_handle_);
    caffe_cpu_fft_destroy_plan<Dtype>(ifft_handle_);
    caffe_cpu_fft_destroy_plan<Dtype>(fft_many_handle_);
#ifdef _OPENMP
    caffe_cpu_fft_cleanup_threads();
#endif
  }
  fft_cpu_initialized_ = false;
}

//=====================Forward CPU========================

//  prepare fft of weights ------------------------------------------
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_compute_weights() {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  // 0-paddinng before FFT ----------------------
  caffe_memset((num_output_*(channels_ / group_) * fft_map_complex_size_ *
            sizeof(std::complex<Dtype>)), 0., fft_weights_complex_);
  int ch_gr = (channels_/group_);
#pragma omp parallel for
  for (int n = 0; n < num_output_; n++) {
    for (int c = 0; c < ch_gr; c++)
      for (int h = 0; h < kernel_h_; h++)
        for (int w = 0; w < kernel_w_; w++ )
          (reinterpret_cast<Dtype*>
            (fft_weights_complex_))[((n * ch_gr + c) * fft_height_ + h)* 2
               * (fft_width_ / 2 + 1) + w] =
               weight[((n * ch_gr + c) * kernel_h_ + h) * kernel_w_ + w];
  }
  //  do FFT for all weights -----------------------------------
  caffe_cpu_fft_execute<Dtype>(fft_many_handle_);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft_task(
      const Dtype* bottom_data, Dtype* top_data,  int n) {
  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Forward_cpu_fft_task: omp_get_thread_num()="
               << tid << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif
  //  buffers per thread ---------------------
  int map_in_size = height_* width_;
  Dtype* map_in_n = const_cast<Dtype*>
                     (bottom_data + n * channels_ * map_in_size);
  Dtype* fft_map_in_real_n = fft_map_in_real_ + tid * fft_map_real_size_;
  std::complex<Dtype>* fft_map_in_complex_n =
          fft_map_in_complex_ + tid * fft_map_complex_size_;
  std::complex<Dtype>* fft_map_out_complex_n =
          fft_map_out_complex_ + tid * (num_output_*fft_map_complex_size_);
  Dtype* fft_map_out_real_n =
          fft_map_out_real_ + tid * (num_output_ * fft_map_real_size_);
  Dtype* map_out_n = top_data + n* (num_output_ * height_out_ * width_out_);
  // clear buffers
  caffe_memset((num_output_*fft_map_complex_size_*
          sizeof(std::complex<Dtype>)), 0., fft_map_out_complex_n);

  // loop over all channels ---------------------
  Dtype* map_in;
  std::complex<Dtype>* weights_complex;
  std::complex<Dtype>* map_out_complex;
  Dtype* map_out_real;
  Dtype* map_out;
  for (int c = 0; c < channels_; c++) {
    map_in = map_in_n + c * map_in_size;

    //  0-padding: map_in --> fft_map_in_real -------------
    caffe_memset((fft_map_real_size_*sizeof(Dtype)), 0., fft_map_in_real_n);
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++ )
      fft_map_in_real_n[(h + pad_h_)* fft_width_ + (w + pad_w_)] =
              map_in[h * width_ + w ];
    }

    // FFT: map_in_real --> map_in_complex
    caffe_cpu_fft_execute_dft_r2c<Dtype>(fft_handle_,
                fft_map_in_real_n, fft_map_in_complex_n);
    int g = c / (channels_ / group_);        // channel group
    int c_offset= c % (channels_ / group_);  // channel_index inside group
    int out_first = g* (num_output_ / group_);
    int out_last = (g+1)*(num_output_ / group_);
    for (int out = out_first; out < out_last; out++) {
      map_out_complex = fft_map_out_complex_n + out * fft_map_complex_size_;
      weights_complex = fft_weights_complex_ +
               (out * (channels_/group_) + c_offset) * fft_map_complex_size_;
      for (int i = 0; i < fft_map_complex_size_; i++) {
        // fft for correlation requires conj (fft_of_weights)
        Dtype x_real = std::real(fft_map_in_complex_n[i]);
        Dtype x_imag = std::imag(fft_map_in_complex_n[i]);
        Dtype y_real = std::real(weights_complex[i]);
        Dtype y_imag = std::imag(weights_complex[i]);
        Dtype z_real = x_real*y_real + x_imag*y_imag;
        Dtype z_imag = - x_real*y_imag + x_imag*y_real;
        map_out_complex[i].real() += z_real;
        map_out_complex[i].imag() += z_imag;
      }
    }
  }

  // IFFT: map_out_complex --> map_out_real
  // OPTION: fft_many ?
  Dtype ifft_scale = 1./((Dtype) fft_map_real_size_);
  for (int out = 0; out < num_output_; out++) {
    map_out_complex = fft_map_out_complex_n  + out * fft_map_complex_size_;
    map_out_real = fft_map_out_real_n + out * fft_map_real_size_;
    map_out = map_out_n + out * map_out_size_;
    caffe_cpu_fft_execute_dft_c2r<Dtype>(ifft_handle_,
             map_out_complex, map_out_real);
    //  post-process: map_out_real --> map_out
    int h, w;
    for (int h_out = 0; h_out < height_out_; h_out++) {
      for (int w_out = 0; w_out < width_out_; w_out++) {
        h = h_out  * stride_h_;
        w = w_out  * stride_w_;
        if ((h < fft_height_) &&  (w < fft_width_)) {
            map_out[h_out*width_out_ + w_out] =
              (ifft_scale * map_out_real[h*fft_width_ + w]);
        }
      }
    }
  }
  //  bias
  if (bias_term_) {
    int top_offset_n= n* (num_output_ * height_out_* width_out_);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_.cpu_data()),
          (Dtype)1., top_data + top_offset_n);
  }
}

template <typename Dtype>
Dtype ConvolutionLayerFFT<Dtype>::Forward_cpu_fft(
     const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (fft_cpu_initialized_ != true ) fft_setup();

  fft_compute_weights();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data= top[i]->mutable_cpu_data();
#pragma omp parallel for
    for (int n = 0; n < num_; ++n) {
      Forward_cpu_fft_task(bottom_data, top_data, n);
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_task(
      const Dtype* bottom_data, Dtype* top_data,
      Dtype* col_buff, const Dtype* weight, int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int height_out = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  int width_out  = (width_  + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
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
                             height_out * width_out;
  int input_data_size= channels_* height_* width_;
  int bottom_offset= n * input_data_size;
  if (!is_1x1_) {
    col_data = & col_buffer_mt_[ tid* col_data_buffer_size];;
  }
  caffe_memset((col_data_buffer_size * sizeof(Dtype)), 0., col_data);
  // im2col transformation: unroll input regions for filtering
  // into column matrix for multplication.
  if (!is_1x1_) {
    im2col_cpu(bottom_data + bottom_offset, channels_, height_,
        width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
        col_data);
  } else {  // special case for 1x1 convolution
    col_data = col_buff + bottom_offset;
  }
  // Take inner products for groups.
  int top_offset_n= n* (num_output_ * height_out * width_out);
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
void ConvolutionLayerFFT<Dtype>::Forward_cpu(
       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  if (fft_on_) {
    Forward_cpu_fft(bottom, top);
  } else {
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      Dtype* col_buff = bottom[i]->mutable_cpu_data();
#pragma omp parallel for  //  shared(bottom,top)
      for (int n = 0; n < num_; ++n) {
        Forward_cpu_task(bottom_data, top_data, col_buff, weight, n);
      }
    }
  }
}

// BACKWARD ===================================================================

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu_fft_task(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
         const Dtype* weight, int i, int n) {
  const Dtype* top_diff = top[i]->cpu_diff();
  Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
  //  back propagation is from top to bottom: the top is the inputs.
  int tid = 0;
  #ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
      LOG(FATAL) << "ConvLayer::Backward_cpu_fft: omp_get_thread_num()="<< tid
                    << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif

  int map_in_size = height_out_* width_out_;
  Dtype* map_in_n = const_cast<Dtype*>(top_diff + top[i]->offset(n));
  Dtype* fft_map_in_real_n = fft_map_in_real_ + tid * fft_map_real_size_;
  std::complex<Dtype>* fft_map_in_complex_n =
         fft_map_in_complex_ + tid * fft_map_complex_size_;
  std::complex<Dtype>* fft_map_out_complex_n =
         fft_map_out_complex_ + tid * (channels_ * fft_map_complex_size_);
  Dtype* fft_map_out_real_n =
         fft_map_out_real_ + tid * (channels_ * fft_map_real_size_);
  Dtype* map_out_n = reinterpret_cast<Dtype*>
                      (bottom_diff + bottom[i]->offset(n));
  // clear buffers
  caffe_memset(fft_map_real_size_ * sizeof(Dtype), 0., fft_map_in_real_n);
  caffe_memset((channels_ * fft_map_complex_size_*
                 sizeof(std::complex<Dtype>)), 0., fft_map_out_complex_n);

  // loop over all outputs ---------------------
  Dtype* map_in;
  std::complex<Dtype>* weights_complex;
  std::complex<Dtype>* map_out_complex;
  Dtype* map_out_real;
  Dtype* map_out;
  for (int out = 0; out < num_output_; out++) {
    map_in = map_in_n + out * map_in_size;
    //  0-padding: map_out --> fft_map_in_real -------------
    for (int h = 0; h < height_out_; h++) {
      for (int w = 0; w < width_out_; w++) {
        int h_pad = h * stride_h_;
        int w_pad = w * stride_w_;
        fft_map_in_real_n[h_pad*fft_width_ + w_pad] =
              map_in[h*width_out_ + w  ];
      }
    }
    // FFT: map_in_real --> map_in_complex
    caffe_cpu_fft_execute_dft_r2c<Dtype>(fft_handle_,
              fft_map_in_real_n, fft_map_in_complex_n);
    int g = out / (num_output_ / group_);           //  group
    int c_first = g* (channels_ / group_);
    int c_last = (g+1)*(channels_ / group_);
    for (int c = c_first; c < c_last; c++) {
      int c_offset = c % (channels_ / group_);
      map_out_complex = fft_map_out_complex_n + c * fft_map_complex_size_;
      weights_complex = fft_weights_complex_   +
                 (out * (channels_/group_) + c_offset) * fft_map_complex_size_;
      for (int i = 0; i < fft_map_complex_size_; i++) {
        Dtype x_real = std::real(fft_map_in_complex_n[i]);
        Dtype x_imag = std::imag(fft_map_in_complex_n[i]);
        Dtype y_real = std::real(weights_complex[i]);
        Dtype y_imag = std::imag(weights_complex[i]);
        Dtype z_real = x_real*y_real - x_imag*y_imag;
        Dtype z_imag = x_real*y_imag + x_imag*y_real;
        map_out_complex[i].real() += z_real;
        map_out_complex[i].imag() += z_imag;
     }
    }
  }
  // IFFT: map_out_complex --> map_out_real
  // OPTION:  fft many ?
  Dtype ifft_scale = 1./((Dtype) fft_map_real_size_);
  for (int c = 0; c < channels_; c++) {
    map_out_complex = fft_map_out_complex_n  + c * fft_map_complex_size_;
    map_out_real = fft_map_out_real_n;
    caffe_cpu_fft_execute_dft_c2r<Dtype>(ifft_handle_,
            map_out_complex, map_out_real);
    //  post-process: map_out_real --> map_out
    map_out = map_out_n + c * map_size_;
    int h, w;
    for (int h_out = 0; h_out < height_; h_out++) {
      for (int w_out = 0; w_out < width_; w_out++) {
        h = h_out + pad_h_;
        w = w_out + pad_w_;
        map_out[h_out*width_ + w_out] =
             (ifft_scale * map_out_real[h*fft_width_ + w]);
      }
    }
  }
}

// regular backward propagation by im2col and gemm  ------------------------
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu_bottom_diff_task(
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

//-------------------------------------------------------
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu_weight_diff_task(
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

//-----------------------------------------------------------------------------

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
#pragma omp parallel for
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
        if (fft_on_) {
          if (!fft_cpu_initialized_) fft_setup();
          // WARNING: Assume fft for weights was computed in Forward
          // This assumption can fail in runtest, if only Backward() is called!
          // fft_compute_weights();
#pragma omp parallel for
          for (int n = 0; n < num_; ++n) {
            Backward_cpu_fft_task(bottom, top, weight, i, n);
          }
        } else {  // back-prop by gemm
#pragma omp parallel for
          for (int n = 0; n < num_; ++n) {
            Backward_cpu_bottom_diff_task(top_diff, bottom_diff,
             weight, i, n);
          }
        }
      }  // end of propagate down
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
