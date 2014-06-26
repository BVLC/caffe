// Copyright 2014 BVLC and contributors.
// FFT version of convolutional layer: boris.ginzburg@intel.com
// For algorithmic details see: [1] Michael Mathieu, Mikael Henaff, Yann LeCun.
// Fast Training of Convolutional Networks through FFTs. arXiv:1312.5851, 2013.
//-----------------------------------------------------------------------------

#include <omp.h>
#include <vector>
#include <algorithm>  // for max

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
//  Layer<Dtype>::SetUp(bottom, top);
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
  K_ = channels_ * kernel_size_ * kernel_size_ / group_;
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

  // fft
  fft_on_ = false;
  if ( (kernel_size_  > 5) &&  ((kernel_size_ /stride_) > 5) )
    fft_on_ = true;
  if (fft_on_)
    fft_setup();
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::FFT_on() {
	fft_on_ = true;
	fft_setup();
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::FFT_off() {
	fft_clean();
	fft_on_ = false;
}

template <typename Dtype>
ConvolutionLayer<Dtype>::~ConvolutionLayer<Dtype>() {
  if (fft_on_ && fft_initialized_)
    fft_clean();
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_setup() {
  if (fft_initialized_)
    fft_clean();
  else
    fft_initialized_ = true;
#ifdef _OPENMP
  caffe_cpu_fft_init_threads();
// #ifdef USE_MKL
//  fftw3_mkl.number_of_user_threads = 4;
// #endif
  caffe_cpu_fft_plan_with_nthreads(num_of_threads_);
#endif
  fft_height_ = height_ + std::max(2*pad_, (kernel_size_ - 1));
  fft_width_  = width_  + std::max(2*pad_, (kernel_size_ - 1));

  fft_map_real_size_ = fft_height_ * fft_width_;
  fft_map_complex_size_ = fft_height_ * (fft_width_/2 +1);
  height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  width_out_  = (width_  + 2 * pad_ - kernel_size_) / stride_ + 1;
  map_out_size_ = height_out_ * width_out_;
  int num_weights = num_output_ * (channels_ / group_);

  //  common buffers
  fft_weights_real_= (Dtype *) caffe_cpu_fft_malloc<Dtype>(
        num_weights * fft_map_real_size_ * sizeof(Dtype));
  fft_weights_complex_ = (std::complex<Dtype> *)  caffe_cpu_fft_malloc<Dtype>(
        num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype> ));
  fft_map_in_real_ = (Dtype *) caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * fft_map_real_size_ * sizeof(Dtype));
  fft_map_in_complex_ = (std::complex<Dtype> *) caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * fft_map_complex_size_ * sizeof(std::complex<Dtype>));
  fft_map_out_complex_ = (std::complex<Dtype>*) caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * num_output_ * fft_map_complex_size_ *
        sizeof(std::complex<Dtype>));
  fft_map_out_real_ = (Dtype *) caffe_cpu_fft_malloc<Dtype>(
        num_of_threads_ * num_output_ * fft_map_real_size_ * sizeof(Dtype));

  //  fftw plans
  fft_handle_ = caffe_cpu_fft_plan_dft_r2c_2d<Dtype>(fft_height_, fft_width_,
        fft_map_in_real_, fft_map_in_complex_, FFTW_ESTIMATE);
  ifft_handle_ = caffe_cpu_fft_plan_dft_c2r_2d<Dtype>(fft_height_, fft_width_,
        fft_map_out_complex_, fft_map_out_real_, FFTW_ESTIMATE);
  // plan for fft_many
  int in_N[2];
  in_N[0] = fft_height_;
  in_N[1] = fft_width_;
  LOG(INFO) << "Conv layer: FFT_height=" << fft_height_ << ", FFT_width =" << fft_width_;
  LOG(INFO) << "Conv layer: num threads_=" << num_of_threads_;
  int in_stride = 1;
  int in_dist = fft_height_ * fft_width_;
  int out_N[2];
  out_N[0] = fft_height_;
  out_N[1] = fft_width_/2 +1;
  int out_stride = 1;
  int out_dist = fft_height_ * (fft_width_/2 +1);
  int num_filters  = num_output_ * (channels_/group_);
  fft_many_handle_ = caffe_cpu_fft_plan_many_dft_r2c<Dtype>(2, in_N,
      num_filters, fft_weights_real_, in_N, in_stride, in_dist,
      fft_weights_complex_, out_N, out_stride, out_dist, FFTW_ESTIMATE);
}

// free FFT buffers--------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_clean() {
  if (fft_initialized_) {
	  fft_initialized_ = false;
    caffe_cpu_fft_free<Dtype>(fft_map_in_real_);
	caffe_cpu_fft_free<Dtype>(fft_weights_real_);
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
}

//  prepare fft of weights ------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_doFFT_weights() {
  const Dtype *weight = this->blobs_[0]->cpu_data();
  // 0-paddinng before FFT ----------------------
  memset(fft_weights_real_, 0.,
          (num_output_*(channels_ / group_)*fft_map_real_size_*sizeof(Dtype)));
  int ch_gr = (channels_/group_);
#pragma omp parallel for
  for (int n = 0; n < num_output_; n++) {
    for (int c = 0; c < ch_gr; c++)
      for (int h = 0; h < kernel_size_; h++)
        for (int w = 0; w < kernel_size_; w++ )
          fft_weights_real_[((n*ch_gr + c)*fft_height_ + h)* fft_width_ + w] =
             weight[((n*ch_gr + c)*kernel_size_ + h)*kernel_size_ + w];
  }
  //  do  FFT -----------------------------------
  Dtype*  filter_pad = fft_weights_real_;
  std::complex<Dtype>* filter_fft = fft_weights_complex_;
  int num_filters  = num_output_ * (channels_/group_);
//  caffe_cpu_fft_execute<Dtype>(fft_many_handle_);
#pragma omp parallel for
  for (int w = 0; w < num_filters ; w++) {
    filter_pad = fft_weights_real_ + w * fft_map_real_size_;
    filter_fft = fft_weights_complex_+ w * fft_map_complex_size_;
    caffe_cpu_fft_execute_dft_r2c<Dtype>(fft_handle_, filter_pad, filter_fft);
  }
  // complex conjugate for correlation ----------
  int num_weights = num_output_ *(channels_/group_)* fft_map_complex_size_;
#pragma omp parallel for
  for (int i = 0; i < num_weights; i++)
    fft_weights_complex_[i]= std::conj(fft_weights_complex_[i]);
}

//-------------------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu_fft_task(
      const Dtype* bottom_data, Dtype* top_data, const Dtype* weight, int n) {
  int tid = 0;
#ifdef _OPENMP
  tid = omp_get_thread_num();
  if (tid >= num_of_threads_)
    LOG(FATAL) << "ConvLayer::Forward_cpu_fft: omp_get_thread_num()="<< tid
               << " > OMP_num_THREADS = " << num_of_threads_;
  tid = tid % num_of_threads_;  //  just to be sure
#endif
  // fprintf(stderr,"fft: num_threads = %d tid= %d\n ", num_of_threads_, tid );
  // buffers per thread ---------------------
  int map_in_size = height_* width_;
  Dtype* map_in_n = (Dtype*) (bottom_data + n * channels_ * map_in_size);
  Dtype* fft_map_in_real_n = fft_map_in_real_ + tid * fft_map_real_size_;
  std::complex<Dtype>* fft_map_in_complex_n =
          fft_map_in_complex_ + tid * fft_map_complex_size_;
  std::complex<Dtype>* fft_map_out_complex_n =
          fft_map_out_complex_ + tid * (num_output_*fft_map_complex_size_);
  Dtype* fft_map_out_real_n =
          fft_map_out_real_ + tid * (num_output_ * fft_map_real_size_);
  Dtype* map_out_n = top_data + n* (num_output_ * height_out_ * width_out_);
  // clear buffers
  memset(fft_map_out_complex_n, 0.,
          (num_output_*fft_map_complex_size_* sizeof(std::complex<Dtype>)));
  // loop over all channels ---------------------
  Dtype* map_in;
  std::complex<Dtype>* map_in_complex;
  std::complex<Dtype>* weights_complex;
  std::complex<Dtype>* map_out_complex;
  Dtype* map_out_real;
  Dtype* map_out;
  for (int c = 0; c < channels_; c++) {
    map_in = map_in_n + c * map_in_size;
    //  0-padding: map_in --> fft_map_in_real -------------
    memset(fft_map_in_real_n, 0., (fft_map_real_size_*sizeof(Dtype)));
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++ )
        fft_map_in_real_n[(h + pad_)* fft_width_ + (w + pad_)] =
                map_in[h * width_ + w ];
    }
    // FFT: map_in_real --> map_in_complex
    caffe_cpu_fft_execute_dft_r2c<Dtype>(fft_handle_,
                fft_map_in_real_n, fft_map_in_complex_n);
    // fft_map_out_complex[c] += fft_map_in_complex * fft_weight_complex[n,c]
    int g = c / (channels_ / group_);        // channel group
    int c_offset= c % (channels_ / group_);  // channel_index inside group
    int out_first = g* (num_output_ / group_);
    int out_last = (g+1)*(num_output_ / group_);
    for (int out = out_first; out < out_last; out++) {
      map_out_complex = fft_map_out_complex_n + out * fft_map_complex_size_;
      weights_complex = fft_weights_complex_ +
               (out * (channels_/group_) + c_offset) * fft_map_complex_size_;
#pragma simd
      for (int i = 0; i < fft_map_complex_size_; i++) {
//      map_out_complex[i]+= fft_map_in_complex_n[i] * weights_complex[i];
        Dtype x_real = std::real(fft_map_in_complex_n[i]);
        Dtype x_imag = std::imag(fft_map_in_complex_n[i]);
        Dtype y_real = std::real(weights_complex[i]);
        Dtype y_imag = std::imag(weights_complex[i]);
        Dtype z_real = x_real*y_real - x_imag*y_imag ;
        Dtype z_imag = x_real*y_imag + x_imag*y_real;
       	map_out_complex[i] =
       	      std::complex <Dtype> (std::real(map_out_complex[i]) + z_real,
                                    std::imag(map_out_complex[i]) + z_imag);
      }
    }
  }
  // IFFT: map_out_complex --> map_out_real
  // TODO: should we change to fft_execute (many) ?
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
        h = h_out * stride_;
        w = w_out * stride_;
        if ((h < fft_height_) &&  (w < fft_width_))
          map_out[h_out*width_out_ + w_out] =
             (ifft_scale * map_out_real[h*fft_width_ + w]);
//      else  map_out[h_out * width_out_ + w_out] = 0.;
      }
    }
  }
  //  bias
  if (bias_term_) {
    int top_offset_n= n* (num_output_ * height_out_* width_out_);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + top_offset_n);
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu_fft(
     const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data= (*top)[0]->mutable_cpu_data();
  const Dtype* weights = this->blobs_[0]->cpu_data();

  fft_doFFT_weights();
#pragma omp parallel for
  for (int n = 0; n < num_; ++n) {
    Forward_cpu_fft_task(bottom_data, top_data, weights, n);
  }
  return Dtype(0.);
}

// ------------------------------------------------------------------
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
//  int max_threads = omp_get_num_threads();
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
  if (fft_on_) {
    Forward_cpu_fft(bottom, top);
  }
  else {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype* top_data= (*top)[0]->mutable_cpu_data();
#pragma omp parallel for  //  shared(bottom,top)
    for (int n = 0; n < num_; ++n) {
      Forward_cpu_task(bottom_data, top_data, weight, n);
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
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

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
