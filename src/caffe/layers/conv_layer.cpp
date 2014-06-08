// Copyright 2014 BVLC and contributors.

#include <vector>
//  fft
#include "mkl.h"
#include "mkl_vsl.h"	// for mkl convolution
#include "mkl_dfti.h"	// for mkl fft

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include <omp.h>

namespace caffe {
template <typename Dtype>
ConvolutionLayer<Dtype>::~ ConvolutionLayer<Dtype>() {
  if (fft_on_ && fft_initialiazed_)
	  fft_free();
}

//---------------------------------------------------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_setup() {
  if (fft_initialiazed_)
    fft_free();
  else
	fft_initialiazed_ = true;

  fft_height_ = height_ + std::max(2*pad_, (kernel_size_ - 1));
  fft_width_  = width_  + std::max(2*pad_, (kernel_size_ - 1));
  fft_map_real_size_   = fft_height_ * fft_width_ ;
  fft_map_complex_size_ = fft_height_ * (fft_width_/2 +1);
  height_out_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  width_out_  = (width_  + 2 * pad_ - kernel_size_) / stride_ + 1;
  map_out_size_ = height_out_ * width_out_;
  int num_weights = num_output_ * (channels_  / group_ );
  fft_weights_real_= (float *) mkl_calloc((num_weights * fft_map_real_size_), sizeof(float), 64);
  fft_weights_complex_ = (MKL_Complex8 *) mkl_calloc((num_weights * fft_map_complex_size_), sizeof(MKL_Complex8 ), 64);
  fft_map_in_real_ = (float *) mkl_calloc(fft_map_real_size_, sizeof(float), 64);
  fft_map_in_complex_  = (MKL_Complex8 *) mkl_calloc(fft_map_complex_size_, sizeof(MKL_Complex8 ), 64);
  fft_map_out_complex_ = (MKL_Complex8 *) mkl_calloc((num_output_ * fft_map_complex_size_), sizeof(MKL_Complex8 ), 64);;
  fft_map_out_real_ = (float *) mkl_calloc((num_output_ * fft_map_real_size_), sizeof(float), 64);

  // MKL setup fft handler-----
  MKL_LONG fft_length[2];
  fft_length[0] = fft_height_;
  fft_length[1] = fft_width_;
  MKL_LONG status = 0;
  fft_handle_ = 0;
  status = DftiCreateDescriptor(&fft_handle_, DFTI_SINGLE, DFTI_REAL, 2, fft_length);
  status = DftiSetValue(fft_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  status = DftiSetValue(fft_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  MKL_LONG fft_in_strides[3];
  fft_in_strides[0] = 0;
  fft_in_strides[1] = fft_width_;
  fft_in_strides[2] = 1;
  status = DftiSetValue(fft_handle_, DFTI_INPUT_STRIDES, fft_in_strides);
  MKL_LONG fft_out_strides[3];
  fft_out_strides[0] = 0;
  fft_out_strides[1] = (fft_width_/2)+1;
  fft_out_strides[2] = 1;
  status = DftiSetValue(fft_handle_, DFTI_OUTPUT_STRIDES, fft_out_strides);
  // TODO: status = DftiSetValue(fft_handle_, DFTI_NUMBER_OF_TRANSFORMS,channels_ );
  status = DftiCommitDescriptor(fft_handle_);

  // MKL setup ifft handler --------
  status = DftiCreateDescriptor(&ifft_handle_, DFTI_SINGLE, DFTI_REAL, 2, fft_length);
  status = DftiSetValue(ifft_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  status = DftiSetValue(ifft_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  status = DftiSetValue(ifft_handle_, DFTI_OUTPUT_STRIDES, fft_in_strides);
  status = DftiSetValue(ifft_handle_, DFTI_INPUT_STRIDES, fft_out_strides);
  float ifft_scale = 1./((float) fft_map_real_size_);
  status = DftiSetValue( ifft_handle_, DFTI_BACKWARD_SCALE,ifft_scale  );
  status = DftiCommitDescriptor(ifft_handle_);    // if (0 != status) return(-1);
}

//-- borisg free FFT ----------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_free() {
  mkl_free(fft_map_in_real_);
  mkl_free(fft_weights_real_);
  mkl_free(fft_map_in_complex_);
  mkl_free(fft_weights_complex_);
  mkl_free(fft_map_out_complex_);
  mkl_free(fft_map_out_real_);
  MKL_LONG status;
  status = DftiFreeDescriptor(&fft_handle_);
  status = DftiFreeDescriptor(&ifft_handle_);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
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

  // fft -----------------------------------------------------------------
  fft_on_ = true;
  if (fft_on_)
	  fft_setup();
}

// 0-pad weights before FFT ---------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_pad_weights(){
  const Dtype *weight=this->blobs_[0]->cpu_data();
  memset(fft_weights_real_, 0.,
		  (num_output_*(channels_ / group_)*fft_map_real_size_*sizeof(float)));
  int ch_gr= (channels_ / group_);
#pragma omp parallel for
  for (int n=0; n < num_output_; n++)
  {
    for (int c=0; c < ch_gr; c++)
     for (int h=0; h < kernel_size_; h++)
       for (int w=0; w < kernel_size_; w++ )
		 fft_weights_real_[((n*ch_gr + c)* fft_height_ + h)* fft_width_ + w] =
		    (float) weight[((n*ch_gr + c)*kernel_size_ + h)*kernel_size_ + w];
  }
}

//-----------------------------------------------------------------------------
// see https://software.intel.com/sites/products/documentation/hpc/mkl/mklman/GUID-F4B8BEC6-70DD-4AB9-BBBB-DE95EBB67FA8.htm
//-----------------------------------------------------------------------------
/*
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_doFFT_weights(){
  MKL_LONG status;
  float*  filter_pad = fft_weights_real_;
  MKL_Complex8* filter_fft = fft_weights_complex_;
  for (int n=0; n < num_output_; n++){
	for (int c=0; c < (channels_/group_); c++){
	  status = DftiComputeForward(fft_handle_, filter_pad , filter_fft) ;
      filter_pad += fft_map_real_size_;
      filter_fft += fft_map_complex_size_;
	}
  }
  //---------complex conjugate for correlation ----------
  for (int i=0; i < (num_output_*(channels_/group_)*fft_map_complex_size_);i++)
    fft_weights_complex_[i].imag = - fft_weights_complex_[i].imag;
}
*/
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_doFFT_weights(){
  MKL_LONG status;
  float*  filter_pad = fft_weights_real_;
  MKL_Complex8* filter_fft = fft_weights_complex_;
  int num_filters  = num_output_ * (channels_/group_);
#pragma omp parallel for
  for (int w = 0; w < num_filters ; w++){
	  filter_pad = fft_weights_real_ + w*fft_map_real_size_;
	  filter_fft = fft_weights_complex_+ w * fft_map_complex_size_;
	  status = DftiComputeForward(fft_handle_, filter_pad , filter_fft) ;
  }
  //---------complex conjugate for correlation ----------
  int num_weights = num_output_ *( channels_ / group_)* fft_map_complex_size_;
#pragma omp parallel for
  for (int i=0; i < num_weights ;i++)
    fft_weights_complex_[i].imag = - fft_weights_complex_[i].imag;
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_clear_buffers(){
//  for (int i =0; i < fft_map_real_size_; i++)
//    fft_map_in_real_[i] =  0.;
//  for (int i =0; i < fft_map_complex_size_; i++){
//	fft_map_in_complex_[i].real = 0.;
//    fft_map_in_complex_[i].imag = 0.;
//  }
  for (int i =0; i < num_output_ * fft_map_complex_size_; i++){
	fft_map_out_complex_[i].real = 0.;
	fft_map_out_complex_[i].imag = 0.;
  }
//  for (int i =0; i < num_output_ * fft_map_real_size_; i++)
//	fft_map_out_real_[i] =  0.;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_pad_map_in(Dtype* map_in){
//  for (int i =0; i < fft_map_real_size_; i++)
//  	fft_map_in_real_[i] =  0.;
  memset(fft_map_in_real_, 0., (fft_map_real_size_*sizeof(float)));
  int h_pad, w_pad;
  for (int h=0; h < height_; h++)
    for (int w=0; w < width_; w++ ){
      h_pad = h + pad_;
      w_pad = w + pad_;
      // TOFO: redundant check?
      if ((h_pad < fft_height_) && (w_pad < fft_width_))
    	fft_map_in_real_[h_pad * fft_width_ + w_pad] =
    	                (float) map_in[h * width_ + w ];
    }
}
//-----------------------------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_mult_acc(int c){
  int weight_base,map_out_base;
  MKL_Complex8 * map_in_complex, weight_complex,map_out_complex;
  int g = c / (channels_ / group_);        // channel group
  int c_offset= c % (channels_ / group_);  // channel_index inside group
  int out_first = g* (num_output_ / group_);
  int out_last = (g+1)*(num_output_ / group_);
  for (int out=out_first; out < out_last; out++){
    map_out_base = out * fft_map_complex_size_ ;
    MKL_Complex8 * map_out_complex = fft_map_out_complex_ + map_out_base;
	weight_base  = (out * (channels_/group_) + c_offset)*fft_map_complex_size_;
    MKL_Complex8 * weights_complex=  fft_weights_complex_ + weight_base;
    for (int i=0; i < fft_map_complex_size_ ; i++ ){
      map_out_complex[i].real +=
        (fft_map_in_complex_[i].real * weights_complex[i].real -
         fft_map_in_complex_[i].imag * weights_complex[i].imag);
     map_out_complex[i].imag +=
        (fft_map_in_complex_[i].real * weights_complex[i].imag +
         fft_map_in_complex_[i].imag * weights_complex[i].real);
	}
  }
}
//----------------------------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_postprocess_map_out(float* map_out_real,
		Dtype* map_out){
  int h, w;
  for (int h_out = 0; h_out < height_out_; h_out++)
    for (int w_out = 0; w_out < width_out_; w_out++){
	 h = h_out * stride_;
     w = w_out * stride_;
     if ( (h < fft_height_) && (w < fft_width_))
       map_out[h_out * width_out_ + w_out] =
    		   (Dtype) map_out_real[h* fft_width_ + w];
     else
       map_out[h_out * width_out_ + w_out] = 0.;
    }
}
// fft debug print ------------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::fft_print_2D(Dtype *data, int height, int width){
  fprintf(stderr,"h=%d, w=%d\n", height, width);
  Dtype v=0.;
  for (int h=0; h< height ; h++){
    for (int w=0; w< width ; w ++){
	  v=data[h* width + w];
	  fprintf(stderr,"%7.3f ", v);
	}
	fprintf(stderr,"\n");
  }
  fprintf(stderr,"-------------------------\n");
}
//----------------------------------------------------------------------------
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu_fft_task(
      const Dtype* bottom_data, Dtype* top_data, const Dtype* weight, int n) {
  fft_clear_buffers();
  int map_in_size = height_* width_;
  Dtype* map_in = (Dtype*) (bottom_data + n * channels_ * map_in_size);
  for (int c=0; c<channels_; c++){
	fft_pad_map_in(map_in); // 0-padding: map_in --> fft_map_in_real;
 	// do FFT: map_in_real --> map_in_complex
	MKL_LONG status = DftiComputeForward(fft_handle_,fft_map_in_real_, fft_map_in_complex_);
	// fft_map_out_complex[c] += fft_map_in_complex * fft_weight_complex[n,c]
	fft_mult_acc(c);
    map_in += map_in_size;
  }
  //---- ifft map_out -------------------------------------------
  MKL_Complex8* map_out_complex = fft_map_out_complex_ ;
  float* map_out_real = fft_map_out_real_;
  Dtype* map_out = (Dtype* )(top_data + n* (num_output_ * height_out_ * width_out_));
  MKL_LONG status;
  for (int out=0; out < num_output_; out++){
	status = DftiComputeBackward(ifft_handle_, map_out_complex , map_out_real) ;
	fft_postprocess_map_out(map_out_real, map_out);
	if (0)
	{
	//  debug
//	fprintf(stderr," Map_out_by FFT[%d]: \n", out );
//	fft_print_2D(map_out_real,fft_height_, fft_width_ );
	fprintf(stderr," Map out_[%d]: \n", out);
	fft_print_2D(map_out,height_out_, width_out_);
	 //				print_2D(map_in, HEIGHT_, WIDTH_ );
	 //				fprintf(stderr,"------ Padded Data----------------\n");
	 //				print_2D(bottom_pad_,(HEIGHT_ + KSIZE_- 1),(WIDTH_ + KSIZE_- 1));
	 //				DftiComputeBackward(ifft_handle_, bottom_fft_ , bottom_debug_) ;
	 //				fprintf(stderr,"------ iFFT(FFT(Padded Data))-------\n");
	 //				print_2D(bottom_debug_,(HEIGHT_ + KSIZE_- 1),(WIDTH_ + KSIZE_- 1));
	 //
	}
	map_out_complex += fft_map_complex_size_;
	map_out_real    += fft_map_real_size_;
	map_out         += map_out_size_;
  }
  if (bias_term_) {
    int top_offset_n= n* (num_output_ * height_out_* width_out_);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
          (Dtype)1., top_data + top_offset_n);
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu_fft(const vector<Blob<Dtype>*>& bottom,
              vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data= (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  fft_pad_weights();
  fft_doFFT_weights();
  for (int n = 0; n < num_; ++n) {
    Forward_cpu_fft_task(bottom_data, top_data, weight, n);
  }
  return Dtype(0.);
}

// ------------------------------------------------------------------------------------------
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

/*
template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
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
  return Dtype(0.);
}
*/
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
//--------JUNK
/*
{
//  debug
//	fprintf(stderr," Map_out_by FFT[%d]: \n", out );
//	fft_print_2D(map_out_real,fft_height_, fft_width_ );
fprintf(stderr," Map out_[%d]: \n", out);
fft_print_2D(map_out,height_out_, width_out_);
 //				print_2D(map_in, HEIGHT_, WIDTH_ );
 //				fprintf(stderr,"------ Padded Data----------------\n");
 //				print_2D(bottom_pad_,(HEIGHT_ + KSIZE_- 1),(WIDTH_ + KSIZE_- 1));
 //				DftiComputeBackward(ifft_handle_, bottom_fft_ , bottom_debug_) ;
 //				fprintf(stderr,"------ iFFT(FFT(Padded Data))-------\n");
 //				print_2D(bottom_debug_,(HEIGHT_ + KSIZE_- 1),(WIDTH_ + KSIZE_- 1));
 //
}
*/


