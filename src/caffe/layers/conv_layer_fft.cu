#ifdef USE_FFT

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_setup() {
  if (fft_gpu_initialized_)
    fft_gpu_clean();
  fft_gpu_initialized_ = false;
  // Evaluate memory needed for buffers
  int num_weights = num_output_ * (channels_ / group_);
  size_t free_byte, total_byte;
  int layerMemoryBytes =
       num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype>) +
       fft_map_real_size_ * sizeof(Dtype) +
       fft_map_complex_size_  * sizeof(std::complex<Dtype>) +
       std::max(num_output_, channels_) * fft_map_complex_size_ *
       sizeof(std::complex<Dtype>) +
       std::max(num_output_, channels_) * fft_map_real_size_ * sizeof(Dtype);
  LOG(INFO) << "FFT buffers - memory needed = "
            << (layerMemoryBytes/(1024*1024)) << " MB";
  cudaMemGetInfo(&free_byte, &total_byte);
  LOG(INFO) << "CUDA free memory before buffers = "
            << free_byte/(1024*1024) <<" MB, out of "
            << total_byte/(1024*1024) << " MB";

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_weights_complex_),
         num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype>)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_in_real_),
         fft_map_real_size_ * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_in_complex_),
         fft_map_complex_size_  * sizeof(std::complex<Dtype>)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_out_complex_),
         std::max(num_output_, channels_) * fft_map_complex_size_ *
         sizeof(std::complex<Dtype>)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fft_gpu_map_out_real_),
       std::max(num_output_, channels_) * fft_map_real_size_ * sizeof(Dtype)));

     cudaMemGetInfo(&free_byte, &total_byte);
     LOG(INFO) << "CUDA free memory after buffers before plans = "
               << free_byte/(1024*1024) << " MB";

  int n[2] = {fft_height_, fft_width_};
  int inembed[] = {fft_height_, 2 * (fft_width_/2 + 1)};
  int in_size = fft_height_ * 2 * (fft_width_/2 + 1);
  int onembed[] = {fft_height_, (fft_width_/2 + 1)};

  // cufft plans
  if (sizeof(Dtype) == sizeof(float)) {
    CUFFT_CHECK(cufftPlan2d(&fft_gpu_handle_, fft_height_, fft_width_,
                 CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&ifft_gpu_handle_, fft_height_, fft_width_,
                 CUFFT_C2R));
    CUFFT_CHECK(cufftCreate(&fft_gpu_many_weights_handle_));
    CUFFT_CHECK(cufftPlanMany(&fft_gpu_many_weights_handle_, 2, n, inembed,
                 1, in_size, onembed, 1, fft_map_complex_size_, CUFFT_R2C,
                 num_weights));
  } else if (sizeof(Dtype) == sizeof(double)) {
    CUFFT_CHECK(cufftPlan2d(&fft_gpu_handle_, fft_height_, fft_width_,
                 CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan2d(&ifft_gpu_handle_, fft_height_, fft_width_,
                 CUFFT_Z2D));
    CUFFT_CHECK(cufftCreate(&fft_gpu_many_weights_handle_));
    CUFFT_CHECK(cufftPlanMany(&fft_gpu_many_weights_handle_, 2, n, inembed,
                 1, in_size, onembed, 1, fft_map_complex_size_, CUFFT_D2Z,
                 num_weights));
  }
  cudaMemGetInfo(&free_byte, &total_byte);
  LOG(INFO) << "CUDA free memory after buffers, after plans is: "
            << free_byte/(1024*1024) << " MB";
  fft_gpu_initialized_ = true;
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_clean() {
  if (fft_gpu_initialized_) {
    cudaFree(fft_gpu_weights_complex_);
    cudaFree(fft_gpu_map_in_real_);
    cudaFree(fft_gpu_map_in_complex_);
    cudaFree(fft_gpu_map_out_complex_);
    cudaFree(fft_gpu_map_out_real_);
    cufftDestroy(fft_gpu_handle_);
    cufftDestroy(ifft_gpu_handle_);
    cufftDestroy(fft_gpu_many_weights_handle_);
  }
  fft_gpu_initialized_ = false;
}

//=====================Forward GPU========================

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_compute_weights() {
  const Dtype *weight = this->blobs_[0]->gpu_data();
  // 0-padding of weights before FFT ----------------------
  caffe_gpu_memset((num_output_ * (channels_ / group_) * fft_map_complex_size_
                 * sizeof(std::complex<Dtype>)), 0., fft_gpu_weights_complex_);

  // Copy weights 2 buffer----------------------------------
  fft_gpu_copy2buffer(reinterpret_cast<Dtype*>(fft_gpu_weights_complex_),
          weight, num_output_, group_, channels_, kernel_h_, kernel_w_,
          fft_height_, fft_width_);

  //  FFT of weights in place ------------------------------
  caffe_gpu_fft_execute_dft_r2c_inplace(fft_gpu_many_weights_handle_,
       fft_gpu_weights_complex_);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft_task(
      const Dtype* bottom_data, Dtype* top_data, const Dtype* weight, int n) {
  int map_in_size = height_* width_;
  Dtype* map_in_n = const_cast<Dtype*>
    (bottom_data + n * channels_ * map_in_size);
  Dtype* map_out_n = top_data + n* (num_output_ * height_out_ * width_out_);

  // clear buffers
  caffe_gpu_memset((num_output_*fft_map_complex_size_*
               sizeof(std::complex<Dtype>)), 0., fft_gpu_map_out_complex_);

  // loop over all channels ---------------------
  Dtype* map_in;
  std::complex<Dtype>* weights_complex;
  std::complex<Dtype>* map_out_complex;
  Dtype* map_out_real;
  Dtype* map_out;
  for (int c = 0; c < channels_; c++) {
    map_in = map_in_n + c * map_in_size;
    //  0-padding: map_in --> fft_map_in_real -------------
    caffe_gpu_memset((fft_map_real_size_ * sizeof(Dtype)), 0.,
                      fft_gpu_map_in_real_);
    fft_gpu_copy2buffer2D_in(reinterpret_cast<Dtype*>(fft_gpu_map_in_real_) ,
             map_in, fft_width_ , height_ , width_ , 1, 1, pad_h_, pad_w_ ,
             (Dtype)1.0);
    //  FFT: map_in_real --> map_in_complex
    caffe_gpu_fft_execute_dft_r2c(fft_gpu_handle_, fft_gpu_map_in_real_,
        fft_gpu_map_in_complex_);
    int g = c / (channels_ / group_);        // group
    int c_offset= c % (channels_ / group_);  // index inside group
    int out_first = g* (num_output_ / group_);
    int out_last = (g+1)*(num_output_ / group_);

    // OPTION: one large kernel with 2D grid
    for (int out = out_first; out < out_last; out++) {
      map_out_complex = fft_gpu_map_out_complex_ + out * fft_map_complex_size_;
      weights_complex = fft_gpu_weights_complex_ +
         (out * (channels_/group_) + c_offset) * fft_map_complex_size_;
      caffe_gpu_elementMulConj((std::complex<Dtype>*)map_out_complex,
         (std::complex<Dtype>*)fft_gpu_map_in_complex_,
         (std::complex<Dtype>*)weights_complex, fft_map_complex_size_);
      }
  }
  //  IFFT: map_out_complex --> map_out_real
  Dtype ifft_scale = 1./((Dtype)fft_map_real_size_);
  //  OPTION: ifft_many ?
  for (int out = 0; out < num_output_; out++) {
    map_out_complex = fft_gpu_map_out_complex_  + out * fft_map_complex_size_;
    map_out_real = fft_gpu_map_out_real_;
    caffe_gpu_fft_execute_dft_c2r(ifft_gpu_handle_,
        map_out_complex, map_out_real);
    //  post-process: map_out_real --> map_out
    map_out = map_out_n + out * map_out_size_;
    fft_gpu_copy2buffer2D_out(map_out , map_out_real, height_out_, width_out_,
         fft_height_, fft_width_, stride_h_, stride_w_ , 0, 0, ifft_scale);
  }
  //  bias
  int top_offset_n = n* (num_output_ * height_out_* width_out_);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
        N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
        bias_multiplier_.gpu_data(), (Dtype)1., top_data + top_offset_n);
  }
}

template <typename Dtype>
Dtype ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
         const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (!fft_gpu_initialized_)
    fft_setup();
  fft_gpu_compute_weights();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int n = 0; n < num_; ++n) {
      Forward_gpu_fft_task(bottom_data, top_data, weight, n);
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_task(
           const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top, int i, int n) {
  const Dtype* bottom_data = bottom[i]->gpu_data();
  Dtype* top_data = top[i]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  Dtype* col_buff = NULL;
  if (!is_1x1_) {
    col_buff = col_buffer_.mutable_gpu_data();
  }
  // im2col transformation: unroll input regions for filtering
  // into column matrix for multiplication.
  if (!is_1x1_) {
    im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
        width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
        col_buff);
  } else {
    col_buff = bottom[i]->mutable_gpu_data() + bottom[i]->offset(n);
  }
  // Take inner products for groups.
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
      (Dtype)1., weight + weight_offset * g, col_buff + col_offset * g,
      (Dtype)0., top_data + top[i]->offset(n) + top_offset * g);
  }
  // Add bias
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
        N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
        bias_multiplier_.gpu_data(),
        (Dtype)1., top_data + top[i]->offset(n));
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu(
       const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (fft_on_) {
    Forward_gpu_fft(bottom, top);
  } else {
    for (int i = 0; i < bottom.size(); ++i) {
      for (int n = 0; n < num_; ++n) {
        Forward_gpu_task(bottom, top, i, n);
      }
    }
  }
}

//===========================BACKWARD GPU============================

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_fft_task(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
         const Dtype* weight, int i, int n) {
  const Dtype* top_diff = top[i]->gpu_diff();
  Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

  //  back propagation is from top to bottom: the top is the inputs.
  int map_in_size = height_out_* width_out_;
  Dtype* map_in_n = const_cast<Dtype*>(top_diff + top[i]->offset(n));
  Dtype* map_out_n = const_cast<Dtype*>(bottom_diff + bottom[i]->offset(n));
  // clear buffers
  caffe_gpu_memset((channels_ * fft_map_complex_size_*
                   sizeof(std::complex<Dtype>)), 0., fft_gpu_map_out_complex_);

  // loop over all channels ---------------------
  Dtype* map_in;
  std::complex<Dtype>* weights_complex;
  std::complex<Dtype>* map_out_complex;
  Dtype* map_out_real;
  Dtype* map_out;
  caffe_gpu_memset((fft_map_real_size_*sizeof(Dtype)), 0.,
                   fft_gpu_map_in_real_);
  for (int out = 0; out < num_output_; out++) {
    map_in = map_in_n + out * map_in_size;
    //  0-padding: map_out --> fft_map_in_real -------------
    fft_gpu_copy2buffer2D_in(reinterpret_cast<Dtype*>(fft_gpu_map_in_real_),
               map_in, fft_width_, height_out_, width_out_,
                stride_h_, stride_w_, 0, 0, (Dtype)1.0);

    //  FFT: map_in_real --> map_in_complex
    caffe_gpu_fft_execute_dft_r2c(fft_gpu_handle_, fft_gpu_map_in_real_,
        fft_gpu_map_in_complex_);

    int g = out / (num_output_ / group_);            // group
    int c_first = g *  (channels_ / group_);
    int c_last  = (g+1) * (channels_ / group_);
    for (int c = c_first; c < c_last; c++) {
      int c_offset = c % (channels_ / group_);
      map_out_complex = fft_gpu_map_out_complex_ + c * fft_map_complex_size_;
      weights_complex = fft_gpu_weights_complex_ +
               (out * (channels_/group_) + c_offset) * fft_map_complex_size_;
      caffe_gpu_elementMul((std::complex<Dtype>*)map_out_complex,
          (std::complex<Dtype>*)fft_gpu_map_in_complex_,
          (std::complex<Dtype>*)weights_complex,
          fft_map_complex_size_);
    }
  }

  //  IFFT: map_out_complex --> map_out_real
  //  Option: fft_execute (many) ?
  Dtype ifft_scale = 1./((Dtype) fft_map_real_size_);
  for (int c = 0; c < channels_; c++) {
    map_out = map_out_n + c * map_size_;
    map_out_complex = fft_gpu_map_out_complex_  + c * fft_map_complex_size_;
    map_out_real = fft_gpu_map_out_real_;
    caffe_gpu_fft_execute_dft_c2r(ifft_gpu_handle_, map_out_complex,
              map_out_real);
    fft_gpu_copy2buffer2D_out(map_out , map_out_real, height_, width_,
              fft_height_, fft_width_, 1, 1, pad_h_, pad_w_, ifft_scale);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_bottom_diff_task(
      const Dtype* top_diff, Dtype* bottom_diff,
       const Dtype* weight, int i,  int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  Dtype* col_buff = NULL;
  int bottom_offset = channels_ * height_ * width_;
  int top_offset_n = n * (num_output_ * height_out_ * width_out_);

  // gradient w.r.t. bottom data, if necessary
    if (!is_1x1_) {
      col_buff = col_buffer_.mutable_gpu_data();
    } else {
      col_buff = bottom_diff + n*bottom_offset;
    }
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
          (Dtype)1., weight + weight_offset * g,
          top_diff + top_offset_n + top_offset * g,
          (Dtype)0., col_buff + col_offset * g);
    }
    // col2im back to the data
    if (!is_1x1_) {
      col2im_gpu(col_buff, channels_, height_, width_,
          kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
          bottom_diff + bottom_offset*n);
    }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_weight_diff_task(
         const Dtype* top_diff, const vector<Blob<Dtype>*>& bottom,
          int i, int n) {
  int weight_offset = M_ * K_;
  int col_offset = K_ * N_;
  int top_offset = M_ * N_;
  int top_offset_n = n* (num_output_ * height_out_* width_out_);
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* col_buff = NULL;
  if (!is_1x1_) {
    col_buff = col_buffer_.mutable_gpu_data();
  }
  const Dtype* bottom_data = bottom[i]->gpu_data();
  // recompute im2col
  if (!is_1x1_) {
    im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
              width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
              stride_h_, stride_w_, col_buff);
  } else {
    col_buff = bottom[i]->mutable_gpu_data() + bottom[i]->offset(n);
  }
  // gradient w.r.t. weight. Note that we will accumulate diffs.
  if (this->param_propagate_down_[0]) {
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
          (Dtype)1., top_diff + top_offset_n + top_offset * g,
          col_buff + col_offset * g, (Dtype)1.,
          weight_diff+ weight_offset * g);
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu(
        const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.gpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        Backward_gpu_weight_diff_task(top_diff, bottom, i, n);
        // gradient w.r.t. bottom data, if necessary
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          if (fft_on_ && (!fft_gpu_initialized_))
            fft_setup();
        // WARNING: Assume fft for weights was computed in Forward
        // This assumption can fail in runtest, if only Backward() is called!
        // fft_compute_weights();
          if ((fft_on_) && (!is_1x1_)) {
            Backward_gpu_fft_task(bottom, top, weight, i, n);
          } else {
            Backward_gpu_bottom_diff_task(
             top_diff, bottom_diff, weight, i, n);
          }
        }
      }
    }
  }
}

// float instantiation
  template
  void ConvolutionLayerFFT<float>::fft_gpu_setup();
  template
  void ConvolutionLayerFFT<float>::fft_gpu_clean();
  template
  float ConvolutionLayerFFT<float>::Forward_gpu_fft(
        const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
  template
  void ConvolutionLayerFFT<float>::Forward_gpu_fft_task(
        const float *bottom_data, float* top_data, const float* weight, int n);
  template
  void ConvolutionLayerFFT<float>::Forward_gpu_task(
       const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
       int i, int n);
  template
  void ConvolutionLayerFFT<float>::fft_gpu_compute_weights();
  template
  void ConvolutionLayerFFT<float>::Forward_gpu(
       const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu_fft_task(
        const vector<Blob<float>*>& bottom,
        const vector<Blob<float>*>& top,
        const float* weight, int i, int n);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu_weight_diff_task(
        const float* top_diff, const vector<Blob<float>*>& bottom,
        int i, int n);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu_bottom_diff_task(
        const float* top_diff, float* bottom_diff,
        const float* weight, int i,  int n);
  template
  void ConvolutionLayerFFT<float>::Backward_gpu(
        const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<float>*>& bottom);

// double instantiation
  template
  void ConvolutionLayerFFT<double>::fft_gpu_setup();
  template
  void ConvolutionLayerFFT<double>::fft_gpu_clean();
  template
  double ConvolutionLayerFFT<double>::Forward_gpu_fft(
        const vector<Blob<double>*>& bottom,
        const vector<Blob<double>*>& top);
  template
  void ConvolutionLayerFFT<double>::Forward_gpu_fft_task(
        const double *bottom_data, double* top_data,
          const double* weight, int n);
  template
  void ConvolutionLayerFFT<double>::Forward_gpu_task(
         const vector<Blob<double>*>& bottom,
         const vector<Blob<double>*>& top, int i, int n);
  template
  void ConvolutionLayerFFT<double>::fft_gpu_compute_weights();
  template
  void ConvolutionLayerFFT<double>::Forward_gpu(
        const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu_fft_task(
        const vector<Blob<double>*>& bottom,
        const vector<Blob<double>*>& top,
        const double* weight, int i, int n);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu_weight_diff_task(
        const double* top_diff, const vector<Blob<double>*>& bottom,
        int i, int n);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu_bottom_diff_task(
        const double* top_diff, double* bottom_diff,
        const double* weight, int i,  int n);
  template
  void ConvolutionLayerFFT<double>::Backward_gpu(
        const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<double>*>& bottom);

}  // namespace caffe
#endif // USE_FFT
