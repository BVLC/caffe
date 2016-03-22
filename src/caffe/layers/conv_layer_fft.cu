#ifndef CPU_ONLY
#include <algorithm>
#include <vector>
#include "caffe/common.hpp"
#if defined(USE_GREENTEA) && defined(USE_FFT)
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/fft.hpp"

#include "caffe/layers/conv_fft_layer.hpp"

#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"


// #define COMPLEX_MULT_CONJ_1D
// #define COMPLEX_NULT_CONJ_RESHAPE
#define COMPLEX_MULT_CONJ_2D         // Best speed for CaffeNet conv1,2,3
// #define COMPLEX_MULT_CONJ_2D_SLM
// #define COMPLEX_MULT_CONJ_3D      // Accuracy issue
// #define COMPLEX_MULT_CONJ_3D_SLM  // Accuracy issue

// #define FFT_BACKWARD
#ifdef FFT_BACKWARD
#define COMPLEX_MULT_1D         // Fast for small size data of unit test
// #define COMPLEX_MULT_2D_SLM  // Segmentation fault on TestGradientGroup
// #define COMPLEX_MULT_3D      // Accuracy issue
#endif

namespace caffe {

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_setup() {
  if (fft_gpu_initialized_) {
    return;
  }

  viennacl::ocl::context &ctx = viennacl::ocl::current_context();

  // Evaluate memory needed for buffers
  int num_weights = this->num_output_ * (this->channels_ / this->group_);
  int tmpMax = std::max(this->num_output_, this->channels_);
  size_t fft_gpu_map_in_real_bytes = fft_map_real_size_ * sizeof(Dtype);
  size_t fft_gpu_map_in_complex_bytes = fft_map_complex_size_ *
                                        sizeof(DtypeComplex<Dtype>);
  size_t fft_gpu_map_out_complex_bytes = tmpMax * fft_gpu_map_in_complex_bytes;
  size_t fft_gpu_map_out_real_bytes = tmpMax * fft_gpu_map_in_real_bytes;
  size_t fft_gpu_weights_complex_bytes =
      num_weights * fft_gpu_map_in_complex_bytes;

  int layerMemoryBytes =
      fft_gpu_weights_complex_bytes +
      fft_gpu_map_in_real_bytes * this->channels_ +
      fft_gpu_map_in_real_bytes * this->num_output_ +
      fft_gpu_map_in_complex_bytes * this->channels_ +
      fft_gpu_map_in_complex_bytes * this->num_output_ +
      fft_gpu_map_out_complex_bytes +
      fft_gpu_map_out_real_bytes;
  LOG(INFO) << "FFT buffers - memory needed = "
            << ((Dtype)layerMemoryBytes / (1024.f * 1024.f)) << " MB";

  cl_int cl_err;
  fft_gpu_weights_complex_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_weights_complex_bytes, NULL, &cl_err);
#ifdef COMPLEX_NULT_CONJ_RESHAPE
  fft_gpu_weights_complex_reshape_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_weights_complex_bytes, NULL, &cl_err);
#endif
  fft_gpu_map_in_real_all_channels_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_map_in_real_bytes * this->channels_,
      NULL, &cl_err);
  fft_gpu_map_in_complex_all_channels_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_map_in_complex_bytes * this->channels_,
      NULL, &cl_err);

  fft_gpu_map_in_real_all_num_output_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_map_in_real_bytes * this->num_output_, NULL,
      &cl_err);
  fft_gpu_map_in_complex_all_num_output_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_map_in_complex_bytes * this->num_output_,
      NULL, &cl_err);

  fft_gpu_map_out_complex_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_map_out_complex_bytes, NULL, &cl_err);
  fft_gpu_map_out_real_ = clCreateBuffer(ctx.handle().get(),
      CL_MEM_READ_WRITE, fft_gpu_map_out_real_bytes, NULL, &cl_err);

  ClFFTState& fft_state = Caffe::cl_fft_state();
  // FFT plan for weights
  fft_gpu_many_weights_handle_ = fft_state.getForwardInPlaceFFTManyPlanHandle(
      fft_height_, fft_width_, num_weights);
  // FFT plan
  fft_gpu_forward_many_handle_ =
      fft_state.getForwardOutOfPlaceFFTManyPlanHandle(fft_height_, fft_width_,
      this->channels_);
  // Inverse FFT plan
  ifft_gpu_forward_many_handle_ =
      fft_state.getForwardOutOfPlaceIFFTManyPlanHandle(fft_height_, fft_width_,
      this->num_output_);
#ifdef FFT_BACKWARD
  // FFT plan
  fft_gpu_backward_many_handle_ =
      fft_state.getBackwardOutOfPlaceFFTManyPlanHandle(fft_height_, fft_width_,
      this->num_output_);
  // Inverse FFT plan
  ifft_gpu_backward_many_handle_ =
      fft_state.getBackwardOutOfPlaceIFFTManyPlanHandle(fft_height_, fft_width_,
      this->channels_);
#endif
  fft_gpu_initialized_ = true;
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_clean() {
  if (fft_gpu_initialized_) {
    clReleaseMemObject((cl_mem)fft_gpu_weights_complex_);
#ifdef COMPLEX_NULT_CONJ_RESHAPE
    clReleaseMemObject(fft_gpu_weights_complex_reshape_);
#endif
    clReleaseMemObject((cl_mem)fft_gpu_map_in_real_all_channels_);
    clReleaseMemObject((cl_mem)fft_gpu_map_in_complex_all_channels_);
    clReleaseMemObject((cl_mem)fft_gpu_map_in_real_all_num_output_);
    clReleaseMemObject((cl_mem)fft_gpu_map_in_complex_all_num_output_);
    clReleaseMemObject((cl_mem)fft_gpu_map_out_complex_);
    clReleaseMemObject((cl_mem)fft_gpu_map_out_real_);
  }
  fft_gpu_initialized_ = false;
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_compute_weights() {
  int num_weights = this->num_output_ * (this->channels_ / this->group_);
  int size = num_weights * fft_map_complex_size_ * sizeof(DtypeComplex<Dtype>);
  // Clear buffer
  clear_gpu_fft_buffer(fft_gpu_weights_complex_, size);

  // Cyclic-shift 0-padding of weights
  const Dtype* weight = this->blobs_[0]->gpu_data();
  fft_gpu_copy2buffer(reinterpret_cast<Dtype*>(fft_gpu_weights_complex_),
      weight, this->num_output_, this->group_, this->channels_,
      this->kernel_h_, this->kernel_w_, kernel_center_h_,
      kernel_center_w_, fft_height_, fft_width_);

  // Batched in-place FFT of weights
  caffe_gpu_fft_execute_r2c_inplace(fft_gpu_many_weights_handle_,
      reinterpret_cast<Dtype*>(fft_gpu_weights_complex_));

  // Reshape
#ifdef COMPLEX_NULT_CONJ_RESHAPE
  reshape_weights(reinterpret_cast< DtypeComplex<Dtype>* >(
      fft_gpu_weights_complex_reshape_),
      reinterpret_cast< DtypeComplex<Dtype>* >(fft_gpu_weights_complex_),
      fft_map_complex_size_, this->num_output_, (this->channels_/this->group_));
#endif
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft_task(const Dtype* bottom_data,
         int bottom_data_offset, Dtype* top_data, int top_data_offset, int n,
         int ch_gr, int out_gr) {
  // Clear buffer
  clear_gpu_fft_buffer(fft_gpu_map_out_complex_,
      this->num_output_ * fft_map_complex_size_ * sizeof(DtypeComplex<Dtype>));
  clear_gpu_fft_buffer(fft_gpu_map_in_real_all_channels_,
      this->channels_ * fft_map_real_size_ * sizeof(Dtype));

  // Left-top 0-padding of bottom data
  fft_gpu_copy2buffer_in_2D(
      reinterpret_cast<Dtype*>(fft_gpu_map_in_real_all_channels_),
      bottom_data, bottom_data_offset,
      this->channels_, fft_height_, fft_width_, this->height_, this->width_,
      1, 1, this->pad_h_, this->pad_w_);

  // Batched FFT for all channels of padded bottom data
  caffe_gpu_fft_execute_r2c(fft_gpu_forward_many_handle_,
      reinterpret_cast<Dtype*>(fft_gpu_map_in_real_all_channels_),
      reinterpret_cast<DtypeComplex<Dtype>*>(
          fft_gpu_map_in_complex_all_channels_));

  // Multiplication of FFT bottom data and FFT weights
#ifdef COMPLEX_MULT_CONJ_1D
  for (int c = 0; c < this->channels_; c+=ch_gr) {
    int g = c / ch_gr;
    int out_first = g * out_gr;
    int out_last = out_first + out_gr;
    for (int out = out_first; out < out_last; ++out) {
      caffe_gpu_elementMulConj_1D(
          reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_) +
              out * fft_map_complex_size_,
          reinterpret_cast<DtypeComplex<Dtype>*>(
              fft_gpu_map_in_complex_all_channels_) +
              c * fft_map_complex_size_,
          reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_) +
              (out * ch_gr) * fft_map_complex_size_,
          fft_map_complex_size_, ch_gr);
    }
  }
#elif defined(COMPLEX_NULT_CONJ_RESHAPE)
  for (int c = 0; c < this->channels_; c+=ch_gr) {
    int g = c / ch_gr;
    int out_first = g * out_gr;
    caffe_gpu_elementMulConj_Reshape(
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_) +
            out_first * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(
            fft_gpu_map_in_complex_all_channels_) +
            c * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(
            fft_gpu_weights_complex_reshape_) +
            (out_first * ch_gr) * fft_map_complex_size_,
        out_gr, fft_map_complex_size_, ch_gr);
  }
#elif defined(COMPLEX_MULT_CONJ_2D)
  for (int c = 0; c < this->channels_; c+=ch_gr) {
    int g = c / ch_gr;
    int out_first = g * out_gr;
    caffe_gpu_elementMulConj_2D(
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_),
            out_first * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(
            fft_gpu_map_in_complex_all_channels_),
            c * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_),
            (out_first * ch_gr) * fft_map_complex_size_,
        out_gr, fft_map_complex_size_, ch_gr);
  }
#elif defined(COMPLEX_MULT_CONJ_2D_SLM)
  for (int c = 0; c < this->channels_; c+=ch_gr) {
    int g = c / ch_gr;
    int out_first = g * out_gr;
    caffe_gpu_elementMulConj_2D_SLM(
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_) +
            out_first * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(
            fft_gpu_map_in_complex_all_channels_) +
            c * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_) +
            (out_first * ch_gr) * fft_map_complex_size_,
        out_gr, fft_map_complex_size_, ch_gr);
  }
#elif defined(COMPLEX_MULT_CONJ_3D)
  for (int c = 0; c < this->channels_; c+=ch_gr) {
    int g = c / ch_gr;
    int out_first = g * out_gr;
    caffe_gpu_elementMulConj_3D(
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_) +
            out_first * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(
            fft_gpu_map_in_complex_all_channels_) +
            c * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_) +
            (out_first * ch_gr) * fft_map_complex_size_,
        out_gr, fft_map_complex_size_, ch_gr);
  }
#elif defined(COMPLEX_MULT_CONJ_3D_SLM)
  for (int c = 0; c < this->channels_; c+=ch_gr) {
    int g = c / ch_gr;
    int out_first = g * out_gr;
    caffe_gpu_elementMulConj_3D_SLM(
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_) +
            out_first * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(
            fft_gpu_map_in_complex_all_channels_) +
            c * fft_map_complex_size_,
        reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_) +
            (out_first * ch_gr) * fft_map_complex_size_,
        out_gr, fft_map_complex_size_, ch_gr);
  }
#endif

  // Batched IFFT for num output of result
  caffe_gpu_fft_execute_c2r(ifft_gpu_forward_many_handle_,
      reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_),
      reinterpret_cast<Dtype*>(fft_gpu_map_out_real_));

  // Mapping from IFFT result to top data
  fft_gpu_copy2buffer_out_forward_2D(
      top_data, top_data_offset,
      reinterpret_cast<Dtype*>(fft_gpu_map_out_real_),
      this->num_output_,
      this->height_out_, this->width_out_, fft_height_, fft_width_,
      kernel_center_h_, kernel_center_w_,
      this->stride_h_, this->stride_w_ , 0, 0);

  if (this->bias_term_) {
    const Dtype* bias = this->blobs_[1]->gpu_data();
    this->forward_gpu_bias(top_data, top_data_offset, bias);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
         const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top) {
  fft_gpu_compute_weights();

  int ch_gr = this->channels_ / this->group_;
  int out_gr = this->num_output_ / this->group_;

  // Calculate tile count based on fft complex data size
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      Forward_gpu_fft_task(bottom_data, n * this->bottom_dim_, top_data,
                           n * this->top_dim_, n, ch_gr, out_gr);
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    Forward_gpu_fft(bottom, top);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_fft_task(
         const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top,
         const Dtype* weight, int i, int n,
         int ch_gr, int out_gr) {
  const Dtype* top_diff = top[i]->gpu_diff();
  Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

  // Clear buffers
  clear_gpu_fft_buffer(fft_gpu_map_in_real_all_num_output_,
      fft_map_real_size_ * this->num_output_ * sizeof(Dtype));
  clear_gpu_fft_buffer(fft_gpu_map_out_complex_,
      this->channels_ * fft_map_complex_size_ * sizeof(DtypeComplex<Dtype>));

  // Left-top 0-padding of top data
  fft_gpu_copy2buffer_in_2D(
      reinterpret_cast<Dtype*>(fft_gpu_map_in_real_all_num_output_),
      top_diff, n * this->top_dim_,
      this->num_output_, fft_height_, fft_width_, this->height_out_,
      this->width_out_, this->stride_h_, this->stride_w_, 0, 0);

  // Batched FFT for all num output of padded top data
  caffe_gpu_fft_execute_r2c(fft_gpu_backward_many_handle_,
      reinterpret_cast<Dtype*>(fft_gpu_map_in_real_all_num_output_),
      reinterpret_cast<DtypeComplex<Dtype>*>(
          fft_gpu_map_in_complex_all_num_output_));

  // Multiplication of FFT top data and FFT weights
#ifdef COMPLEX_MULT_1D
  for (int out = 0; out < this->num_output_; out++) {
    int g = out / out_gr;
    int c_first = g * ch_gr;
    int c_last = (g + 1) * ch_gr;
    for (int c = c_first; c < c_last; c+=ch_gr) {
      caffe_gpu_elementMul_1D(
          reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_) +
              c * fft_map_complex_size_,
          reinterpret_cast<DtypeComplex<Dtype>*>(
              fft_gpu_map_in_complex_all_num_output_) +
                  out * fft_map_complex_size_,
          reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_) +
              (out * ch_gr) * fft_map_complex_size_,
          fft_map_complex_size_, ch_gr);
    }
  }
#elif defined(COMPLEX_MULT_2D_SLM)
  caffe_gpu_elementMul_2D_SLM(
      reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_),
      reinterpret_cast<DtypeComplex<Dtype>*>(
          fft_gpu_map_in_complex_all_num_output_),
      reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_),
      fft_map_complex_size_, ch_gr, this->num_output_);
#elif defined(COMPLEX_MULT_3D)  // TEST in: WIP: Unit test accuracy issue
  caffe_gpu_elementMul_3D(
      reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_),
      reinterpret_cast<DtypeComplex<Dtype>*>(
          fft_gpu_map_in_complex_all_num_output_),
      reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_weights_complex_),
      fft_map_complex_size_, ch_gr, out_gr, this->num_output_);
#endif

  // Batched IFFT for all channels of result
  caffe_gpu_fft_execute_c2r(ifft_gpu_backward_many_handle_,
      reinterpret_cast<DtypeComplex<Dtype>*>(fft_gpu_map_out_complex_),
      reinterpret_cast<Dtype*>(fft_gpu_map_out_real_));

  // Mapping from IFFT result to bottom diff
// TEST out
/*
  for (int c = 0; c < this->channels_; c++) {
    fft_gpu_copy2buffer_out_backward(
        bottom_diff + n * this->bottom_dim_ + c * map_size_,
        reinterpret_cast<Dtype*>(fft_gpu_map_out_real_) +
            c * fft_map_real_size_,
        this->height_, this->width_, fft_height_, fft_width_,
        kernel_center_h_, kernel_center_w_, 1, 1, this->pad_h_, this->pad_w_);
  }
*/
  fft_gpu_copy2buffer_out_backward_2D(
      bottom_diff, n * this->bottom_dim_,
      reinterpret_cast<Dtype*>(fft_gpu_map_out_real_),
      this->channels_,
      this->height_, this->width_, fft_height_, fft_width_,
      kernel_center_h_, kernel_center_w_, 1, 1, this->pad_h_, this->pad_w_);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu(
         const vector<Blob<Dtype>*>& top,
         const vector<bool>& propagate_down,
         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  if (this->param_propagate_down_[0]) {
    greentea_gpu_set(this->device_->id(), this->blobs_[0]->count(), Dtype(0),
                     (cl_mem)weight_diff, Dtype(0));
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    greentea_gpu_set(this->device_->id(), this->blobs_[1]->count(), Dtype(0),
        (cl_mem)this->blobs_[1]->mutable_gpu_diff(), Dtype(0));
  }


  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
#ifdef FFT_BACKWARD
      int ch_gr = this->channels_ / this->group_;
      int out_gr = this->num_output_ / this->group_;
      if (this->param_propagate_down_[0]) {
        for (int n = 0; n < this->num_; ++n) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
      }
      if (propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          Backward_gpu_fft_task(bottom, top, weight, i, n, ch_gr, out_gr);
        }
      }
#else  // Default GEMM approach
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, n * this->bottom_dim_,
              top_diff, n * this->top_dim_, weight_diff);
        }
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff, n * this->top_dim_,
              weight, bottom_diff, n * this->bottom_dim_);
        }
      }
#endif
    }
  }
}

// float instantiation
template void ConvolutionLayerFFT<float>::fft_gpu_setup();
template void ConvolutionLayerFFT<float>::fft_gpu_clean();
template void ConvolutionLayerFFT<float>::Forward_gpu_fft(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void ConvolutionLayerFFT<float>::Forward_gpu_fft_task(
    const float *bottom_data, int bottom_data_offset, float* top_data,
    int top_data_offset, int n, int ch_gr, int out_gr);
template void ConvolutionLayerFFT<float>::fft_gpu_compute_weights();
template void ConvolutionLayerFFT<float>::Backward_gpu_fft_task(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    const float* weight, int i, int n, int ch_gr, int out_gr);

// double instantiation
template<>
void ConvolutionLayerFFT<double>::fft_gpu_setup() {
  NOT_IMPLEMENTED;
}

template<>
void ConvolutionLayerFFT<double>::fft_gpu_clean() {
  NOT_IMPLEMENTED;
}

template<>
void ConvolutionLayerFFT<double>::Forward_gpu_fft(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}
template<>
void ConvolutionLayerFFT<double>::Forward_gpu_fft_task(
    const double *bottom_data, int bottom_data_offset, double* top_data,
    int top_data_offset, int n, int ch_gr, int out_gr) {
  NOT_IMPLEMENTED;
}
template<>
void ConvolutionLayerFFT<double>::fft_gpu_compute_weights() {
  NOT_IMPLEMENTED;
}
template<> void ConvolutionLayerFFT<double>::Backward_gpu_fft_task(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    const double* weight, int i, int n, int ch_gr, int out_gr) {
  NOT_IMPLEMENTED;
}
template <>
void ConvolutionLayerFFT<double>::Forward_gpu(
         const vector<Blob<double>*>& bottom,
         const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}
template <>
void ConvolutionLayerFFT<double>::Backward_gpu(
         const vector<Blob<double>*>& top,
         const vector<bool>& propagate_down,
         const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayerFFT);

}  // namespace caffe
#endif  // USE_GREENTEA && USE_FFT
#endif  // !CPU_ONLY
