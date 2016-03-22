#ifdef USE_FFT

#include <algorithm>  // for max
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/conv_fft_layer.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_shape_data[i])
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT<Dtype>() {
  fft_clean();
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  fft_setup(bottom, top);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_setup(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  // TODO: Temporary speed-up trick
  /*if (this->group_ == 1) {
    if (this->num_output_ % 2 == 0 && this->channels_ % 2 == 0)
      this->group_ = 2;
    else if (this->num_output_ % 3 == 0 && this->channels_ % 3 == 0)
      this->group_ = 3;
  }*/
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  kernel_h_ = kernel_shape_data[0];
  kernel_w_ = kernel_shape_data[1];
  height_ = bottom[0]->shape(this->channel_axis_ + 1);
  width_ = bottom[0]->shape(this->channel_axis_ + 2);
  height_out_ = top[0]->shape(this->channel_axis_ + 1);
  width_out_ = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  pad_h_ = pad_data[0];
  pad_w_ = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  stride_h_ = stride_data[0];
  stride_w_ = stride_data[1];

  kernel_center_h_ = static_cast<int>(
      static_cast<float>(kernel_h_) / 2.f - 0.5f);
  kernel_center_w_ = static_cast<int>(
      static_cast<float>(kernel_w_) / 2.f - 0.5f);
  // Pad this size due to circular convolution of FFT
  fft_height_ = height_ +
      std::max(2 * pad_h_, (kernel_h_ - 1));
  fft_width_ = width_ +
      std::max(2 * pad_w_, (kernel_w_ - 1));
  // FFT size should be power of 2
  fft_height_ = next_mix_of_235(fft_height_);
  fft_width_  = next_mix_of_235(fft_width_);
  // Note: 16 equals to 64 byte (cache line) for float
  const int m = 16;
  if ((fft_height_ % m) > 0)
    fft_height_ = fft_height_ + (m - (fft_height_ % m));
  if ((fft_width_ % m) > 0)
    fft_width_ = fft_width_ + (m - (fft_width_ % m));
  fft_complex_width_ = fft_width_/2 + 1;

  map_size_ = height_ * width_;
  fft_map_real_size_ = fft_height_ * fft_width_;
  fft_map_complex_size_ = fft_height_ * fft_complex_width_;
  map_out_size_ = height_out_ * width_out_;

  switch (Caffe::mode()) {
    case Caffe::CPU:
      fft_cpu_setup();
      break;
    case Caffe::GPU:
#ifdef USE_GREENTEA
      fft_gpu_setup();
#endif
      break;
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_cpu_setup() {
  if (fft_cpu_initialized_) {
    return;
  }

  // Allocate buffers for fft
  int num_weights = this->num_output_ * (this->channels_ / this->group_);
  fft_weights_complex_ = (std::complex<Dtype> *) caffe_cpu_fft_malloc<Dtype>(
      num_weights * fft_map_complex_size_ * sizeof(std::complex<Dtype> ));
  fft_map_in_real_ = reinterpret_cast<Dtype *> (caffe_cpu_fft_malloc<Dtype>(
      fft_map_real_size_ * sizeof(Dtype)));
  fft_map_in_complex_ = (std::complex<Dtype> *) caffe_cpu_fft_malloc<Dtype>(
      fft_map_complex_size_ * sizeof(std::complex<Dtype>));
  fft_map_out_complex_ = (std::complex<Dtype>*) caffe_cpu_fft_malloc<Dtype>(
      std::max(this->num_output_, this->channels_) *
      fft_map_complex_size_ * sizeof(std::complex<Dtype>));
  fft_map_out_real_ = reinterpret_cast<Dtype *> (caffe_cpu_fft_malloc<Dtype>(
      std::max(this->num_output_, this->channels_) *
      fft_map_real_size_ * sizeof(Dtype)));

  // Create fft and ifft plans
  fft_handle_ = caffe_cpu_fft_plan_dft_r2c_2d<Dtype>(fft_height_, fft_width_,
      fft_map_in_real_, fft_map_in_complex_, FFTW_ESTIMATE);
  ifft_handle_ = caffe_cpu_fft_plan_dft_c2r_2d<Dtype>(fft_height_, fft_width_,
      fft_map_out_complex_, fft_map_out_real_, FFTW_ESTIMATE);

  // Create plan for batched in place transform
  int in_N[2] = { fft_height_, fft_width_ };
  int in_stride = 1;
  int in_dist = fft_height_ * 2*fft_complex_width_;
  int out_N[2] = { fft_height_, fft_complex_width_ };
  int out_stride = 1;
  int out_dist = fft_height_ * fft_complex_width_;
  int in_N_inplace[2] = { fft_height_, 2*fft_complex_width_ };
  fft_many_handle_ = caffe_cpu_fft_plan_many_dft_r2c<Dtype>(2, in_N,
      num_weights, reinterpret_cast<Dtype*>(fft_weights_complex_),
      in_N_inplace, in_stride, in_dist, fft_weights_complex_,
      out_N, out_stride, out_dist, FFTW_ESTIMATE);

  fft_cpu_initialized_ = true;
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_clean() {
  if (fft_cpu_initialized_) {
    fft_cpu_clean();
  }
#ifdef USE_GREENTEA
  if (fft_gpu_initialized_) {
    fft_gpu_clean();
  }
#endif
}

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
  }
  fft_cpu_initialized_ = false;
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_compute_weights() {
  int ch_gr = (this->channels_ / this->group_);
  int num_weights = this->num_output_ * ch_gr;
  caffe_memset(num_weights*fft_map_complex_size_*sizeof(std::complex<Dtype>),
      0., fft_weights_complex_);
  // Left-top 0-padding of weights

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int n = 0; n < this->num_output_; n++) {
    for (int c = 0; c < ch_gr; c++) {
      for (int h = 0; h < kernel_h_; h++) {
        for (int w = 0; w < kernel_w_; w++) {
          int map_offset = n * ch_gr + c;
          int src_idx = (map_offset*kernel_h_ + h)*kernel_w_ + w;
          int dst_idx = (map_offset*fft_height_ + h)*2*fft_complex_width_ + w;
          (reinterpret_cast<Dtype*>(fft_weights_complex_))[dst_idx] =
              weight[src_idx];
        }
      }
    }
  }
  // Batched in-place FFT of padded weights
  caffe_cpu_fft_execute<Dtype>(fft_many_handle_);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft_task(const Dtype* bottom_data,
         int bottom_data_offset, Dtype* top_data, int top_data_offset, int n) {
  // clear buffer
  caffe_memset((this->num_output_ * fft_map_complex_size_ *
      sizeof(std::complex<Dtype>)), 0., fft_map_out_complex_);

  int ch_gr = this->channels_ / this->group_;
  int out_gr = this->num_output_ / this->group_;
  int map_in_size = height_ * width_;
  for (int c = 0; c < this->channels_; c++) {
    caffe_memset(fft_map_real_size_ * sizeof(Dtype), 0., fft_map_in_real_);

    // Select a specific channel map in a specific feature map in bottom data
    const Dtype* map_in = const_cast<Dtype*>(bottom_data + bottom_data_offset +
        c * map_in_size);
    // Left-top 0-padding of bottom data
    for (int h = 0; h < height_; h++) {
      for (int w = 0; w < width_; w++) {
        int src_idx = h * width_ + w;
        int dst_idx = (h + pad_h_) * fft_width_ + (w + pad_w_);
        fft_map_in_real_[dst_idx] = map_in[src_idx];
      }
    }

    // FFT of padded bottom data
    caffe_cpu_fft_execute_dft_r2c<Dtype>(fft_handle_, fft_map_in_real_,
        fft_map_in_complex_);

    // Multiplication of FFT bottom data and FFT weights
    int g = c / ch_gr;
    int c_offset= c % ch_gr;
    int out_first = g * out_gr;
    int out_last = (g + 1) * out_gr;
    for (int out = out_first; out < out_last; out++) {
      std::complex<Dtype>* map_out_complex = fft_map_out_complex_ +
          out * fft_map_complex_size_;
      std::complex<Dtype>* weights_complex = fft_weights_complex_ +
          (out * ch_gr + c_offset) * fft_map_complex_size_;
      for (int i = 0; i < fft_map_complex_size_; i++) {
        // FFT for correlation requires conj (fft_of_weights)
        Dtype x_real = std::real(fft_map_in_complex_[i]);
        Dtype x_imag = std::imag(fft_map_in_complex_[i]);
        Dtype y_real = std::real(weights_complex[i]);
        Dtype y_imag = std::imag(weights_complex[i]);
        Dtype z_real = x_real*y_real + x_imag*y_imag;
        Dtype z_imag = - x_real*y_imag + x_imag*y_real;
        map_out_complex[i] += std::complex<Dtype>(z_real, z_imag);
      }
    }
  }

  Dtype ifft_scale = 1. / ((Dtype) fft_map_real_size_);
  for (int out = 0; out < this->num_output_; out++) {
    // IFFT of results
    std::complex<Dtype>* map_out_complex = fft_map_out_complex_ +
        out * fft_map_complex_size_;
    Dtype* map_out_real = fft_map_out_real_ + out * fft_map_real_size_;
    caffe_cpu_fft_execute_dft_c2r<Dtype>(ifft_handle_, map_out_complex,
        map_out_real);

    // Mapping from IFFT result to top data
    Dtype* map_out = top_data + top_data_offset + out * map_out_size_;
    for (int h_out = 0; h_out < height_out_; h_out++) {
      for (int w_out = 0; w_out < width_out_; w_out++) {
        int h = h_out  * stride_h_;
        int w = w_out  * stride_w_;
        if ((h < fft_height_) &&  (w < fft_width_)) {
          int src_idx = h * fft_width_ + w;
          int dst_idx = h_out * width_out_ + w_out;
          map_out[dst_idx] = ifft_scale * map_out_real[src_idx];
        }
      }
    }
  }
  // bias
  if (this->bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data();
    this->forward_cpu_bias(top_data + top_data_offset, bias);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft(
         const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top) {
  fft_compute_weights();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data= top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      Forward_cpu_fft_task(bottom_data, n * this->bottom_dim_, top_data,
          n * this->top_dim_, n);
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu(
         const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top) {
  Forward_cpu_fft(bottom, top);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu_fft_task(
         const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top,
         const Dtype* weight, int i, int n) {
  const Dtype* top_diff = top[i]->cpu_diff();
  Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

  // Clear buffers
  caffe_memset(fft_map_real_size_ * sizeof(Dtype), 0., fft_map_in_real_);
  caffe_memset(this->channels_ * fft_map_complex_size_*
      sizeof(std::complex<Dtype>), 0., fft_map_out_complex_);

  int ch_gr = this->channels_ / this->group_;
  int out_gr = this->num_output_ / this->group_;
  int map_in_size = height_out_ * width_out_;
  for (int out = 0; out < this->num_output_; out++) {
    const Dtype* map_in = const_cast<Dtype*>(top_diff + n * this->top_dim_ +
        out * map_in_size);
    // Left-top 0-padding of top data
    for (int h = 0; h < height_out_; h++) {
      for (int w = 0; w < width_out_; w++) {
        int h_pad = h * stride_h_;
        int w_pad = w * stride_w_;
        fft_map_in_real_[h_pad * fft_width_ + w_pad] =
              map_in[h * width_out_ + w];
      }
    }

    // FFT of padded top data
    caffe_cpu_fft_execute_dft_r2c<Dtype>(fft_handle_, fft_map_in_real_,
        fft_map_in_complex_);

    // Multiplication of FFT top data and FFT weights
    int g = out / out_gr;
    int c_first = g * ch_gr;
    int c_last = (g + 1) * ch_gr;
    for (int c = c_first; c < c_last; c++) {
      int c_offset = c % ch_gr;
      std::complex<Dtype>* map_out_complex = fft_map_out_complex_ +
          c * fft_map_complex_size_;
      std::complex<Dtype>* weights_complex = fft_weights_complex_ +
          (out * ch_gr + c_offset) * fft_map_complex_size_;
      for (int i = 0; i < fft_map_complex_size_; i++) {
        Dtype x_real = std::real(fft_map_in_complex_[i]);
        Dtype x_imag = std::imag(fft_map_in_complex_[i]);
        Dtype y_real = std::real(weights_complex[i]);
        Dtype y_imag = std::imag(weights_complex[i]);
        Dtype z_real = x_real * y_real - x_imag * y_imag;
        Dtype z_imag = x_real * y_imag + x_imag * y_real;
        map_out_complex[i] += std::complex<Dtype>(z_real, z_imag);
     }
    }
  }

  Dtype ifft_scale = 1. / ((Dtype) fft_map_real_size_);
  for (int c = 0; c < this->channels_; c++) {
    // IFFT of results
    std::complex<Dtype>* map_out_complex = fft_map_out_complex_ +
        c * fft_map_complex_size_;
    caffe_cpu_fft_execute_dft_c2r<Dtype>(ifft_handle_, map_out_complex,
        fft_map_out_real_);

    // Mapping from IFFT result to bottom data
    Dtype* map_out = reinterpret_cast<Dtype*>(bottom_diff +
        n * this->bottom_dim_ + c * map_size_);
    for (int h_out = 0; h_out < height_; h_out++) {
      for (int w_out = 0; w_out < width_; w_out++) {
        int h = h_out + pad_h_;
        int w = w_out + pad_w_;
        map_out[h_out * width_ + w_out] =
            ifft_scale * fft_map_out_real_[h * fft_width_ + w];
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu(
         const vector<Blob<Dtype>*>& top,
         const vector<bool>& propagate_down,
         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }

  // Compute weight_diff
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      if (this->param_propagate_down_[0]) {
        for (int n = 0; n < this->num_; ++n) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
      }
      if (propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          Backward_cpu_fft_task(bottom, top, weight, i, n);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
// while CPU_ONLY is on, stub functions
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_setup() { NO_GPU; }
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_clean() { NO_GPU; }
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_gpu_compute_weights() { NO_GPU; }
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft_task(const Dtype* bottom_data,
    int bottom_data_offset, Dtype* top_data, int top_data_offset, int n,
    int ch_gr, int out_gr) { NO_GPU; }
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    NO_GPU; }
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_fft_task(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    const Dtype* weight, int i, int n, int ch_gr, int out_gr) { NO_GPU; }
STUB_GPU(ConvolutionLayerFFT);
#endif  // CPU_ONLY

INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
#endif  // USE_FFT
