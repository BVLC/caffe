#ifndef CAFFE_CONV_FFT_LAYER_HPP_
#define CAFFE_CONV_FFT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_FFT
#ifndef CPU_ONLY
#ifdef USE_GREENTEA
#include <clFFT.h>
#endif
#endif
#endif

#ifdef USE_FFT
#include <complex>
#endif

namespace caffe {
#ifdef USE_FFT

template <typename Dtype>
class ConvolutionLayerFFT : public BaseConvolutionLayer<Dtype> {
 public:
  explicit ConvolutionLayerFFT(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) , fft_cpu_initialized_(false),
        fft_gpu_initialized_(false) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual ~ConvolutionLayerFFT<Dtype>();

  virtual inline const char* type() const { return "Convolution"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
#ifdef USE_GREENTEA
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
#endif
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
#ifdef USE_GREENTEA
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
#endif

  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  // Forward CPU
  virtual void Forward_cpu_fft(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu_fft_task(const Dtype *bottom_data,
      int bottom_data_offset, Dtype* top_data, int top_data_offset, int n);
  virtual void fft_compute_weights();
  // Forward GPU
#ifdef USE_GREENTEA
  virtual void Forward_gpu_fft(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu_fft_task(const Dtype *bottom_data,
      int bottom_data_offset, Dtype* top_data, int top_data_offset, int n,
      int ch_gr, int out_gr);
  virtual void fft_gpu_compute_weights();
#endif
  // Backward CPU
  virtual void Backward_cpu_fft_task(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const Dtype* weight, int i, int n);
  // Backward GPU
#ifdef USE_GREENTEA
  virtual void Backward_gpu_fft_task(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top, const Dtype* weight, int i, int n,
      int ch_gr, int out_gr);
#endif

  // fft setup function for CPU and GPU
  virtual void fft_setup(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void fft_cpu_setup();
#ifdef USE_GREENTEA
  virtual void fft_gpu_setup();
#endif
  virtual void fft_clean();
  virtual void fft_cpu_clean();
#ifdef USE_GREENTEA
  virtual void fft_gpu_clean();
#endif

  // FFT variables
  bool fft_cpu_initialized_;
  bool fft_gpu_initialized_;
  int fft_height_;
  int fft_width_;
  int fft_complex_width_;
  int fft_map_real_size_;
  int fft_map_complex_size_;
  int map_size_;
  int map_out_size_;
  int kernel_center_h_;
  int kernel_center_w_;
  int kernel_h_;
  int kernel_w_;
  int height_;
  int width_;
  int height_out_;
  int width_out_;
  int pad_w_;
  int pad_h_;
  int stride_w_;
  int stride_h_;

  // CPU buffers and handles
  Dtype* fft_weights_real_;
  Dtype* fft_map_in_real_;
  std::complex<Dtype>* fft_weights_complex_;
  std::complex<Dtype>* fft_map_in_complex_;
  std::complex<Dtype>* fft_map_out_complex_;
  Dtype* fft_map_out_real_;
  void* fft_handle_;
  void* ifft_handle_;
  void* fft_many_handle_;

  // GPU buffers and handles
#ifndef CPU_ONLY
#ifdef USE_GREENTEA
  // FFT data in Forward
  clfftPlanHandle fft_gpu_forward_many_handle_;
  void* fft_gpu_map_in_real_all_channels_;
  void* fft_gpu_map_in_complex_all_channels_;
  // FFT data in Backward
  clfftPlanHandle fft_gpu_backward_many_handle_;
  void* fft_gpu_map_in_real_all_num_output_;
  void* fft_gpu_map_in_complex_all_num_output_;
  // FFT weight in Forward
  clfftPlanHandle fft_gpu_many_weights_handle_;
  void* fft_gpu_weights_complex_;
  // IFFT
  clfftPlanHandle ifft_gpu_forward_many_handle_;
  clfftPlanHandle ifft_gpu_backward_many_handle_;
  void* fft_gpu_map_out_complex_;
  void* fft_gpu_map_out_real_;
#endif
#endif
};
#endif  // USE_FFT

}  // namespace caffe

#endif  // CAFFE_CONV_FFT_LAYER_HPP_
