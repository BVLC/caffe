#ifndef CAFFE_UTIL_caffe_cpu_fft_H_
#define CAFFE_UTIL_caffe_cpu_fft_H_
#ifdef CMAKE_BUILD
#include <caffe_config.h>
#endif
#ifdef USE_FFT
#ifndef CPU_ONLY
#ifdef USE_GREENTEA
#include <clFFT.h>
#endif
#endif

#include <fftw3.h>
#include <complex>


namespace caffe {

inline int next_mix_of_235(int value) {
  // Using mixed radix instead of power of 2 saves more memory
  /*int k = value;
  int next_mix_of_235 = value;
  while (1) {
    while (k % 2 == 0) {
      k /= 2;
    }
    while (k % 3 == 0) {
      k /= 3;
    }
    while (k % 5 == 0) {
      k /= 5;
    }
    if (k == 1) {
      return next_mix_of_235;
    } else {
      k = ++next_mix_of_235;
    }
  }*/
  // Power of 2
  value -= 1;
  int power = 1;
  while (power < sizeof(int)*8) {
    value |= (value >> power);
    power <<= 1;
  }
  return (value+1);
}

template <typename Dtype> void* caffe_cpu_fft_malloc(int n);
template <typename Dtype> void  caffe_cpu_fft_free(void* p);
template <typename Dtype> void* caffe_cpu_fft_plan_dft_r2c_2d(int n0, int n1,
    Dtype *in, std::complex<Dtype> *out, unsigned flags);
template <typename Dtype> void* caffe_cpu_fft_plan_dft_c2r_2d(int n0, int n1,
    std::complex<Dtype> *in, Dtype *out, unsigned flags);
template <typename Dtype> void* caffe_cpu_fft_plan_many_dft_r2c(int rank,
    const int *n, int howmany, Dtype *in, const int *inemded, int istride,
    int idist, std::complex<Dtype> *out, const int *onembed, int ostride,
    int odist, unsigned flags);
template <typename Dtype> void caffe_cpu_fft_destroy_plan(void* plan);
template <typename Dtype> void caffe_cpu_fft_execute(const void* plan);
template <typename Dtype> void caffe_cpu_fft_execute_dft_r2c(const void* plan,
    Dtype *in, std::complex<Dtype> *out);
template <typename Dtype> void caffe_cpu_fft_execute_dft_c2r(const void* plan,
    std::complex<Dtype> *in, Dtype  *out);

// --- GPU ---

#ifndef CPU_ONLY
#ifdef USE_GREENTEA
template <typename T>
struct DtypeComplex {
  T x, y;
};
void clear_gpu_fft_buffer(void* data, const int size);
template <typename Dtype>
void fft_gpu_copy2buffer(Dtype* fft_gpu_weights_real, const Dtype* weight,
    int num_output, int group, int channels, int ker_h, int ker_w,
    int ker_c_h, int ker_c_w, int fft_height, int fft_width);
/*template <typename Dtype>
void fft_gpu_copy2buffer_in(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int height, int width,
    int stride_h, int stride_w, int pad_h, int pad_w);
*/
template <typename Dtype>
void fft_gpu_copy2buffer_in_2D(Dtype* map_out, const Dtype* map_in,
    int in_offset, int channels, int height_out, int width_out, int height,
    int width, int stride_h, int stride_w, int pad_h, int pad_w);
/*template <typename Dtype>
void fft_gpu_copy2buffer_out_forward(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
*/
template <typename Dtype>
void fft_gpu_copy2buffer_out_forward_2D(Dtype* map_out, int out_offset,
    const Dtype* map_in, int num_output, int height_out, int width_out,
    int fft_height, int fft_width, int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template <typename Dtype>
void fft_gpu_copy2buffer_out_backward(Dtype* map_out, const Dtype* map_in,
    int height_out, int width_out, int fft_height, int fft_width,
    int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template <typename Dtype>
void fft_gpu_copy2buffer_out_backward_2D(Dtype* map_out, int out_offset,
    const Dtype* map_in, int channels, int height_out, int width_out,
    int fft_height, int fft_width, int kernel_center_h, int kernel_center_w,
    int stride_h, int stride_w, int pad_h, int pad_w);
template <typename Dtype>
void caffe_gpu_elementMulConj_1D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int map_size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMulConj_Reshape(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMulConj_2D(DtypeComplex<Dtype>* dst, int dst_offset,
    const DtypeComplex<Dtype>* src1, int src1_offset,
    const DtypeComplex<Dtype>* src2, int src2_offset,
    const int out_gr, const int map_size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMulConj_2D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMulConj_3D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMulConj_3D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int out_gr, const int map_size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMul_1D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int size, const int ch_gr);
template <typename Dtype>
void caffe_gpu_elementMul_2D_SLM(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int size, const int ch_gr, const int num_output);
template <typename Dtype>
void caffe_gpu_elementMul_3D(DtypeComplex<Dtype>* dst,
    const DtypeComplex<Dtype>* src1, const DtypeComplex<Dtype>* src2,
    const int size, const int ch_gr, const int out_gr, const int num_output);
template <typename Dtype>
void caffe_gpu_fft_execute_r2c(clfftPlanHandle plan, const Dtype* in,
    DtypeComplex<Dtype>* out);
template <typename Dtype>
void caffe_gpu_fft_execute_r2c_inplace(clfftPlanHandle plan, Dtype* inout);
template <typename Dtype>
void caffe_gpu_fft_execute_c2r(clfftPlanHandle plan,
    const DtypeComplex<Dtype>* in, Dtype* out);
template <typename Dtype>
void reshape_weights(DtypeComplex<Dtype>* dst, DtypeComplex<Dtype>* src,
    const int size, const int num_output, const int ch_gr);
#endif  // USE_GREENTEA
#endif  // CPU_ONLY

}  // namespace caffe

#endif  // USE_FFT

#endif  // CAFFE_UTIL_caffe_cpu_fft_H_
