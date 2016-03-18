#ifdef USE_FFT
#include "caffe/util/fft.hpp"

namespace caffe {

template <>
void* caffe_cpu_fft_malloc<float>(int n) {
  return (reinterpret_cast<void *>(fftwf_malloc(n)));
}
template <>
void* caffe_cpu_fft_malloc<double>(int n) {
  return (reinterpret_cast<void *>(fftw_malloc(n)));
}

template <>
void caffe_cpu_fft_free<float>(void * p) {
  fftwf_free(p);
}
template <>
void caffe_cpu_fft_free<double>(void * p) {
  fftw_free(p);
}

template <>
void* caffe_cpu_fft_plan_dft_r2c_2d<float>(int n0, int n1,
    float *in, std::complex<float> *out, unsigned flags) {
  return (reinterpret_cast<void *>(
      fftwf_plan_dft_r2c_2d(n0, n1, in,
      reinterpret_cast<fftwf_complex *>(out), flags)));
}
template <>
void* caffe_cpu_fft_plan_dft_r2c_2d<double>(int n0, int n1,
    double *in, std::complex<double> *out, unsigned flags) {
  return (reinterpret_cast<void *>(
      fftw_plan_dft_r2c_2d(n0, n1, in,
      reinterpret_cast<fftw_complex*>(out), flags)));
}

template <>
void* caffe_cpu_fft_plan_dft_c2r_2d<float>(int n0, int n1,
    std::complex<float> *in, float *out, unsigned flags) {
  return (reinterpret_cast<void *>(
      fftwf_plan_dft_c2r_2d(n0, n1, reinterpret_cast<fftwf_complex *> (in),
      out, flags)));
}
template <>
void* caffe_cpu_fft_plan_dft_c2r_2d<double>(int n0, int n1,
    std::complex<double> *in, double *out, unsigned flags) {
  return (reinterpret_cast<void *>(
      fftw_plan_dft_c2r_2d(n0, n1, reinterpret_cast<fftw_complex*> (in),
      out, flags)));
}

template <>
void* caffe_cpu_fft_plan_many_dft_r2c<float>(int rank, const int *n,
    int howmany, float *in, const int *inemded, int istride, int idist,
    std::complex<float> *out, const int *onembed, int ostride, int odist,
    unsigned flags) {
  return (reinterpret_cast<void *>(
      fftwf_plan_many_dft_r2c(rank, n, howmany, in, inemded, istride, idist,
      reinterpret_cast<fftwf_complex *> (out), onembed, ostride, odist,
      flags)));
}
template <>
void* caffe_cpu_fft_plan_many_dft_r2c<double>(int rank, const int *n,
    int howmany, double *in, const int *inemded, int istride, int idist,
    std::complex<double> *out, const int *onembed, int ostride, int odist,
    unsigned flags) {
  return (reinterpret_cast<void *>(
      fftw_plan_many_dft_r2c(rank, n, howmany, in, inemded, istride, idist,
      reinterpret_cast<fftw_complex*> (out), onembed, ostride, odist,
      flags)));
}

template <>
void caffe_cpu_fft_destroy_plan<float>(void* plan) {
  fftwf_destroy_plan((fftwf_plan)plan);
}
template <>
void caffe_cpu_fft_destroy_plan<double>(void* plan) {
  fftw_destroy_plan((fftw_plan)plan);
}

template <>
void caffe_cpu_fft_execute<float>(const void* plan) {
  fftwf_execute((const fftwf_plan) plan);
}
template <>
void caffe_cpu_fft_execute<double>(const void* plan) {
  fftw_execute((const fftw_plan) plan);
}
template <>
void caffe_cpu_fft_execute_dft_r2c<float>(const void* plan,
    float *in, std::complex<float> *out) {
  fftwf_execute_dft_r2c((const fftwf_plan) plan, in,
      reinterpret_cast<fftwf_complex *> (out));
}
template <>
void caffe_cpu_fft_execute_dft_r2c<double>(const void* plan,
    double *in, std::complex<double> *out) {
  fftw_execute_dft_r2c((const fftw_plan) plan, in,
      reinterpret_cast<fftw_complex*> (out));
}
template <>
void caffe_cpu_fft_execute_dft_c2r<float>(const void* plan,
    std::complex<float> *in, float *out) {
  fftwf_execute_dft_c2r((const fftwf_plan) plan,
      reinterpret_cast<fftwf_complex *> (in), out);
}
template <>
void caffe_cpu_fft_execute_dft_c2r<double>(const void* plan,
    std::complex<double> *in, double *out) {
  fftw_execute_dft_c2r((const fftw_plan) plan,
      reinterpret_cast<fftw_complex*> (in),  out);
}

}  // namespace caffe
#endif  // USE_FFT
