// Copyright 2014 BVLC and contributors.

#include "caffe/util/fft.hpp"

namespace caffe {

template <>
void* caffe_cpu_fft_malloc<float>(int n) {
  return((void*)fftwf_malloc(n));
}
template <>
void* caffe_cpu_fft_malloc<double>(int n) {
  return((void*)fftw_malloc(n));
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
  fftwf_plan_dft_r2c_2d(n0, n1, in, (fftwf_complex *)out, flags);
}
template <>
void* caffe_cpu_fft_plan_dft_r2c_2d<double>(int n0, int n1,
         double *in, std::complex<double> *out, unsigned flags) {
  fftw_plan_dft_r2c_2d(n0, n1, in, (fftw_complex *)out, flags);
}

template <>
void* caffe_cpu_fft_plan_dft_c2r_2d<float>(int n0, int n1,
         std::complex<float> *in, float *out, unsigned flags) {
  fftwf_plan_dft_c2r_2d(n0, n1, (fftwf_complex *) in, out, flags);
}
template <>
void* caffe_cpu_fft_plan_dft_c2r_2d<double>(int n0, int n1,
         std::complex<double> *in, double *out, unsigned flags) {
  fftw_plan_dft_c2r_2d(n0, n1, (fftw_complex *) in, out, flags);
}

template <>
void* caffe_cpu_fft_plan_many_dft_r2c<float>(int rank, const int *n,
         int howmany, float *in, const int *inemded, int istride, int idist,
         std::complex<float> *out, const int *onembed, int ostride, int odist,
         unsigned flags) {
  fftwf_plan_many_dft_r2c(rank, n, howmany, in, inemded, istride, idist,
         (fftwf_complex *) out, onembed, ostride, odist, flags);
}
template <>
void* caffe_cpu_fft_plan_many_dft_r2c<double>(int rank, const int *n,
         int howmany, double *in, const int *inemded, int istride, int idist,
         std::complex<double> *out, const int *onembed, int ostride, int odist,
         unsigned flags) {
  fftw_plan_many_dft_r2c(rank, n, howmany, in, inemded, istride, idist,
         (fftw_complex *) out, onembed, ostride, odist, flags);
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
void caffe_cpu_fft_execute<float>(void* plan) {
  fftwf_execute((const fftwf_plan) plan);
}
template <>
void caffe_cpu_fft_execute<double>(void* plan) {
  fftw_execute((const fftw_plan) plan);
}
template <>
void caffe_cpu_fft_execute_dft_r2c<float>(void* plan,
         float *in, std::complex<float> *out) {
  fftwf_execute_dft_r2c((const fftwf_plan) plan, in, (fftwf_complex *) out);
}
template <>
void caffe_cpu_fft_execute_dft_r2c<double>(void* plan,
         double *in, std::complex<double> *out) {
  fftw_execute_dft_r2c((const fftw_plan) plan, in, (fftw_complex *) out);
}
template <>
void caffe_cpu_fft_execute_dft_c2r<float>(void* plan,
          std::complex<float> *in, float *out) {
  fftwf_execute_dft_c2r((const fftwf_plan) plan, (fftwf_complex *) in, out);
}
template <>
void caffe_cpu_fft_execute_dft_c2r<double>(void* plan,
         std::complex<double> *in, double *out) {
  fftw_execute_dft_c2r((const fftw_plan) plan, (fftw_complex *) in,  out);
}

}  // namespace caffe
