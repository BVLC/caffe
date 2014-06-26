// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_caffe_cpu_fft_H_
#define CAFFE_UTIL_caffe_cpu_fft_H_

#include <complex>
#include <fftw3.h>

// #ifdef USE_MKL
// #include <fftw3_mkl.h>
// #endif

namespace caffe {

template <typename Dtype>
void* caffe_cpu_fft_malloc(int n);

template <typename Dtype>
void  caffe_cpu_fft_free(void* p);

template <typename Dtype>
void* caffe_cpu_fft_plan_dft_r2c_2d(int n0, int n1,
    Dtype *in, std::complex<Dtype> *out, unsigned flags);

template <typename Dtype>
void* caffe_cpu_fft_plan_dft_c2r_2d(int n0, int n1,
    std::complex<Dtype> *in, Dtype *out, unsigned flags);

template <typename Dtype>
void* caffe_cpu_fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
    Dtype *in, const int *inemded, int istride, int idist,
    std::complex<Dtype> *out, const int *onembed, int ostride, int odist,
    unsigned flags);

template <typename Dtype>
void caffe_cpu_fft_destroy_plan(void* plan);

template <typename Dtype>
void caffe_cpu_fft_execute(void* plan);

template <typename Dtype>
void caffe_cpu_fft_execute_dft_r2c(void* plan,
    Dtype *in, std::complex<Dtype> *out);

template <typename Dtype>
void caffe_cpu_fft_execute_dft_c2r(void* plan,
    std::complex<Dtype> *in, Dtype  *out);

inline
int caffe_cpu_fft_init_threads(void ) {
  return(fftw_init_threads());
}
inline
void  caffe_cpu_fft_plan_with_nthreads(int num_of_threads_) {
  fftw_plan_with_nthreads(num_of_threads_);
}
inline
void caffe_cpu_fft_cleanup_threads(void) {
  fftw_cleanup_threads();
}

}  // namespace caffe

#endif  // CAFFE_UTIL_caffe_cpu_fft_H_
