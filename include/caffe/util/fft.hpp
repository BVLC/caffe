// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_FFT_H_
#define CAFFE_UTIL_FFT_H_

#include <complex>
#include <fftw3.h>

namespace caffe {

template <typename Dtype>
void* fft_malloc(int n);

template <typename Dtype>
void  fft_free(void* p);

template <typename Dtype>
void* fft_plan_dft_r2c_2d(int n0, int n1,
    Dtype *in, std::complex<Dtype> *out, unsigned flags);

template <typename Dtype>
void* fft_plan_dft_c2r_2d(int n0, int n1,
    std::complex<Dtype> *in, Dtype *out, unsigned flags);

template <typename Dtype>
void  fft_destroy_plan(void* plan);

template <typename Dtype>
void  fft_execute_dft_r2c(void* plan, Dtype *in, std::complex<Dtype> *out);

template <typename Dtype>
 void  fft_execute_dft_c2r(void* plan, std::complex<Dtype> *in,Dtype  *out);

}  // namespace caffe


#endif  // CAFFE_UTIL_FFT_H_
