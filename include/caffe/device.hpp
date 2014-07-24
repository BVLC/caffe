// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_DEVICE_H_
#define CAFFE_DEVICE_H_

extern "C" {
#include <cblas.h>
}

#include "caffe/common.hpp"

namespace caffe {

template<typename Dtype>
class Device {
 public:
  virtual ~Device() {
  }
  virtual void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype* A, const Dtype* B, const Dtype beta,
                    Dtype* C) { NOT_IMPLEMENTED; }

  virtual void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype* A, const Dtype* x,
                    const Dtype beta, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void axpy(const int N, const Dtype alpha, const Dtype* X,
                    Dtype* Y) { NOT_IMPLEMENTED; }

  virtual void axpby(const int N, const Dtype alpha, const Dtype* X,
                     const Dtype beta, Dtype* Y) { NOT_IMPLEMENTED; }

  /* NOLINT_NEXT_LINE(build/include_what_you_use) */
  virtual void copy(const int N, const Dtype *X, Dtype *Y) { NOT_IMPLEMENTED; }

  virtual void set(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED;
  }

  virtual void add_scalar(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED;
  }

  virtual void scal(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED;
  }

  virtual void sqr(const int N, const Dtype* a, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void rng_uniform(const int N, const Dtype a, const Dtype b,
                           Dtype* r) { NOT_IMPLEMENTED; }

  virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
                            Dtype* r) { NOT_IMPLEMENTED; }

  virtual void rng_bernoulli(const int N, const Dtype p, int* r) {
    NOT_IMPLEMENTED;
  }

  virtual void rng_bernoulli(const int N, const Dtype p, unsigned int* r) {
    NOT_IMPLEMENTED;
  }

  virtual void exp(const int N, const Dtype* a, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out) {
    NOT_IMPLEMENTED;
  }

  virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
                                int* out) { NOT_IMPLEMENTED; }

  // Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void sign(const int N, const Dtype* x, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void sgnbit(const int N, const Dtype* x, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void fabs(const int N, const Dtype* x, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void scale(const int N, const Dtype alpha, const Dtype *x,
                     Dtype* y) { NOT_IMPLEMENTED; }

  virtual void im2col(const Dtype* data_im, const int channels,
                      const int height, const int width, const int ksize,
                      const int pad, const int stride, Dtype* data_col) {
    NOT_IMPLEMENTED;
  }

  virtual void col2im(const Dtype* data_col, const int channels,
                      const int height, const int width, const int ksize,
                      const int pad, const int stride, Dtype* data_im) {
    NOT_IMPLEMENTED;
  }
};

// Device factory function
template<typename Dtype>
Device<Dtype>* GetDevice(Caffe::Brew mode = Caffe::UNSPECIFIED);

}  // namespace caffe

#endif  // CAFFE_DEVICE_H_
