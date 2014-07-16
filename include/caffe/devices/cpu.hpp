// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_DEVICES_CPU_H_
#define CAFFE_DEVICES_CPU_H_

extern "C" {
#include <cblas.h>
}

#include "caffe/device.hpp"

namespace caffe {

template<typename Dtype>
class CPUDevice : public Device<Dtype> {
 public:
  CPUDevice() {
  }
  virtual ~CPUDevice() {
  }
  virtual void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);

  virtual void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype* A, const Dtype* x,
                    const Dtype beta, Dtype* y);

  virtual void axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

  virtual void axpby(const int N, const Dtype alpha, const Dtype* X,
                     const Dtype beta, Dtype* Y);

  /* NOLINT_NEXT_LINE(build/include_what_you_use) */
  virtual void copy(const int N, const Dtype *X, Dtype *Y);

  virtual void copy_from_cpu(const int N, const Dtype* X, Dtype* Y);

  virtual void set(const int N, const Dtype alpha, Dtype *X);

  virtual void add_scalar(const int N, const Dtype alpha, Dtype *X);

  virtual void scal(const int N, const Dtype alpha, Dtype *X);

  virtual void sqr(const int N, const Dtype* a, Dtype* y);

  virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

  virtual void rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* r);

  virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
                            Dtype* r);

  virtual void rng_bernoulli(const int N, const Dtype p, int* r);

  virtual void rng_bernoulli(const int N, const Dtype p, unsigned int* r);

  virtual void exp(const int N, const Dtype* a, Dtype* y);

  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out);

  virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
                                int* out);

  // Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y);

  virtual void sign(const int N, const Dtype* x, Dtype* y);

  virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

  virtual void fabs(const int N, const Dtype* x, Dtype* y);

  virtual void scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y);

  virtual void im2col(const Dtype* data_im, const int channels,
                      const int height, const int width, const int ksize,
                      const int pad, const int stride, Dtype* data_col);

  virtual void col2im(const Dtype* data_col, const int channels,
                      const int height, const int width, const int ksize,
                      const int pad, const int stride, Dtype* data_im);
};

}  // namespace caffe

#endif  // CAFFE_DEVICES_CPU_H_
