#ifndef CAFFE_DEVICES_CPU_H_
#define CAFFE_DEVICES_CPU_H_

#include "caffe/util/mkl_alternate.hpp"

#include "caffe/device.hpp"

namespace caffe {

template<typename Dtype>
class CPUDevice : public Device<Dtype> {
 public:
  CPUDevice() {
  }

  virtual ~CPUDevice() {
  }

  virtual void abs(const int n, const Dtype* a, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void add_scalar(const int N, const Dtype alpha, Dtype *X);

  // Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void axpby(const int N, const Dtype alpha, const Dtype* X,
      const Dtype beta, Dtype* Y) {
    NOT_IMPLEMENTED;
  }

  virtual void axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y) {
    NOT_IMPLEMENTED;
  }

  virtual void col2im(const Dtype* data_col, const int channels,
      const int height, const int width, const int patch_h, const int patch_w,
      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      Dtype* data_im);

  /* NOLINT_NEXT_LINE(build/include_what_you_use) */
  virtual void copy(const int N, const Dtype *X, Dtype *Y);

  virtual inline void copy_void(const size_t N, const void *X, void* Y) {
    NOT_IMPLEMENTED;
  }

  virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out) {
    NOT_IMPLEMENTED;
  }

  virtual void exp(const int N, const Dtype* a, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void fabs(const int N, const Dtype* x, Dtype* y);

  virtual void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
      const int M, const int N, const int K, const Dtype alpha, const Dtype* A,
      const Dtype* B, const Dtype beta, Dtype* C) {
    NOT_IMPLEMENTED;
  }

  virtual void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
      const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
      Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
      int* out) {
    NOT_IMPLEMENTED;
  }

  virtual void im2col(const Dtype* data_im, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
      Dtype* data_col);

  virtual void log(const int n, const Dtype* a, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void mem_set(const size_t N, const int alpha, void* X) {
    NOT_IMPLEMENTED;
  }

  virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void rng_bernoulli(const int N, const Dtype p, int* r);

  virtual void rng_bernoulli(const int N, const Dtype p, unsigned int* r);

  virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
      Dtype* r);

  virtual void rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* r);

  virtual void scal(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED;
  }

  virtual void scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void set(const int N, const Dtype alpha, Dtype *X);

  virtual inline void set_void(const size_t N, const int alpha, void *X) {
    NOT_IMPLEMENTED;
  }

  virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

  virtual void sign(const int N, const Dtype* x, Dtype* y);

  virtual void sqr(const int N, const Dtype* a, Dtype* y) {
    NOT_IMPLEMENTED;
  }

  virtual void strided_dot(const int n, const Dtype* x, const int incx,
      const Dtype* y, const int incy, Dtype* out) {
    NOT_IMPLEMENTED;
  }

  virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif  // CAFFE_DEVICES_CPU_H_
