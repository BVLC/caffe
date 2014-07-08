// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_DEVICE_H_
#define CAFFE_UTIL_DEVICE_H_

#include <cublas_v2.h>
#include <stdint.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

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

  virtual void copy(const int N, const Dtype *X, Dtype *Y) { NOT_IMPLEMENTED; }
  virtual void copy_from_cpu(const int N, const Dtype* X, Dtype* Y) {
    NOT_IMPLEMENTED; }

  virtual void set(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED; }

  virtual void add_scalar(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED; }

  virtual void scal(const int N, const Dtype alpha, Dtype *X) {
    NOT_IMPLEMENTED; }

  virtual void sqr(const int N, const Dtype* a, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED; }

  virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED; }

  virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED; }

  virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
    NOT_IMPLEMENTED; }

  virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y) {
    NOT_IMPLEMENTED; }

  virtual void rng_uniform(const int N, const Dtype a, const Dtype b,
                           Dtype* r) { NOT_IMPLEMENTED; }

  virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
                            Dtype* r) { NOT_IMPLEMENTED; }

  virtual void rng_bernoulli(const int N, const Dtype p, int* r) {
    NOT_IMPLEMENTED; }

  virtual void rng_bernoulli(const int N, const Dtype p, unsigned int* r) {
    NOT_IMPLEMENTED; }

  virtual void exp(const int N, const Dtype* a, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out) {
    NOT_IMPLEMENTED; }

  virtual uint32_t hamming_distance(const int N, const Dtype* x,
      const Dtype* y) { NOT_IMPLEMENTED; return 0; }

// Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void sign(const int N, const Dtype* x, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void sgnbit(const int N, const Dtype* x, Dtype* y) {
    NOT_IMPLEMENTED; }

  virtual void fabs(const int N, const Dtype* x, Dtype* y) { NOT_IMPLEMENTED; }

  virtual void scale(const int N, const Dtype alpha, const Dtype *x,
                     Dtype* y) { NOT_IMPLEMENTED; }

  virtual void im2col(const Dtype* data_im, const int channels,
      const int height, const int width, const int ksize, const int pad,
      const int stride, Dtype* data_col) { NOT_IMPLEMENTED; }

  virtual void col2im(const Dtype* data_col, const int channels,
      const int height, const int width, const int ksize, const int pad,
      const int stride, Dtype* data_im) { NOT_IMPLEMENTED; }
};

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

  virtual uint32_t hamming_distance(const int N, const Dtype* x,
      const Dtype* y);

// Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y);

  virtual void sign(const int N, const Dtype* x, Dtype* y);

  virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

  virtual void fabs(const int N, const Dtype* x, Dtype* y);

  virtual void scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y);

  virtual void im2col(const Dtype* data_im, const int channels,
      const int height, const int width, const int ksize, const int pad,
      const int stride, Dtype* data_col);

  virtual void col2im(const Dtype* data_col, const int channels,
      const int height, const int width, const int ksize, const int pad,
      const int stride, Dtype* data_im);
};

template<typename Dtype>
class GPUDevice : public Device<Dtype> {
 public:
  GPUDevice() {
  }
  virtual ~GPUDevice() {
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

  virtual void exp(const int N, const Dtype* a, Dtype* y);

  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out);

  virtual uint32_t hamming_distance(const int N, const Dtype* x,
      const Dtype* y);

// Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y);

  virtual void sign(const int N, const Dtype* x, Dtype* y);

  virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

  virtual void fabs(const int N, const Dtype* x, Dtype* y);

  virtual void scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y);

  virtual void im2col(const Dtype* data_im, const int channels,
      const int height, const int width, const int ksize, const int pad,
      const int stride, Dtype* data_col);

  virtual void col2im(const Dtype* data_col, const int channels,
      const int height, const int width, const int psize, const int pad,
      const int stride, Dtype* data_im);
};

// Device factory function
template<typename Dtype>
Device<Dtype>* GetDevice(Caffe::Brew mode = Caffe::UNSPECIFIED);

}  // namespace caffe

#endif  // CAFFE_UTIL_DEVICE_H_
