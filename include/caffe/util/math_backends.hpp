// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_MATH_BACKENDS_H_
#define CAFFE_UTIL_MATH_BACKENDS_H_

#include <cublas_v2.h>
#include <stdint.h>

#include "glog/logging.h"

#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
class MathBackend {
public:
	virtual ~MathBackend() {}
	virtual void gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
			Dtype* C) = 0;

	virtual void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
			const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
			Dtype* y) = 0;

	virtual void axpy(const int N, const Dtype alpha, const Dtype* X,
			Dtype* Y) = 0;

	virtual void axpby(const int N, const Dtype alpha, const Dtype* X,
			const Dtype beta, Dtype* Y) = 0;

	virtual void copy(const int N, const Dtype *X, Dtype *Y) = 0;

	virtual void set(const int N, const Dtype alpha, Dtype *X) = 0;

	virtual void add_scalar(const int N, const Dtype alpha, Dtype *X) = 0;

	virtual void scal(const int N, const Dtype alpha, Dtype *X) = 0;

	virtual void sqr(const int N, const Dtype* a, Dtype* y) = 0;

	virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y) = 0;

	virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) = 0;

	virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) = 0;

	virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y) = 0;

	virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y) = 0;

	virtual void rng_uniform(const int N, const Dtype a, const Dtype b,
			Dtype* r) = 0;

	virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
			Dtype* r) = 0;

	virtual void rng_bernoulli(const int N, const Dtype p, int* r) = 0;

	virtual void exp(const int N, const Dtype* a, Dtype* y) = 0;

	virtual void dot(const int N, const Dtype* x, const Dtype* y,
			Dtype* out) = 0;

	virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
			uint32_t* out) = 0;

// Returns the sum of the absolute values of the elements of vector x
	virtual void asum(const int N, const Dtype* x, Dtype* y) = 0;

	virtual void sign(const int N, const Dtype* x, Dtype* y) = 0;

	virtual void sgnbit(const int N, const Dtype* x, Dtype* y) = 0;

	virtual void fabs(const int N, const Dtype* x, Dtype* y) = 0;

	virtual void scale(const int N, const Dtype alpha, const Dtype *x,
			Dtype* y) = 0;
};

template<typename Dtype>
class CPUMathBackend: public MathBackend<Dtype> {
public:
	CPUMathBackend() {}
	virtual ~CPUMathBackend() {}
	virtual void gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
			Dtype* C);

	virtual void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
			const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
			Dtype* y);

	virtual void axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

	virtual void axpby(const int N, const Dtype alpha, const Dtype* X,
			const Dtype beta, Dtype* Y);

	virtual void copy(const int N, const Dtype *X, Dtype *Y);

	virtual void set(const int N, const Dtype alpha, Dtype *X);

	virtual void add_scalar(const int N, const Dtype alpha, Dtype *X);

	virtual void scal(const int N, const Dtype alpha, Dtype *X);

	virtual void sqr(const int N, const Dtype* a, Dtype* y);

	virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

	virtual void rng_uniform(const int N, const Dtype a, const Dtype b,
			Dtype* r);

	virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
			Dtype* r);

	virtual void rng_bernoulli(const int N, const Dtype p, int* r);

	virtual void exp(const int N, const Dtype* a, Dtype* y);

	virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out);

	virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
			uint32_t* out);

// Returns the sum of the absolute values of the elements of vector x
	virtual void asum(const int N, const Dtype* x, Dtype* y);

	virtual void sign(const int N, const Dtype* x, Dtype* y);

	virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

	virtual void fabs(const int N, const Dtype* x, Dtype* y);

	virtual void scale(const int N, const Dtype alpha, const Dtype *x,
			Dtype* y);
};

template<typename Dtype>
class GPUMathBackend: public MathBackend<Dtype> {
public:
	GPUMathBackend() {}
	virtual ~GPUMathBackend() {}
	virtual void gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
			Dtype* C);

	virtual void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
			const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
			Dtype* y);

	virtual void axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

	virtual void axpby(const int N, const Dtype alpha, const Dtype* X,
			const Dtype beta, Dtype* Y);

	virtual void copy(const int N, const Dtype *X, Dtype *Y);

	virtual void set(const int N, const Dtype alpha, Dtype *X);

	virtual void add_scalar(const int N, const Dtype alpha, Dtype *X);

	virtual void scal(const int N, const Dtype alpha, Dtype *X);

	virtual void sqr(const int N, const Dtype* a, Dtype* y);

	virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

	virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

	virtual void rng_uniform(const int N, const Dtype a, const Dtype b,
			Dtype* r);

	virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
			Dtype* r);

	virtual void rng_bernoulli(const int N, const Dtype p, int* r);

	virtual void exp(const int N, const Dtype* a, Dtype* y);

	virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out);

	virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
			uint32_t* out);

// Returns the sum of the absolute values of the elements of vector x
	virtual void asum(const int N, const Dtype* x, Dtype* y);

	virtual void sign(const int N, const Dtype* x, Dtype* y);

	virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

	virtual void fabs(const int N, const Dtype* x, Dtype* y);

	virtual void scale(const int N, const Dtype alpha, const Dtype *x,
			Dtype* y);
};

template<typename Dtype>
class MathBackendFactory {
public:
	static MathBackend<Dtype>* GetMathBackend();
private:
	static MathBackend<Dtype>* cpu_math_backend_;
	static MathBackend<Dtype>* gpu_math_backend_;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_BACKENDS_H_
