// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/math_backends.hpp"

namespace caffe {
template<typename Dtype>
void CPUMathBackend<Dtype>::gemm(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		Dtype* C) {
	caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::gemv(const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const Dtype alpha, const Dtype* A, const Dtype* x,
		const Dtype beta, Dtype* y) {
	caffe_cpu_gemv<Dtype>(TransA, M, N, alpha, A, x, beta, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::axpy(const int N, const Dtype alpha, const Dtype* X,
		Dtype* Y) {
	caffe_axpy<Dtype>(N, alpha, X, Y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::axpby(const int N, const Dtype alpha, const Dtype* X,
		const Dtype beta, Dtype* Y) {
	caffe_cpu_axpby<Dtype>(N, alpha, X, beta, Y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::copy(const int N, const Dtype *X, Dtype *Y) {
	caffe_copy<Dtype>(N, X, Y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::set(const int N, const Dtype alpha, Dtype *X) {
	caffe_set<Dtype>(N, alpha, X);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::add_scalar(const int N, const Dtype alpha, Dtype *X) {
	caffe_add_scalar<Dtype>(N, alpha, X);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::scal(const int N, const Dtype alpha, Dtype *X) {
	caffe_scal<Dtype>(N, alpha, X);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::sqr(const int N, const Dtype* a, Dtype* y) {
	caffe_sqr<Dtype>(N, a, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::add(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	caffe_add<Dtype>(N, a, b, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::sub(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	caffe_sub<Dtype>(N, a, b, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::mul(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	caffe_mul<Dtype>(N, a, b, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::div(const int N, const Dtype* a, const Dtype* b,
		Dtype* y) {
	caffe_div<Dtype>(N, a, b, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::powx(const int N, const Dtype* a, const Dtype b,
		Dtype* y) {
	caffe_powx<Dtype>(N, a, b, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::rng_uniform(const int N, const Dtype a, const Dtype b,
		Dtype* r) {
	caffe_rng_uniform<Dtype>(N, a, b, r);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::rng_gaussian(const int N, const Dtype mu,
		const Dtype sigma, Dtype* r) {
	caffe_rng_gaussian<Dtype>(N, mu, sigma, r);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::rng_bernoulli(const int N, const Dtype p, int* r) {
	caffe_rng_bernoulli<Dtype>(N, p, r);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::exp(const int N, const Dtype* a, Dtype* y) {
	caffe_exp<Dtype>(N, a, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::dot(const int N, const Dtype* x, const Dtype* y,
		Dtype* out) {
	*out = caffe_cpu_dot<Dtype>(N, x, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::hamming_distance(const int N, const Dtype* x,
		const Dtype* y, uint32_t* out) {
	*out = caffe_cpu_hamming_distance<Dtype>(N, x, y);
}

template<typename Dtype>
// Returns the sum of the absolute values of the elements of vector x
void CPUMathBackend<Dtype>::asum(const int N, const Dtype* x, Dtype* y) {
	*y = caffe_cpu_asum<Dtype>(N, x);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::sign(const int N, const Dtype* x, Dtype* y) {
	caffe_cpu_sign<Dtype>(N, x, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::sgnbit(const int N, const Dtype* x, Dtype* y) {
	caffe_gpu_sgnbit<Dtype>(N, x, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::fabs(const int N, const Dtype* x, Dtype* y) {
	caffe_cpu_fabs<Dtype>(N, x, y);
}

template<typename Dtype>
void CPUMathBackend<Dtype>::scale(const int N, const Dtype alpha, const Dtype *x,
		Dtype* y) {
	caffe_cpu_scale<Dtype>(N, alpha, x, y);
}

INSTANTIATE_CLASS(CPUMathBackend);

}  // namespace caffe
