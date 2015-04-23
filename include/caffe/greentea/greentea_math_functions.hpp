/*
 * greentea_math_functions.hpp
 *
 *  Created on: Apr 6, 2015
 *      Author: fabian
 */

#ifndef GREENTEA_MATH_FUNCTIONS_HPP_
#define GREENTEA_MATH_FUNCTIONS_HPP_

#include "caffe/greentea/greentea.hpp"
#include "caffe/util/math_functions.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

void greentea_gpu_memcpy(const size_t N, const cl_mem X, void *Y, viennacl::ocl::context &ctx);

void greentea_gpu_memcpy(const size_t N, const void* X, cl_mem Y, viennacl::ocl::context &ctx);


template<typename Dtype>
void greentea_copy(const int N, const cl_mem X, cl_mem Y, viennacl::ocl::context &ctx);

template<typename Dtype>
void greentea_gpu_gemm(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                              const CBLAS_TRANSPOSE TransB, const int M,
                              const int N, const int K, const Dtype alpha,
                              const cl_mem A, const int offA, const cl_mem B, const int offB, const Dtype beta,
                              cl_mem C, const int offC);

template<typename Dtype>
void greentea_gpu_gemv(const int ctx_id, const CBLAS_TRANSPOSE TransA, const int M, const int N,
                       const Dtype alpha, const cl_mem A, const int offA,
                       const cl_mem x, const int offx, const Dtype beta,
                       cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_axpy(const int ctx_id, const int N, const Dtype alpha, const cl_mem X,
                              const int offX, cl_mem Y, const int offY);


/*
 template <typename Dtype>
 void greentea_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
 Dtype* Y);

 template <typename Dtype>
 void greentea_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
 const Dtype beta, Dtype* Y);



 template <typename Dtype>
 void greentea_gpu_set(const int N, const Dtype alpha, Dtype *X);

 inline void greentea_gpu_memset(const size_t N, const int alpha, void* X) {
 /*  viennacl::m
 #ifndef CPU_ONLY
 CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
 #else
 NO_GPU;
 #endif*/
/*}

 template <typename Dtype>
 void greentea_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

 template <typename Dtype>
 void greentea_gpu_scal(const int N, const Dtype alpha, Dtype *X);

 template <typename Dtype>
 void greentea_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_abs(const int n, const Dtype* a, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_exp(const int n, const Dtype* a, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

 void greentea_gpu_rng_uniform(const int n, unsigned int* r);

 template <typename Dtype>
 void greentea_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

 template <typename Dtype>
 void greentea_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
 Dtype* r);

 template <typename Dtype>
 void greentea_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

 template <typename Dtype>
 void greentea_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

 template <typename Dtype>
 uint32_t greentea_gpu_hamming_distance(const int n, const Dtype* x,
 const Dtype* y);

 template <typename Dtype>
 void greentea_gpu_asum(const int n, const Dtype* x, Dtype* y);

 template<typename Dtype>
 void greentea_gpu_sign(const int n, const Dtype* x, Dtype* y);

 template<typename Dtype>
 void greentea_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_fabs(const int n, const Dtype* x, Dtype* y);

 template <typename Dtype>
 void greentea_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);*/

}

#endif /* GREENTEA_MATH_FUNCTIONS_HPP_ */
