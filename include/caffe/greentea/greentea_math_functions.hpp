/*
 * greentea_math_functions.hpp
 *
 *  Created on: Apr 6, 2015
 *      Author: fabian
 */

#ifndef GREENTEA_MATH_FUNCTIONS_HPP_
#define GREENTEA_MATH_FUNCTIONS_HPP_
#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/math_functions.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

void greentea_memset(const int ctx_id, const size_t N, const int alpha,
                     cl_mem X, const int offX);

void greentea_gpu_memcpy(const size_t N, const cl_mem X, const int offX,
                         void *Y, viennacl::ocl::context &ctx);

void greentea_gpu_memcpy(const size_t N, const void* X, cl_mem Y,
                         const int offY, viennacl::ocl::context &ctx);

void greentea_gpu_memcpy(const size_t N, const cl_mem X, const int offX,
                         cl_mem Y, const int offY, viennacl::ocl::context &ctx);

template<typename Dtype>
void greentea_copy(const int N, const cl_mem X, cl_mem Y,
                   viennacl::ocl::context &ctx);

template<typename Dtype>
void greentea_gpu_gemm(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const Dtype alpha, const cl_mem A,
                       const int offA, const cl_mem B, const int offB,
                       const Dtype beta, cl_mem C, const int offC);

template<typename Dtype>
void greentea_gpu_gemv(const int ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int M, const int N, const Dtype alpha,
                       const cl_mem A, const int offA, const cl_mem x,
                       const int offx, const Dtype beta, cl_mem y,
                       const int offy);

template<typename Dtype>
void greentea_gpu_axpy(const int ctx_id, const int N, const Dtype alpha,
                       const cl_mem x, const int offx, cl_mem y,
                       const int offy);

template<typename Dtype>
void greentea_gpu_mul(const int ctx_id, const int N, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy);

template<typename Dtype>
void greentea_gpu_scal(const int ctx_id, const int N, const Dtype alpha,
                       cl_mem x, int offx);

template<typename Dtype>
void greentea_gpu_axpby(const int ctx_id, const int N, const Dtype alpha,
                        const cl_mem X, const int offX, const Dtype beta,
                        cl_mem Y, const int offY);

template<typename Dtype>
void greentea_gpu_dot(const int ctx_id, const int n, const cl_mem X,
                      const int offX, const cl_mem Y, const int offY,
                      Dtype* out);

template<typename Dtype>
void greentea_gpu_asum(const int ctx_id, const int n, const cl_mem X,
                       const int offX, Dtype* Y);

template<typename Dtype>
void greentea_gpu_scale(const int ctx_id, const int n, const Dtype alpha,
                        const cl_mem X, const int offX, cl_mem Y,
                        const int offY);

template<typename Dtype>
void greentea_gpu_set(const int ctx_id, const int N, const Dtype alpha,
                      cl_mem Y, const int offY);

template<typename Dtype>
void greentea_gpu_add_scalar(const int ctx_id, const int N, const Dtype alpha,
                             cl_mem Y, const int offY);

template<typename Dtype>
void greentea_gpu_add(const int ctx_id, const int n, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy);

template<typename Dtype>
void greentea_gpu_sub(const int ctx_id, const int n, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy);

template<typename Dtype>
void greentea_gpu_div(const int ctx_id, const int N, const cl_mem a,
                      const int offa, const cl_mem b, const int offb, cl_mem y,
                      const int offy);

template<typename Dtype>
void greentea_gpu_abs(const int ctx_id, const int N, const cl_mem a,
                      const int offa, cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_exp(const int ctx_id, const int N, const cl_mem a,
                      const int offa, cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_powx(const int ctx_id, const int N, const cl_mem a,
                       const int offa, const Dtype alpha, cl_mem y,
                       const int offy);

template<typename Dtype>
void greentea_gpu_sign(const int ctx_id, const int n, const cl_mem x, int offx,
                       cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_sgnbit(const int ctx_id, const int n, const cl_mem x,
                         int offx, cl_mem y, const int offy);

template<typename Dtype>
void greentea_gpu_rng_uniform(const int ctx_id, const int n, const Dtype a,
                              const Dtype b, cl_mem r, const int offr);

template<typename Dtype>
void greentea_gpu_rng_gaussian(const int ctx_id, const int n, const Dtype mu,
                               const Dtype sigma, cl_mem r, const int offr);

}

#endif // USE GREENTEA
#endif /* GREENTEA_MATH_FUNCTIONS_HPP_ */
