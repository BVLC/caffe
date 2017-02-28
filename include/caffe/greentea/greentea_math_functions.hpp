/*
 * greentea_math_functions.hpp
 *
 *  Created on: Apr 6, 2015
 *      Author: fabian
 */

#ifndef GREENTEA_MATH_FUNCTIONS_HPP_
#define GREENTEA_MATH_FUNCTIONS_HPP_

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/math_functions.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

void greentea_memset(const int_tp ctx_id, const uint_tp N, const int_tp alpha,
                     cl_mem X, const int_tp offX);

void greentea_gpu_memcpy(const uint_tp N, const cl_mem X, const int_tp offX,
                         void *Y, viennacl::ocl::context *ctx);

void greentea_gpu_memcpy(const uint_tp N, const void* X, cl_mem Y,
                         const int_tp offY, viennacl::ocl::context *ctx);

void greentea_gpu_memcpy(const uint_tp N, const cl_mem X, const int_tp offX,
                         cl_mem Y, const int_tp offY,
                         viennacl::ocl::context *ctx);

template<typename Dtype>
void greentea_copy(const int_tp N, const cl_mem X, const int_tp offX, cl_mem Y,
                   const int_tp offY, viennacl::ocl::context *ctx);

template<typename Dtype>
void greentea_copy(const int_tp N, const cl_mem X, const int_tp offX, Dtype* Y,
                   viennacl::ocl::context *ctx);

template<typename Dtype>
void greentea_copy(const int_tp N, const Dtype* X, cl_mem Y, const int_tp offY,
                   viennacl::ocl::context *ctx);

template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC);

template<typename Dtype>
void greentea_gpu_gemv(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const int_tp M, const int_tp N, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem x,
                       const int_tp offx, const Dtype beta, cl_mem y,
                       const int_tp offy);

template<typename Dtype>
void greentea_gpu_axpy(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       const cl_mem x, const int_tp offx, cl_mem y,
                       const int_tp offy);

template<typename Dtype>
void greentea_gpu_mul(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_scal(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                       cl_mem x, int_tp offx);

template<typename Dtype>
void greentea_gpu_axpby(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                        const cl_mem X, const int_tp offX, const Dtype beta,
                        cl_mem Y, const int_tp offY);

template<typename Dtype>
void greentea_gpu_dot(const int_tp ctx_id, const int_tp n, const cl_mem X,
                      const int_tp offX, const cl_mem Y, const int_tp offY,
                      Dtype* out);

template<typename Dtype>
void greentea_gpu_asum(const int_tp ctx_id, const int_tp n, const cl_mem X,
                       const int_tp offX, Dtype* Y);

template<typename Dtype>
void greentea_gpu_scale(const int_tp ctx_id, const int_tp n, const Dtype alpha,
                        const cl_mem X, const int_tp offX, cl_mem Y,
                        const int_tp offY);

template<typename Dtype>
void greentea_gpu_set(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                      cl_mem Y, const int_tp offY);

template<typename Dtype>
void greentea_gpu_add_scalar(const int_tp ctx_id, const int_tp N,
                             const Dtype alpha, cl_mem Y, const int_tp offY);

template<typename Dtype>
void greentea_gpu_add(const int_tp ctx_id, const int_tp n, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sub(const int_tp ctx_id, const int_tp n, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_div(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, const cl_mem b, const int_tp offb,
                      cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_abs(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_exp(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_powx(const int_tp ctx_id, const int_tp N, const cl_mem a,
                       const int_tp offa, const Dtype alpha, cl_mem y,
                       const int_tp offy);

template<typename Dtype>
void greentea_gpu_log(const int_tp ctx_id, const int_tp N, const cl_mem a,
                      const int_tp offa, cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sign(const int_tp ctx_id, const int_tp n, const cl_mem x,
                       int_tp offx, cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_sgnbit(const int_tp ctx_id, const int_tp n, const cl_mem x,
int_tp offx,
                         cl_mem y, const int_tp offy);

template<typename Dtype>
void greentea_gpu_rng_uniform(const int_tp ctx_id, const int_tp n,
                              const Dtype a, const Dtype b, cl_mem r,
                              const int_tp offr);

void greentea_gpu_rng_uniform(const int_tp ctx_id, const int_tp n, cl_mem r,
int_tp offr);

template<typename Dtype>
void greentea_gpu_rng_gaussian(const int_tp ctx_id, const int_tp n,
                               const Dtype mu, const Dtype sigma, cl_mem r,
                               const int_tp offr);

}  // namespace caffe

#endif  // USE GREENTEA
#endif  /* GREENTEA_MATH_FUNCTIONS_HPP_ */
