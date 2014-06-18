// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_OPENCL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_OPENCL_MATH_FUNCTIONS_H_

#include "caffe/util/opencl_device.hpp"

namespace caffe {

#define CREATE_CL_MEM(A, M, K, FLAG) \
  cl_mem buf##A; \
  do { \
    cl_int error; \
    buf##A = clCreateBuffer( \
      CaffeOpenCL::context(), CL_MEM_##FLAG, M * K * sizeof(*A), \
      NULL, &error); \
    CL_CHECK(error); \
  } while(0)

#define RELEASE_CL_MEM(A) clReleaseMemObject(buf##A)

#define ENQUEUE_CL_BUFFER(FLAG, A, M, K) \
  CL_CHECK(clEnqueue##FLAG##Buffer( \
    CaffeOpenCL::queue(), buf##A, CL_TRUE, 0, M * K * sizeof(*A), \
    A, 0, NULL, NULL));

#define PRE_CLBLAS_CALL \
  cl_uint num_command_queues = 1; \
  cl_uint num_events_in_wait_list = 0; \
  cl_event *event_wait_list = NULL; \
  cl_event events = NULL; \
  cl_command_queue queue = CaffeOpenCL::queue();

#define ARRAY(A) buf##A, 0, ld##A

#define CLBALS_TRAILING_ARGS \
    num_command_queues, &queue, num_events_in_wait_list, \
    event_wait_list, &events

inline clblasTranspose to_clblasTranspose(const CBLAS_TRANSPOSE trans) {
  switch (trans) {
  case CblasNoTrans:
    return clblasNoTrans;
  case CblasTrans:
    return clblasTrans;
  case CblasConjTrans:
    return clblasConjTrans;
  default:
    LOG(FATAL) << "Unknown CBLAS_TRANSPOSE " << trans;
  }
}

template <typename Dtype>
void caffe_opencl_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);


template <typename Dtype>
void caffe_opencl_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_opencl_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_opencl_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_opencl_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_opencl_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_opencl_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_opencl_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
Dtype caffe_opencl_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
int caffe_opencl_hamming_distance(const int n, const Dtype* x, const Dtype* y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_opencl_asum(const int n, const Dtype* x);

template <typename Dtype>
void caffe_opencl_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

template<typename Dtype>
void caffe_opencl_copy_from_cpu(const int N, const Dtype *X, Dtype *Y);

template<typename Dtype>
void caffe_opencl_sqr(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_opencl_exp(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_opencl_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_opencl_sgnbit(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_opencl_fabs(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_opencl_add(const int N, const Dtype* a,
                      const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_opencl_sub(const int N, const Dtype* a,
                      const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_opencl_mul(const int N, const Dtype* a,
                      const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_opencl_div(const int N, const Dtype* a,
                      const Dtype* b, Dtype* y);
}  // namespace caffe


#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
