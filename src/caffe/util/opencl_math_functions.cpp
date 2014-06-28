// Copyright 2014 BVLC and contributors.

#ifdef USE_OPENCL
//#include "caffe/common.hpp"
#include "caffe/util/opencl_math_functions.hpp"

namespace caffe {

template <>
void caffe_opencl_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int ldA = (TransA == CblasNoTrans) ? K : M;
  int ldB = (TransB == CblasNoTrans) ? N : K;
  int ldC = N;
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  clblasTranspose clTransB = to_clblasTranspose(TransB);
  CREATE_CL_MEM(A, M, K, READ_ONLY);
  CREATE_CL_MEM(B, K, N, READ_ONLY);
  CREATE_CL_MEM(C, M, N, READ_WRITE);
  ENQUEUE_CL_BUFFER(Write, A, M, K);
  ENQUEUE_CL_BUFFER(Write, B, K, N);
  ENQUEUE_CL_BUFFER(Write, C, M, N);
  PRE_CLBLAS_CALL;
  // bufX is defined by the macro CREATE_CL_MEM(X, ...)
  CLBLAS_CHECK(clblasSgemm(clblasRowMajor, clTransA, clTransB,
      M, N, K, alpha, ARRAY(A), ARRAY(B), beta, ARRAY(C),
      CLBLAS_TRAILING_ARGS));
  /* Release OpenCL memory objects. */
  RELEASE_CL_MEM(C);
  RELEASE_CL_MEM(B);
  RELEASE_CL_MEM(A);
}

template <>
void caffe_opencl_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int ldA = (TransA == CblasNoTrans) ? K : M;
  int ldB = (TransB == CblasNoTrans) ? N : K;
  int ldC = N;
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  clblasTranspose clTransB = to_clblasTranspose(TransB);
  CREATE_CL_MEM(A, M, K, READ_ONLY);
  CREATE_CL_MEM(B, K, N, READ_ONLY);
  CREATE_CL_MEM(C, M, N, READ_WRITE);
  ENQUEUE_CL_BUFFER(Write, A, M, K);
  ENQUEUE_CL_BUFFER(Write, B, K, N);
  ENQUEUE_CL_BUFFER(Write, C, M, N);
  PRE_CLBLAS_CALL;
  // bufX is defined by the macro CREATE_CL_MEM(X, ...)
  CLBLAS_CHECK(clblasDgemm(clblasRowMajor, clTransA, clTransB,
      M, N, K, alpha, ARRAY(A), ARRAY(B), beta, ARRAY(C),
      CLBLAS_TRAILING_ARGS));
  /* Release OpenCL memory objects. */
  RELEASE_CL_MEM(C);
  RELEASE_CL_MEM(B);
  RELEASE_CL_MEM(A);
}

template <>
void caffe_opencl_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  int ldA = (TransA == CblasNoTrans) ? N : M;
  int ldx = N;
  int ldy = N;
  CREATE_CL_MEM(A, M, N, READ_ONLY);
  CREATE_CL_MEM(x, N, 1, READ_ONLY);
  CREATE_CL_MEM(y, M, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSgemv(clblasRowMajor, clTransA, M, N, alpha,
      ARRAY(A), ARRAY(x), beta, ARRAY(y),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  clblasTranspose clTransA = to_clblasTranspose(TransA);
  int ldA = (TransA == CblasNoTrans) ? N : M;
  int ldx = N;
  int ldy = N;
  CREATE_CL_MEM(A, M, N, READ_ONLY);
  CREATE_CL_MEM(x, N, 1, READ_ONLY);
  CREATE_CL_MEM(y, M, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDgemv(clblasRowMajor, clTransA, M, N, alpha,
      ARRAY(A), ARRAY(x), beta, ARRAY(y),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSaxpy(
      N, alpha, ARRAY(X), ARRAY(Y),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDaxpy(
      N, alpha, ARRAY(X), ARRAY(Y),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_copy<float>(const int N, const float* X, float* Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasScopy(
      N, ARRAY(X), ARRAY(Y),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_copy<double>(const int N, const double* X, double* Y) {
  int ldX = N;
  int ldY = N;
  CREATE_CL_MEM(X, N, 1, READ_ONLY);
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDcopy(
      N, ARRAY(X), ARRAY(Y),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_scal<float>(const int N, const float alpha, float *X) {
  int ldX = N;
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasSscal(
      N, alpha, ARRAY(X),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_scal<double>(const int N, const double alpha, double *X) {
  int ldX = N;
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  PRE_CLBLAS_CALL;
  CLBLAS_CHECK(clblasDscal(
      N, alpha, ARRAY(X),
      CLBLAS_TRAILING_ARGS));
}

template <>
void caffe_opencl_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_opencl_scal(N, beta, Y);
  caffe_opencl_axpy(N, alpha, X, Y);
}

template <>
void caffe_opencl_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_opencl_scal(N, beta, Y);
  caffe_opencl_axpy(N, alpha, X, Y);
}

template<typename Dtype>
void caffe_opencl_copy_from_cpu(const int N, const Dtype *X, Dtype *Y) {
  CREATE_CL_MEM(Y, N, 1, READ_WRITE);
  cl_bool blocking_write = CL_TRUE;
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event events = NULL;
  CL_CHECK(clEnqueueWriteBuffer(
      CaffeOpenCL::queue(), bufY, blocking_write, 0, N * sizeof(Dtype),
      X, num_events_in_wait_list, event_wait_list, &events));
}

template
void caffe_opencl_copy_from_cpu<float>(const int N, const float *X, float *Y);
template
void caffe_opencl_copy_from_cpu<double>(const int N, const double *X, double *Y);


template <typename Dtype>
void caffe_opencl_set(const int N, const Dtype alpha, Dtype *X) {
#ifdef CL_VERSION_1_2
  CREATE_CL_MEM(X, N, 1, READ_WRITE);
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event event = NULL;
  CL_CHECK(clEnqueueFillBuffer(
      CaffeOpenCL::queue(), bufX, static_cast<void*>(&alpha), sizeof(Dtype),
      0, sizeof(Dtype) * N, num_events_in_wait_list, event_wait_list, &event));
#else
  std::vector<Dtype> tmp(N, alpha);
  caffe_opencl_copy_from_cpu(N, &tmp[0], X);
#endif
}

template
void caffe_opencl_set(const int N, const float alpha, float *X);
template
void caffe_opencl_set(const int N, const double alpha, double *X);


DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sqr, y[i] = x[i] * x[i]);
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(exp, y[i] = exp(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sign, y[i] = sign<Dtype>(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sgnbit, y[i] = signbit(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(fabs, y[i] = fabs(x[i]));

DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(add, y[i] = a[i] + b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(sub, y[i] = a[i] - b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(mul, y[i] = a[i] * b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(div, y[i] = a[i] / b[i]);

}  // namespace caffe

#endif  // USE_OPENCL
