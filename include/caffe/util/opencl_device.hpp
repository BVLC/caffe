// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_OPENCL_DEVICE_H_
#define CAFFE_UTIL_OPENCL_DEVICE_H_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include "clBLAS.h"

#include "glog/logging.h"

#include "caffe/util/device.hpp"

namespace caffe {

#define CL_CHECK(condition) \
    /* Code block avoids redefinition of cudaError_t error */ \
    do { \
      cl_int error = condition; \
      CHECK_EQ(error, CL_SUCCESS) << " " << clGetErrorString(error); \
    } while (0)

#define CLBLAS_CHECK(condition) \
  do { \
    clblasStatus status = condition; \
    CHECK_EQ(status, clblasSuccess) << " " \
      << caffe::clblasGetErrorString(status); \
  } while (0)

#define CREATE_CL_MEM(A, M, K, FLAG) \
  cl_mem buf##A; \
  do { \
    cl_int error; \
    buf##A = clCreateBuffer( \
      OpenCLDevice::context(), CL_MEM_##FLAG, M * K * sizeof(*A), \
      NULL, &error); \
    CL_CHECK(error); \
  } while(0)

#define RELEASE_CL_MEM(A) clReleaseMemObject(buf##A)

#define ENQUEUE_CL_BUFFER(FLAG, A, M, K) \
  CL_CHECK(clEnqueue##FLAG##Buffer( \
    OpenCLDevice::queue(), bufA, CL_TRUE, 0, M * K * sizeof(*A), \
    A, 0, NULL, NULL));

#define PRE_CLBLAS_CALL \
  cl_uint num_command_queues = 1; \
  cl_uint num_events_in_wait_list = 0; \
  cl_event *event_wait_list = NULL; \
  cl_event events = NULL; \
  cl_command_queue queue = OpenCLDevice::queue();

#define ARRAY(A) buf##A, 0, ld##A

#define CLBALS_TRAILING_ARGS \
    num_command_queues, &queue, num_events_in_wait_list, \
    event_wait_list, &events

const char* clGetErrorString(cl_int error);
const char* clblasGetErrorString(clblasStatus status);

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

// OpenCL: grid stride looping
#define OPENCL_KERNEL_LOOP(i, n) \
  for (int i = get_global_id(0); \
       i < (n); \
       i += get_global_size(0))

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

template<typename Dtype>
class OpenCLDevice : public Device<Dtype> {
 public:
  OpenCLDevice() {
  }
  virtual ~OpenCLDevice() {
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

//  virtual void sqr(const int N, const Dtype* a, Dtype* y);
//
//  virtual void add(const int N, const Dtype* a, const Dtype* b, Dtype* y);
//
//  virtual void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);
//
//  virtual void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
//
//  virtual void div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

  virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

  virtual void rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* r);

  virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
                            Dtype* r);

  virtual void rng_bernoulli(const int N, const Dtype p, int* r);

//  virtual void exp(const int N, const Dtype* a, Dtype* y);

  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out);

  virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
                                uint32_t* out);

// Returns the sum of the absolute values of the elements of vector x
  virtual void asum(const int N, const Dtype* x, Dtype* y);

//  virtual void sign(const int N, const Dtype* x, Dtype* y);

//  virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

//  virtual void fabs(const int N, const Dtype* x, Dtype* y);

  virtual void scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y);

  virtual void im2col(const Dtype* data_im, const int channels,
      const int height, const int width, const int ksize, const int pad,
      const int stride, Dtype* data_col);

  virtual void col2im(const Dtype* data_col, const int channels,
      const int height, const int width, const int psize, const int pad,
      const int stride, Dtype* data_im);

  static cl_context context();
  static cl_command_queue queue();
 private:
  static cl_device_id cl_device_id_;
  static cl_context cl_context_;
  static cl_command_queue cl_command_queue_;
  static bool cl_context_created_;
  static bool cl_command_queue_created_;
};


}  // namespace caffe

#endif  // CAFFE_UTIL_OPENCL_DEVICE_H_
