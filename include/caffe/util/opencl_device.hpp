// Copyright 2014 BVLC and contributors.

#ifdef USE_OPENCL
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

#include <vector>

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

const char* clGetErrorString(cl_int error);
const char* clblasGetErrorString(clblasStatus status);

class CaffeOpenCL {
 public:
  inline static CaffeOpenCL& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new CaffeOpenCL());
    }
    return *singleton_;
  }

  virtual ~CaffeOpenCL() {
  }

  void SetDevice(const int device_id);
  inline static cl_context context() {
    if (Get().cl_context_ == NULL) {
      Get().create_context();
    }
    return Get().cl_context_;
  }
  inline static cl_command_queue queue() {
    if (Get().cl_command_queue_ == NULL) {
      Get().create_queue();
    }
    return Get().cl_command_queue_;
  }
 protected:
  cl_device_type get_device_type();
  cl_device_id current_cl_device_id();
  void create_context();
  void release_context();
  void create_queue();
  void release_queue();
  void initialize_clblas();
  void finalize_clblas();
 protected:
  static shared_ptr<CaffeOpenCL> singleton_;

  int current_device_id_;
  cl_platform_id current_cl_platform_id_;
  cl_int current_platform_device_count_;
  std::vector<cl_device_id> current_platform_device_ids_;
  int current_platform_device_id_;
  cl_context cl_context_;
  cl_command_queue cl_command_queue_;
  bool clblas_initialized_;
 private:
  CaffeOpenCL() :
    current_device_id_(0), current_cl_platform_id_(NULL),
    current_platform_device_count_(0), current_platform_device_id_(0),
    cl_context_(NULL), cl_command_queue_(NULL), clblas_initialized_(false) {
    initialize_clblas();
  }

  DISABLE_COPY_AND_ASSIGN(CaffeOpenCL);
};


template<typename Dtype>
class OpenCLDevice : public Device<Dtype> {
 public:
  OpenCLDevice() : Device<Dtype>() {
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

//  virtual void add_scalar(const int N, const Dtype alpha, Dtype *X);

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

//  virtual void powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

//  virtual void rng_uniform(const int N, const Dtype a, const Dtype b, Dtype* r);
//
//  virtual void rng_gaussian(const int N, const Dtype mu, const Dtype sigma,
//                            Dtype* r);
//
//  virtual void rng_bernoulli(const int N, const Dtype p, int* r);

//  virtual void exp(const int N, const Dtype* a, Dtype* y);

//  virtual void dot(const int N, const Dtype* x, const Dtype* y, Dtype* out);
//
//  virtual void hamming_distance(const int N, const Dtype* x, const Dtype* y,
//                                uint32_t* out);

// Returns the sum of the absolute values of the elements of vector x
//  virtual void asum(const int N, const Dtype* x, Dtype* y);

//  virtual void sign(const int N, const Dtype* x, Dtype* y);

//  virtual void sgnbit(const int N, const Dtype* x, Dtype* y);

//  virtual void fabs(const int N, const Dtype* x, Dtype* y);

//  virtual void scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y);

//  virtual void im2col(const Dtype* data_im, const int channels,
//      const int height, const int width, const int ksize, const int pad,
//      const int stride, Dtype* data_col);
//
//  virtual void col2im(const Dtype* data_col, const int channels,
//      const int height, const int width, const int psize, const int pad,
//      const int stride, Dtype* data_im);
};


}  // namespace caffe

#endif  // CAFFE_UTIL_OPENCL_DEVICE_H_
#endif  // USE_OPENCL

