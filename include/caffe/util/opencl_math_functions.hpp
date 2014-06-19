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

#define CLBLAS_TRAILING_ARGS \
    num_command_queues, &queue, num_events_in_wait_list, \
    event_wait_list, &events

#define OPENCL_UNARY_KERNEL(Dtype_str, name_str, operation_str) \
"template<typename Dtype>           \n" \
"__kernel void " name_str "(        \n" \
"  __global " Dtype_str "* x,       \n" \
"  __global " Dtype_str "* y,       \n" \
"  const unsigned int count) {      \n" \
"  for (int i = get_global_id(0);   \n" \
"       i < (count);                \n" \
"       i += get_global_size(0)) {  \n" \
"       " operation_str ";              \n" \
"}                                  \n" \
"\n";

#define OPENCL_BINARY_KERNEL(Dtype_str, name_str, operation_str) \
"__kernel void " name_str "(        \n" \
"  __global " Dtype_str "* a,       \n" \
"  __global " Dtype_str "* b,       \n" \
"  __global " Dtype_str "* y,       \n" \
"  const unsigned int count) {      \n" \
"  for (int i = get_global_id(0);   \n" \
"       i < (count);                \n" \
"       i += get_global_size(0)) {  \n" \
"       " operation_str ";              \n" \
"}                                  \n" \
"\n"

// local_size: Number of work items in each local work group
// global_size: Number of total work items
#define DEFINE_LOCAL_AND_GLOBAL_SIZE(n) \
  const size_t local_size = 64; \
  const size_t global_size = (n + local_size - 1) \
    / local_size


// https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/
#define DEFINE_OPENCL_UNARY_FUNC(Dtype, name, operation) \
template <> \
void caffe_opencl_##name<Dtype>(const int n, const Dtype *x, Dtype *y) { \
  const char* kernel_source = OPENCL_UNARY_KERNEL(#Dtype, #name, \
                                                  #operation); \
  cl_context context = CaffeOpenCL::context(); \
  cl_command_queue queue = CaffeOpenCL::queue(); \
  cl_int error; \
  const size_t bytes = n * sizeof(Dtype); \
  cl_program program = clCreateProgramWithSource( \
    context, 1, (const char **) & kernel_source, NULL, &error); \
  CL_CHECK(error); \
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL); \
  cl_kernel kernel = clCreateKernel(program, #name, &error); \
  CL_CHECK(error); \
  cl_mem d_x = clCreateBuffer(context, \
                              CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
                              bytes, \
                              const_cast<void*>(static_cast<const void*>(x)), \
                              &error); \
  cl_mem d_y = clCreateBuffer(context, \
                              CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, \
                              bytes, static_cast<void*>(y), &error); \
  void* mapped_x = clEnqueueMapBuffer( \
    queue, d_x, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &error); \
  CL_CHECK(error); \
  CL_CHECK(clEnqueueUnmapMemObject( \
    CaffeOpenCL::queue(), d_x, mapped_x, \
    0, NULL, NULL)); \
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x)); \
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y)); \
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(unsigned int), &n)); \
  DEFINE_LOCAL_AND_GLOBAL_SIZE(n); \
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, \
                                  &local_size, 0, NULL, NULL)); \
  CL_CHECK(clFinish(queue)); \
  CL_CHECK(clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, \
                               bytes, y, 0, NULL, NULL )); \
  void* mapped_y = clEnqueueMapBuffer( \
   queue, d_y, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &error); \
  CL_CHECK(error); \
  CL_CHECK(clEnqueueUnmapMemObject( \
    CaffeOpenCL::queue(), d_y, mapped_y, \
    0, NULL, NULL)); \
  CL_CHECK(clReleaseMemObject(d_x)); \
  CL_CHECK(clReleaseMemObject(d_y)); \
  CL_CHECK(clReleaseProgram(program)); \
  CL_CHECK(clReleaseKernel(kernel)); \
}

#define DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(name, operation) \
    DEFINE_OPENCL_UNARY_FUNC(float, name, operation) \
    DEFINE_OPENCL_UNARY_FUNC(double, name, operation)

#define DEFINE_OPENCL_BINARY_FUNC(Dtype, name, operation) \
template <> \
void caffe_opencl_##name<Dtype>(const int n, const Dtype *a, const Dtype *b, \
                         Dtype *y) { \
  const char* kernel_source = OPENCL_BINARY_KERNEL(#Dtype, #name, \
                                                   #operation); \
  cl_context context = CaffeOpenCL::context(); \
  cl_command_queue queue = CaffeOpenCL::queue(); \
  cl_int error; \
  const size_t bytes = n * sizeof(Dtype); \
  cl_program program = clCreateProgramWithSource( \
    context, 1, (const char **) & kernel_source, NULL, &error); \
  CL_CHECK(error); \
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL); \
  cl_kernel kernel = clCreateKernel(program, #name, &error); \
  CL_CHECK(error); \
  cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, \
                              const_cast<void*>(static_cast<const void*>(a)),\
                              &error); \
  cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, \
                              const_cast<void*>(static_cast<const void*>(b)), \
                              &error); \
  cl_mem d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, \
                              static_cast<void*>(y), &error); \
  void* mapped_a = clEnqueueMapBuffer( \
    queue, d_a, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &error); \
  CL_CHECK(error); \
  CL_CHECK(clEnqueueUnmapMemObject( \
    CaffeOpenCL::queue(), d_a, mapped_a, \
    0, NULL, NULL)); \
  void* mapped_b = clEnqueueMapBuffer( \
    queue, d_b, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &error); \
  CL_CHECK(error); \
  CL_CHECK(clEnqueueUnmapMemObject( \
    CaffeOpenCL::queue(), d_b, mapped_b, \
    0, NULL, NULL)); \
  CL_CHECK(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, \
                                bytes, a, 0, NULL, NULL)); \
  CL_CHECK(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, \
                                bytes, b, 0, NULL, NULL)); \
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a)); \
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b)); \
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_y)); \
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(unsigned int), &n)); \
  DEFINE_LOCAL_AND_GLOBAL_SIZE(n); \
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, \
                                  &local_size, 0, NULL, NULL)); \
  CL_CHECK(clFinish(queue)); \
  void* mapped_y = clEnqueueMapBuffer( \
   queue, d_y, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &error); \
  CL_CHECK(error); \
  CL_CHECK(clEnqueueUnmapMemObject( \
    CaffeOpenCL::queue(), d_y, mapped_y, \
    0, NULL, NULL)); \
  CL_CHECK(clReleaseMemObject(d_a)); \
  CL_CHECK(clReleaseMemObject(d_b)); \
  CL_CHECK(clReleaseMemObject(d_y)); \
  CL_CHECK(clReleaseProgram(program)); \
  CL_CHECK(clReleaseKernel(kernel)); \
}

#define DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(name, operation) \
    DEFINE_OPENCL_BINARY_FUNC(float, name, operation) \
    DEFINE_OPENCL_BINARY_FUNC(double, name, operation)

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
    const CBLAS_TRANSPOSE TransB, const int M, const int n, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);


template <typename Dtype>
void caffe_opencl_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int n,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_opencl_axpy(const int n, const Dtype alpha, const Dtype* x,
    Dtype* y);

template <typename Dtype>
void caffe_opencl_axpby(const int n, const Dtype alpha, const Dtype* x,
    const Dtype beta, Dtype* y);

template <typename Dtype>
void caffe_opencl_copy(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void caffe_opencl_set(const int n, const Dtype alpha, Dtype *x);

template <typename Dtype>
void caffe_opencl_add_scalar(const int n, const Dtype alpha, Dtype *x);

template <typename Dtype>
void caffe_opencl_scal(const int n, const Dtype alpha, Dtype *x);

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
void caffe_opencl_copy_from_cpu(const int n, const Dtype *x, Dtype *y);

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
void caffe_opencl_add(const int n, const Dtype* a,
                      const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_opencl_sub(const int n, const Dtype* a,
                      const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_opencl_mul(const int n, const Dtype* a,
                      const Dtype* b, Dtype* y);

template<typename Dtype>
void caffe_opencl_div(const int n, const Dtype* a,
                      const Dtype* b, Dtype* y);
}  // namespace caffe


#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
