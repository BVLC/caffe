#ifndef CAFFE_UTIL_CL_KERNEL_H_
#define CAFFE_UTIL_CL_KERNEL_H_

#include <CL/cl.h>
#include <utility>

namespace caffe {

const char* clGetErrorString(cl_int);

class ClState;

class ClKernel {
 public:
  ClKernel() : kernel_(NULL) { }
  explicit ClKernel(cl_kernel kernel) : kernel_(kernel) { }

  operator cl_kernel() const { return kernel_; }

  template<typename T>
  void set_arg(cl_uint index, T value) {
    cl_int err = clSetKernelArg(kernel_, index, sizeof value, &value);
    CHECK_EQ(err, CL_SUCCESS) << " " << clGetErrorString(err);
  }

  template<typename T>
  void set_arg(cl_uint index, T* ptr) {
    set_arg(index, static_cast<const T*>(ptr));
  }

  template<typename T>
  void set_arg(cl_uint index, const T* ptr) {
    cl_int err;
    if (ptr != NULL) {
      cl_mem mem = get_buffer_from_ptr(ptr);
      err = clSetKernelArg(kernel_, index, sizeof mem, &mem);
    } else {
      err = clSetKernelArg(kernel_, index, sizeof (cl_mem), NULL);
    }
    CHECK_EQ(err, CL_SUCCESS) << " " << clGetErrorString(err);
  }

  template<typename T>
  void set_arg_ptr_off(cl_uint index, const T* ptr) {
    if (ptr != NULL) {
      std::pair<cl_mem, int> buf = ClKernel::get_buffer_offset_from_ptr(ptr);
      buf.second = buf.second / sizeof (T);
      set_arg_mem(index, buf.first);
      set_arg(index + 1, buf.second);
    } else {
      cl_int err = clSetKernelArg(kernel_, index, sizeof (cl_mem), NULL);
      CHECK_EQ(err, CL_SUCCESS) << " " << clGetErrorString(err);
      set_arg(index + 1, 0);
    }
  }

  void set_arg_mem(cl_uint index, cl_mem mem) {
    cl_int err = clSetKernelArg(kernel_, index, sizeof mem, &mem);
    CHECK_EQ(err, CL_SUCCESS) << " " << clGetErrorString(err);
  }

  void set_arg_mem_local(cl_uint index, size_t size) {
    cl_int err = clSetKernelArg(kernel_, index, size, NULL);
    CHECK_EQ(err, CL_SUCCESS) << " " << clGetErrorString(err);
  }

  void enqueue(const size_t size);
  void enqueue_blocking(const size_t size);
  void enqueue_params(const size_t size, const int N);
  template<typename T>
  void enqueue_params(const size_t size, const int N, const T scalar,
    T* dstBuffer) {
    set_arg(0, N);
    set_arg(1, scalar);
    set_arg_ptr_off(2, dstBuffer);
    enqueue_blocking(size);
  }
  template<typename T>
  void enqueue_params(const size_t size, const int N, const T* srcBuffer,
    T* dstBuffer) {
    set_arg(0, N);
    set_arg_ptr_off(1, srcBuffer);
    set_arg_ptr_off(3, dstBuffer);
    enqueue_blocking(size);
  }
  template<typename T>
  void enqueue_params(const size_t size, const int N, const T* src1Buffer,
    const T* src2Buffer, T* dstBuffer) {
    set_arg(0, N);
    set_arg_ptr_off(1, src1Buffer);
    set_arg_ptr_off(3, src2Buffer);
    set_arg_ptr_off(5, dstBuffer);
    enqueue_blocking(size);
  }

 private:
  cl_mem get_buffer_from_ptr(const void* ptr);
  std::pair<cl_mem, int> get_buffer_offset_from_ptr(const void* ptr);

  // cl_kernel is not owned by this class, it just wraps it for convenience
  //   functions
  cl_kernel kernel_;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_CL_KERNEL_H_
