#ifndef CAFFE_BACKEND_VPTR_HPP_
#define CAFFE_BACKEND_VPTR_HPP_

#include "caffe/definitions.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/util/half_fp.hpp"

namespace caffe {

template<typename Dtype, typename = void>
class vptr {
};

template<typename Dtype>
class vptr<Dtype, typename const_enable_if<Dtype>::type> {
 public:
  explicit vptr();
  explicit vptr(shared_ptr<dev_ptr<Dtype> > ptr);

  vptr(const vptr<bool> &other);
  vptr(const vptr<char> &other);
  vptr(const vptr<int8_t> &other);
  vptr(const vptr<int16_t> &other);
  vptr(const vptr<int32_t> &other);
  vptr(const vptr<int64_t> &other);
  vptr(const vptr<uint8_t> &other);
  vptr(const vptr<uint16_t> &other);
  vptr(const vptr<uint32_t> &other);
  vptr(const vptr<uint64_t> &other);
  vptr(const vptr<half_fp> &other);
  vptr(const vptr<float> &other);
  vptr(const vptr<double> &other);
  vptr(const vptr<void> &other);

  vptr(const vptr<const bool> &other);
  vptr(const vptr<const char> &other);
  vptr(const vptr<const int8_t> &other);
  vptr(const vptr<const int16_t> &other);
  vptr(const vptr<const int32_t> &other);
  vptr(const vptr<const int64_t> &other);
  vptr(const vptr<const uint8_t> &other);
  vptr(const vptr<const uint16_t> &other);
  vptr(const vptr<const uint32_t> &other);
  vptr(const vptr<const uint64_t> &other);
  vptr(const vptr<const half_fp> &other);
  vptr(const vptr<const float> &other);
  vptr(const vptr<const double> &other);
  vptr(const vptr<const void> &other);

  shared_ptr<dev_ptr<Dtype> > get_dev_ptr() const;
  dev_ptr<Dtype>* get() const;

  // Convenience implementation access (CUDA)
#ifdef USE_CUDA
  Dtype* get_cuda_ptr() const;
  void* get_cuda_ptr_ptr() const;
#endif

  // Convenience implementation access (OpenCL)
#ifdef USE_OPENCL
  cl_mem get_ocl_mem() const;
  uint_tp get_ocl_off() const;
#endif

  vptr<Dtype>& operator=(const vptr<bool> &other);
  vptr<Dtype>& operator=(const vptr<char> &other);
  vptr<Dtype>& operator=(const vptr<int8_t> &other);
  vptr<Dtype>& operator=(const vptr<int16_t> &other);
  vptr<Dtype>& operator=(const vptr<int32_t> &other);
  vptr<Dtype>& operator=(const vptr<int64_t> &other);
  vptr<Dtype>& operator=(const vptr<uint8_t> &other);
  vptr<Dtype>& operator=(const vptr<uint16_t> &other);
  vptr<Dtype>& operator=(const vptr<uint32_t> &other);
  vptr<Dtype>& operator=(const vptr<uint64_t> &other);
  vptr<Dtype>& operator=(const vptr<half_fp> &other);
  vptr<Dtype>& operator=(const vptr<float> &other);
  vptr<Dtype>& operator=(const vptr<double> &other);
  vptr<Dtype>& operator=(const vptr<void> &other);

  vptr<Dtype>& operator=(const vptr<const bool> &other);
  vptr<Dtype>& operator=(const vptr<const char> &other);
  vptr<Dtype>& operator=(const vptr<const int8_t> &other);
  vptr<Dtype>& operator=(const vptr<const int16_t> &other);
  vptr<Dtype>& operator=(const vptr<const int32_t> &other);
  vptr<Dtype>& operator=(const vptr<const int64_t> &other);
  vptr<Dtype>& operator=(const vptr<const uint8_t> &other);
  vptr<Dtype>& operator=(const vptr<const uint16_t> &other);
  vptr<Dtype>& operator=(const vptr<const uint32_t> &other);
  vptr<Dtype>& operator=(const vptr<const uint64_t> &other);
  vptr<Dtype>& operator=(const vptr<const half_fp> &other);
  vptr<Dtype>& operator=(const vptr<const float> &other);
  vptr<Dtype>& operator=(const vptr<const double> &other);
  vptr<Dtype>& operator=(const vptr<const void> &other);

  vptr<Dtype> operator++();
  vptr<Dtype> operator--();
  vptr<Dtype> operator++(int val);
  vptr<Dtype> operator--(int val);
  vptr<Dtype> operator+(uint_tp val);
  vptr<Dtype> operator-(uint_tp val);
  vptr<Dtype> operator+=(uint_tp val);
  vptr<Dtype> operator-=(uint_tp val);

  bool is_valid() const;

private:
  static void* nullptr_ref_;
  shared_ptr<dev_ptr<Dtype> > dev_ptr_;
};

template<typename Dtype>
class vptr<Dtype, typename non_const_enable_if<Dtype>::type> {
 public:
  explicit vptr();
  explicit vptr(shared_ptr<dev_ptr<Dtype> > ptr);

  vptr(const vptr<bool> &other);
  vptr(const vptr<char> &other);
  vptr(const vptr<int8_t> &other);
  vptr(const vptr<int16_t> &other);
  vptr(const vptr<int32_t> &other);
  vptr(const vptr<int64_t> &other);
  vptr(const vptr<uint8_t> &other);
  vptr(const vptr<uint16_t> &other);
  vptr(const vptr<uint32_t> &other);
  vptr(const vptr<uint64_t> &other);
  vptr(const vptr<half_fp> &other);
  vptr(const vptr<float> &other);
  vptr(const vptr<double> &other);
  vptr(const vptr<void> &other);

  shared_ptr<dev_ptr<Dtype> > get_dev_ptr() const;
  dev_ptr<Dtype>* get() const;

  // Convenience implementation access (CUDA)
#ifdef USE_CUDA
  Dtype* get_cuda_ptr() const;
  void* get_cuda_ptr_ptr() const;
#endif

  // Convenience implementation access (OpenCL)
#ifdef USE_OPENCL
  cl_mem get_ocl_mem() const;
  uint_tp get_ocl_off() const;
#endif

  vptr<Dtype>& operator=(const vptr<bool> &other);
  vptr<Dtype>& operator=(const vptr<char> &other);
  vptr<Dtype>& operator=(const vptr<int8_t> &other);
  vptr<Dtype>& operator=(const vptr<int16_t> &other);
  vptr<Dtype>& operator=(const vptr<int32_t> &other);
  vptr<Dtype>& operator=(const vptr<int64_t> &other);
  vptr<Dtype>& operator=(const vptr<uint8_t> &other);
  vptr<Dtype>& operator=(const vptr<uint16_t> &other);
  vptr<Dtype>& operator=(const vptr<uint32_t> &other);
  vptr<Dtype>& operator=(const vptr<uint64_t> &other);
  vptr<Dtype>& operator=(const vptr<half_fp> &other);
  vptr<Dtype>& operator=(const vptr<float> &other);
  vptr<Dtype>& operator=(const vptr<double> &other);
  vptr<Dtype>& operator=(const vptr<void> &other);

  vptr<Dtype> operator++();
  vptr<Dtype> operator--();
  vptr<Dtype> operator++(int val);
  vptr<Dtype> operator--(int val);
  vptr<Dtype> operator+(uint_tp val);
  vptr<Dtype> operator-(uint_tp val);
  vptr<Dtype> operator+=(uint_tp val);
  vptr<Dtype> operator-=(uint_tp val);

  bool is_valid() const;

private:
  static void* nullptr_ref_;
  shared_ptr<dev_ptr<Dtype> > dev_ptr_;
};

}  // namespace caffe

#endif  // CAFFE_BACKEND_VPTR_HPP_
