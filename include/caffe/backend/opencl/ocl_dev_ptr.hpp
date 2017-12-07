#ifndef CAFFE_BACKEND_OPENCL_OCL_DEV_PTR_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_DEV_PTR_HPP_

#include "caffe/backend/backend.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/util/half_fp.hpp"

namespace caffe {

#ifdef USE_OPENCL

template<typename Dtype, typename = void>
class ocl_dev_ptr : public dev_ptr<Dtype> { };


template<typename Dtype>
class ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
            : public dev_ptr<Dtype, typename const_enable_if<Dtype>::type> {
 public:
  explicit ocl_dev_ptr(cl_mem ocl_mem);
  explicit ocl_dev_ptr(cl_mem ocl_mem, uint_tp off);

  ocl_dev_ptr(const dev_ptr<bool> &other);
  ocl_dev_ptr(const dev_ptr<char> &other);
  ocl_dev_ptr(const dev_ptr<int8_t> &other);
  ocl_dev_ptr(const dev_ptr<int16_t> &other);
  ocl_dev_ptr(const dev_ptr<int32_t> &other);
  ocl_dev_ptr(const dev_ptr<int64_t> &other);
  ocl_dev_ptr(const dev_ptr<uint8_t> &other);
  ocl_dev_ptr(const dev_ptr<uint16_t> &other);
  ocl_dev_ptr(const dev_ptr<uint32_t> &other);
  ocl_dev_ptr(const dev_ptr<uint64_t> &other);
  ocl_dev_ptr(const dev_ptr<half_fp> &other);
  ocl_dev_ptr(const dev_ptr<float> &other);
  ocl_dev_ptr(const dev_ptr<double> &other);
  ocl_dev_ptr(const dev_ptr<void> &other);

  ocl_dev_ptr(const dev_ptr<const bool> &other);
  ocl_dev_ptr(const dev_ptr<const char> &other);
  ocl_dev_ptr(const dev_ptr<const int8_t> &other);
  ocl_dev_ptr(const dev_ptr<const int16_t> &other);
  ocl_dev_ptr(const dev_ptr<const int32_t> &other);
  ocl_dev_ptr(const dev_ptr<const int64_t> &other);
  ocl_dev_ptr(const dev_ptr<const uint8_t> &other);
  ocl_dev_ptr(const dev_ptr<const uint16_t> &other);
  ocl_dev_ptr(const dev_ptr<const uint32_t> &other);
  ocl_dev_ptr(const dev_ptr<const uint64_t> &other);
  ocl_dev_ptr(const dev_ptr<const half_fp> &other);
  ocl_dev_ptr(const dev_ptr<const float> &other);
  ocl_dev_ptr(const dev_ptr<const double> &other);
  ocl_dev_ptr(const dev_ptr<const void> &other);

  virtual shared_ptr<dev_ptr<const bool> > instance(const bool* dummy);
  virtual shared_ptr<dev_ptr<const char> > instance(const char* dummy);
  virtual shared_ptr<dev_ptr<const int8_t> > instance(const int8_t* dummy);
  virtual shared_ptr<dev_ptr<const int16_t> >
                                    instance(const int16_t* dummy);
  virtual shared_ptr<dev_ptr<const int32_t> >
                                    instance(const int32_t* dummy);
  virtual shared_ptr<dev_ptr<const int64_t> >
                                    instance(const int64_t* dummy);
  virtual shared_ptr<dev_ptr<const uint8_t> >
                                    instance(const uint8_t* dummy);
  virtual shared_ptr<dev_ptr<const uint16_t> >
                                    instance(const uint16_t* dummy);
  virtual shared_ptr<dev_ptr<const uint32_t> >
                                    instance(const uint32_t* dummy);
  virtual shared_ptr<dev_ptr<const uint64_t> >
                                    instance(const uint64_t* dummy);
  virtual shared_ptr<dev_ptr<const half_fp> >
                                    instance(const half_fp* dummy);
  virtual shared_ptr<dev_ptr<const float> > instance(const float* dummy);
  virtual shared_ptr<dev_ptr<const double> > instance(const double* dummy);
  virtual shared_ptr<dev_ptr<const void> > instance(const void* dummy);

  cl_mem get_ocl_mem() const;
  uint_tp get_off() const;

  virtual void increment();
  virtual void decrement();
  virtual void increment(int_tp val);
  virtual void decrement(int_tp val);

 private:
  cl_mem ocl_mem_;
  uint_tp off_;
};


template<typename Dtype>
class ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
            : public dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type> {
 public:
  explicit ocl_dev_ptr(cl_mem ocl_mem);
  explicit ocl_dev_ptr(cl_mem ocl_mem, uint_tp off);

  ocl_dev_ptr(const dev_ptr<bool> &other);
  ocl_dev_ptr(const dev_ptr<char> &other);
  ocl_dev_ptr(const dev_ptr<int8_t> &other);
  ocl_dev_ptr(const dev_ptr<int16_t> &other);
  ocl_dev_ptr(const dev_ptr<int32_t> &other);
  ocl_dev_ptr(const dev_ptr<int64_t> &other);
  ocl_dev_ptr(const dev_ptr<uint8_t> &other);
  ocl_dev_ptr(const dev_ptr<uint16_t> &other);
  ocl_dev_ptr(const dev_ptr<uint32_t> &other);
  ocl_dev_ptr(const dev_ptr<uint64_t> &other);
  ocl_dev_ptr(const dev_ptr<half_fp> &other);
  ocl_dev_ptr(const dev_ptr<float> &other);
  ocl_dev_ptr(const dev_ptr<double> &other);
  ocl_dev_ptr(const dev_ptr<void> &other);

  virtual shared_ptr<dev_ptr<bool> > instance(bool* dummy);
  virtual shared_ptr<dev_ptr<char> > instance(char* dummy);
  virtual shared_ptr<dev_ptr<int8_t> > instance(int8_t* dummy);
  virtual shared_ptr<dev_ptr<int16_t> > instance(int16_t* dummy);
  virtual shared_ptr<dev_ptr<int32_t> > instance(int32_t* dummy);
  virtual shared_ptr<dev_ptr<int64_t> > instance(int64_t* dummy);
  virtual shared_ptr<dev_ptr<uint8_t> > instance(uint8_t* dummy);
  virtual shared_ptr<dev_ptr<uint16_t> > instance(uint16_t* dummy);
  virtual shared_ptr<dev_ptr<uint32_t> > instance(uint32_t* dummy);
  virtual shared_ptr<dev_ptr<uint64_t> > instance(uint64_t* dummy);
  virtual shared_ptr<dev_ptr<half_fp> > instance(half_fp* dummy);
  virtual shared_ptr<dev_ptr<float> > instance(float* dummy);
  virtual shared_ptr<dev_ptr<double> > instance(double* dummy);
  virtual shared_ptr<dev_ptr<void> > instance(void* dummy);

  virtual shared_ptr<dev_ptr<const bool> > instance(const bool* dummy);
  virtual shared_ptr<dev_ptr<const char> > instance(const char* dummy);
  virtual shared_ptr<dev_ptr<const int8_t> > instance(const int8_t* dummy);
  virtual shared_ptr<dev_ptr<const int16_t> >
                                    instance(const int16_t* dummy);
  virtual shared_ptr<dev_ptr<const int32_t> >
                                    instance(const int32_t* dummy);
  virtual shared_ptr<dev_ptr<const int64_t> >
                                    instance(const int64_t* dummy);
  virtual shared_ptr<dev_ptr<const uint8_t> >
                                    instance(const uint8_t* dummy);
  virtual shared_ptr<dev_ptr<const uint16_t> >
                                    instance(const uint16_t* dummy);
  virtual shared_ptr<dev_ptr<const uint32_t> >
                                    instance(const uint32_t* dummy);
  virtual shared_ptr<dev_ptr<const uint64_t> >
                                    instance(const uint64_t* dummy);
  virtual shared_ptr<dev_ptr<const half_fp> >
                                    instance(const half_fp* dummy);
  virtual shared_ptr<dev_ptr<const float> > instance(const float* dummy);
  virtual shared_ptr<dev_ptr<const double> > instance(const double* dummy);
  virtual shared_ptr<dev_ptr<const void> > instance(const void* dummy);

  cl_mem get_ocl_mem() const;
  uint_tp get_off() const;

  virtual void increment();
  virtual void decrement();
  virtual void increment(int_tp val);
  virtual void decrement(int_tp val);

 private:
  cl_mem ocl_mem_;
  uint_tp off_;
};

#endif  // USE_OPENCL

}  // namespace caffe

#endif  // CAFFE_BACKEND_OPENCL_OCL_DEV_PTR_HPP_
