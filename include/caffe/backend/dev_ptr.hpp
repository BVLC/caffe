#ifndef CAFFE_BACKEND_DEV_PTR_HPP_
#define CAFFE_BACKEND_DEV_PTR_HPP_

#include "caffe/definitions.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/util/half_fp.hpp"

namespace caffe {

template<typename Dtype, typename = void>
class dev_ptr { };

template<typename Dtype>
class dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type> {
 public:
  virtual shared_ptr<dev_ptr<bool> > instance(bool* dummy) = 0;
  virtual shared_ptr<dev_ptr<char> > instance(char* dummy) = 0;
  virtual shared_ptr<dev_ptr<int8_t> > instance(int8_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<int16_t> > instance(int16_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<int32_t> > instance(int32_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<int64_t> > instance(int64_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<uint8_t> > instance(uint8_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<uint16_t> > instance(uint16_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<uint32_t> > instance(uint32_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<uint64_t> > instance(uint64_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<half_fp> >
                                          instance(half_fp* dummy) = 0;
  virtual shared_ptr<dev_ptr<float> > instance(float* dummy) = 0;
  virtual shared_ptr<dev_ptr<double> > instance(double* dummy) = 0;
  virtual shared_ptr<dev_ptr<void> > instance(void* dummy) = 0;

  virtual shared_ptr<dev_ptr<const bool> > instance(const bool* dummy) = 0;
  virtual shared_ptr<dev_ptr<const char> > instance(const char* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int8_t> > instance(const int8_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int16_t> >
                                    instance(const int16_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int32_t> >
                                    instance(const int32_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int64_t> >
                                    instance(const int64_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint8_t> >
                                    instance(const uint8_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint16_t> >
                                    instance(const uint16_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint32_t> >
                                    instance(const uint32_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint64_t> >
                                    instance(const uint64_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const half_fp> >
                                    instance(const half_fp* dummy) = 0;
  virtual shared_ptr<dev_ptr<const float> > instance(const float* dummy) = 0;
  virtual shared_ptr<dev_ptr<const double> > instance(const double* dummy) = 0;
  virtual shared_ptr<dev_ptr<const void> > instance(const void* dummy) = 0;

  virtual void increment() = 0;
  virtual void decrement() = 0;
  virtual void increment(int_tp val) = 0;
  virtual void decrement(int_tp val) = 0;
};

template<typename Dtype>
class dev_ptr<Dtype, typename const_enable_if<Dtype>::type> {
 public:
  virtual shared_ptr<dev_ptr<const bool> > instance(const bool* dummy) = 0;
  virtual shared_ptr<dev_ptr<const char> > instance(const char* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int8_t> > instance(const int8_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int16_t> >
                                    instance(const int16_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int32_t> >
                                    instance(const int32_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const int64_t> >
                                    instance(const int64_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint8_t> >
                                    instance(const uint8_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint16_t> >
                                    instance(const uint16_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint32_t> >
                                    instance(const uint32_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const uint64_t> >
                                    instance(const uint64_t* dummy) = 0;
  virtual shared_ptr<dev_ptr<const half_fp> >
                                    instance(const half_fp* dummy) = 0;
  virtual shared_ptr<dev_ptr<const float> > instance(const float* dummy) = 0;
  virtual shared_ptr<dev_ptr<const double> > instance(const double* dummy) = 0;
  virtual shared_ptr<dev_ptr<const void> > instance(const void* dummy) = 0;

  virtual void increment() = 0;
  virtual void decrement() = 0;
  virtual void increment(int_tp val) = 0;
  virtual void decrement(int_tp val) = 0;
};


}  // namespace caffe

#endif  // CAFFE_BACKEND_DEV_PTR_HPP_
