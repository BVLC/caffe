#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

#ifdef USE_CUDA

template<typename Dtype>
cuda_dev_ptr<Dtype,  typename const_enable_if<Dtype>::type>
                                                 ::cuda_dev_ptr(Dtype* raw_ptr)
    : raw_ptr_(raw_ptr) { }
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                 ::cuda_dev_ptr(Dtype* raw_ptr)
    : raw_ptr_(raw_ptr) { }


template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                    ::cuda_dev_ptr(const dev_ptr<bool> &other) {
  const cuda_dev_ptr<bool>& cast_other =
      dynamic_cast<const cuda_dev_ptr<bool>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                    ::cuda_dev_ptr(const dev_ptr<bool> &other) {
  const cuda_dev_ptr<bool>& cast_other =
      dynamic_cast<const cuda_dev_ptr<bool>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                    ::cuda_dev_ptr(const dev_ptr<char> &other) {
  const cuda_dev_ptr<char>& cast_other =
      dynamic_cast<const cuda_dev_ptr<char>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                    ::cuda_dev_ptr(const dev_ptr<char> &other) {
  const cuda_dev_ptr<char>& cast_other =
      dynamic_cast<const cuda_dev_ptr<char>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::cuda_dev_ptr(const dev_ptr<int8_t> &other) {
  const cuda_dev_ptr<int8_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int8_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                  ::cuda_dev_ptr(const dev_ptr<int8_t> &other) {
  const cuda_dev_ptr<int8_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int8_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<int16_t> &other) {
  const cuda_dev_ptr<int16_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int16_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<int16_t> &other) {
  const cuda_dev_ptr<int16_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int16_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<int32_t> &other) {
  const cuda_dev_ptr<int32_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int32_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<int32_t> &other) {
  const cuda_dev_ptr<int32_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int32_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<int64_t> &other) {
  const cuda_dev_ptr<int64_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int64_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<int64_t> &other) {
  const cuda_dev_ptr<int64_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<int64_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<uint8_t> &other) {
  const cuda_dev_ptr<uint8_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint8_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<uint8_t> &other) {
  const cuda_dev_ptr<uint8_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint8_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                ::cuda_dev_ptr(const dev_ptr<uint16_t> &other) {
  const cuda_dev_ptr<uint16_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint16_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                ::cuda_dev_ptr(const dev_ptr<uint16_t> &other) {
  const cuda_dev_ptr<uint16_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint16_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                ::cuda_dev_ptr(const dev_ptr<uint32_t> &other) {
  const cuda_dev_ptr<uint32_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint32_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                ::cuda_dev_ptr(const dev_ptr<uint32_t> &other) {
  const cuda_dev_ptr<uint32_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint32_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                ::cuda_dev_ptr(const dev_ptr<uint64_t> &other) {
  const cuda_dev_ptr<uint64_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint64_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                ::cuda_dev_ptr(const dev_ptr<uint64_t> &other) {
  const cuda_dev_ptr<uint64_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<uint64_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<half_fp> &other) {
  const cuda_dev_ptr<half_fp>& cast_other =
      dynamic_cast<const cuda_dev_ptr<half_fp>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::cuda_dev_ptr(const dev_ptr<half_fp> &other) {
  const cuda_dev_ptr<half_fp>& cast_other =
      dynamic_cast<const cuda_dev_ptr<half_fp>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                   ::cuda_dev_ptr(const dev_ptr<float> &other) {
  const cuda_dev_ptr<float>& cast_other =
      dynamic_cast<const cuda_dev_ptr<float>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                   ::cuda_dev_ptr(const dev_ptr<float> &other) {
  const cuda_dev_ptr<float>& cast_other =
      dynamic_cast<const cuda_dev_ptr<float>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::cuda_dev_ptr(const dev_ptr<double> &other) {
  const cuda_dev_ptr<double>& cast_other =
      dynamic_cast<const cuda_dev_ptr<double>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                  ::cuda_dev_ptr(const dev_ptr<double> &other) {
  const cuda_dev_ptr<double>& cast_other =
      dynamic_cast<const cuda_dev_ptr<double>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                    ::cuda_dev_ptr(const dev_ptr<void> &other) {
  const cuda_dev_ptr<void>& cast_other =
      dynamic_cast<const cuda_dev_ptr<void>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}
template<typename Dtype>
cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                    ::cuda_dev_ptr(const dev_ptr<void> &other) {
  const cuda_dev_ptr<void>& cast_other =
      dynamic_cast<const cuda_dev_ptr<void>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}


template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                              ::cuda_dev_ptr(const dev_ptr<const bool> &other) {
  const cuda_dev_ptr<const bool>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const bool>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                              ::cuda_dev_ptr(const dev_ptr<const char> &other) {
  const cuda_dev_ptr<const char>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const char>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                            ::cuda_dev_ptr(const dev_ptr<const int8_t> &other) {
  const cuda_dev_ptr<const int8_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const int8_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::cuda_dev_ptr(const dev_ptr<const int16_t> &other) {
  const cuda_dev_ptr<const int16_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const int16_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::cuda_dev_ptr(const dev_ptr<const int32_t> &other) {
  const cuda_dev_ptr<const int32_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const int32_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::cuda_dev_ptr(const dev_ptr<const int64_t> &other) {
  const cuda_dev_ptr<const int64_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const int64_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::cuda_dev_ptr(const dev_ptr<const uint8_t> &other) {
  const cuda_dev_ptr<const uint8_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const uint8_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                          ::cuda_dev_ptr(const dev_ptr<const uint16_t> &other) {
  const cuda_dev_ptr<const uint16_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const uint16_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                          ::cuda_dev_ptr(const dev_ptr<const uint32_t> &other) {
  const cuda_dev_ptr<const uint32_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const uint32_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                          ::cuda_dev_ptr(const dev_ptr<const uint64_t> &other) {
  const cuda_dev_ptr<const uint64_t>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const uint64_t>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::cuda_dev_ptr(const dev_ptr<const half_fp> &other) {
  const cuda_dev_ptr<const half_fp>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const half_fp>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                             ::cuda_dev_ptr(const dev_ptr<const float> &other) {
  const cuda_dev_ptr<const float>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const float>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                            ::cuda_dev_ptr(const dev_ptr<const double> &other) {
  const cuda_dev_ptr<const double>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const double>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}

template<typename Dtype>
cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                              ::cuda_dev_ptr(const dev_ptr<const void> &other) {
  const cuda_dev_ptr<const void>& cast_other =
      dynamic_cast<const cuda_dev_ptr<const void>&>(other);
  raw_ptr_ = reinterpret_cast<Dtype*>(cast_other.get());
}


template<typename Dtype>
shared_ptr<dev_ptr<bool> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(bool* dummy) {
  return make_shared<cuda_dev_ptr<bool> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<char> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(char* dummy) {
  return make_shared<cuda_dev_ptr<char> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int8_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int8_t* dummy) {
  return make_shared<cuda_dev_ptr<int8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int16_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int16_t* dummy) {
  return make_shared<cuda_dev_ptr<int16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int32_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int32_t* dummy) {
  return make_shared<cuda_dev_ptr<int32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int64_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int64_t* dummy) {
  return make_shared<cuda_dev_ptr<int64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint8_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint8_t* dummy) {
  return make_shared<cuda_dev_ptr<uint8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint16_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint16_t* dummy) {
  return make_shared<cuda_dev_ptr<uint16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint32_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint32_t* dummy) {
  return make_shared<cuda_dev_ptr<uint32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint64_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint64_t* dummy) {
  return make_shared<cuda_dev_ptr<uint64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<half_fp> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(
      half_fp* dummy) {
  return make_shared<cuda_dev_ptr<half_fp> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<float> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(float* dummy) {
  return make_shared<cuda_dev_ptr<float> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<double> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(double* dummy) {
  return make_shared<cuda_dev_ptr<double> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<void> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(void* dummy) {
  return make_shared<cuda_dev_ptr<void> >(*this);
}


template<typename Dtype>
shared_ptr<dev_ptr<const bool> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const bool* dummy) {
  return make_shared<cuda_dev_ptr<const bool> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const bool> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const bool* dummy) {
  return make_shared<cuda_dev_ptr<const bool> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const char> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const char* dummy) {
  return make_shared<cuda_dev_ptr<const char> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const char> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const char* dummy) {
  return make_shared<cuda_dev_ptr<const char> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int8_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int8_t* dummy) {
  return make_shared<cuda_dev_ptr<const int8_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int8_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int8_t* dummy) {
  return make_shared<cuda_dev_ptr<const int8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int16_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int16_t* dummy) {
  return make_shared<cuda_dev_ptr<const int16_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int16_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int16_t* dummy) {
  return make_shared<cuda_dev_ptr<const int16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int32_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int32_t* dummy) {
  return make_shared<cuda_dev_ptr<const int32_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int32_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int32_t* dummy) {
  return make_shared<cuda_dev_ptr<const int32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int64_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int64_t* dummy) {
  return make_shared<cuda_dev_ptr<const int64_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int64_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int64_t* dummy) {
  return make_shared<cuda_dev_ptr<const int64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint8_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint8_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint8_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint8_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint8_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint16_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint16_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint16_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint16_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint16_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint32_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint32_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint32_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint32_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint32_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint64_t> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint64_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint64_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint64_t> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint64_t* dummy) {
  return make_shared<cuda_dev_ptr<const uint64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const half_fp> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(
      const half_fp* dummy) {
  return make_shared<cuda_dev_ptr<const half_fp> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const half_fp> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(
      const half_fp* dummy) {
  return make_shared<cuda_dev_ptr<const half_fp> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const float> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const float* dummy) {
  return make_shared<cuda_dev_ptr<const float> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const float> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const float* dummy) {
  return make_shared<cuda_dev_ptr<const float> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const double> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const double* dummy) {
  return make_shared<cuda_dev_ptr<const double> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const double> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const double* dummy) {
  return make_shared<cuda_dev_ptr<const double> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const void> > cuda_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const void* dummy) {
  return make_shared<cuda_dev_ptr<const void> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const void> > cuda_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const void* dummy) {
  return make_shared<cuda_dev_ptr<const void> >(*this);
}


template<typename Dtype>
void cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                                 ::increment() {
  ++raw_ptr_;
}
template<typename Dtype>
void cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                                 ::increment() {
  ++raw_ptr_;
}
template<typename Dtype>
void cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                                 ::decrement() {
  --raw_ptr_;
}
template<typename Dtype>
void cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                                 ::decrement() {
  --raw_ptr_;
}

template<typename Dtype>
void cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                       ::increment(int_tp val) {
  raw_ptr_ += val;
}
template<typename Dtype>
void cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                       ::increment(int_tp val) {
  raw_ptr_ += val;
}
template<typename Dtype>
void cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                       ::decrement(int_tp val) {
  raw_ptr_ -= val;
}
template<typename Dtype>
void cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                       ::decrement(int_tp val) {
  raw_ptr_ -= val;
}

template<>
void cuda_dev_ptr<void>::increment() { }
template<>
void cuda_dev_ptr<const void>::increment() { }
template<>
void cuda_dev_ptr<void>::decrement() { }
template<>
void cuda_dev_ptr<const void>::decrement() { }

template<>
void cuda_dev_ptr<void>::increment(int_tp val) { }
template<>
void cuda_dev_ptr<const void>::increment(int_tp val) { }
template<>
void cuda_dev_ptr<void>::decrement(int_tp val) { }
template<>
void cuda_dev_ptr<const void>::decrement(int_tp val) { }

template<typename Dtype>
Dtype*  cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                         ::get() const {
  return raw_ptr_;
}
template<typename Dtype>
Dtype*  cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                         ::get() const {
  return raw_ptr_;
}

template<typename Dtype>
void* cuda_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                             ::get_ptr() {
  return reinterpret_cast<void*>(&raw_ptr_);
}
template<typename Dtype>
void* cuda_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                             ::get_ptr() {
  return reinterpret_cast<void*>(&raw_ptr_);
}


INSTANTIATE_POINTER_CLASS(cuda_dev_ptr)

#endif  // USE_CUDA

}  // namespace caffe

