#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"
#include "caffe/util/type_utils.hpp"

namespace caffe {

#ifdef USE_OPENCL

template<typename Dtype, typename Otype>
inline uint_tp compute_off(const ocl_dev_ptr<Otype> &other) {
  uint_tp osize = safe_sizeof<Otype>();
  uint_tp dsize = safe_sizeof<Dtype>();
  return (other.get_off() * osize) / dsize;
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                   ::ocl_dev_ptr(cl_mem ocl_mem)
    : ocl_mem_(ocl_mem), off_(0) { }
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                   ::ocl_dev_ptr(cl_mem ocl_mem)
    : ocl_mem_(ocl_mem), off_(0) { }

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                      ::ocl_dev_ptr(cl_mem ocl_mem, uint_tp off)
    : ocl_mem_(ocl_mem), off_(off) { }
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                      ::ocl_dev_ptr(cl_mem ocl_mem, uint_tp off)
    : ocl_mem_(ocl_mem), off_(off) { }


template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                     ::ocl_dev_ptr(const dev_ptr<bool> &other) {
  const ocl_dev_ptr<bool>& cast_other =
      dynamic_cast<const ocl_dev_ptr<bool>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, bool>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                     ::ocl_dev_ptr(const dev_ptr<bool> &other) {
  const ocl_dev_ptr<bool>& cast_other =
      dynamic_cast<const ocl_dev_ptr<bool>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, bool>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                     ::ocl_dev_ptr(const dev_ptr<char> &other) {
  const ocl_dev_ptr<char>& cast_other =
      dynamic_cast<const ocl_dev_ptr<char>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, char>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                     ::ocl_dev_ptr(const dev_ptr<char> &other) {
  const ocl_dev_ptr<char>& cast_other =
      dynamic_cast<const ocl_dev_ptr<char>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, char>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                   ::ocl_dev_ptr(const dev_ptr<int8_t> &other) {
  const ocl_dev_ptr<int8_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int8_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int8_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                   ::ocl_dev_ptr(const dev_ptr<int8_t> &other) {
  const ocl_dev_ptr<int8_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int8_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int8_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<int16_t> &other) {
  const ocl_dev_ptr<int16_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int16_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int16_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<int16_t> &other) {
  const ocl_dev_ptr<int16_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int16_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int16_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<int32_t> &other) {
  const ocl_dev_ptr<int32_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int32_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int32_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<int32_t> &other) {
  const ocl_dev_ptr<int32_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int32_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int32_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<int64_t> &other) {
  const ocl_dev_ptr<int64_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int64_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int64_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<int64_t> &other) {
  const ocl_dev_ptr<int64_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<int64_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, int64_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<uint8_t> &other) {
  const ocl_dev_ptr<uint8_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint8_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint8_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                  ::ocl_dev_ptr(const dev_ptr<uint8_t> &other) {
  const ocl_dev_ptr<uint8_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint8_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint8_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::ocl_dev_ptr(const dev_ptr<uint16_t> &other) {
  const ocl_dev_ptr<uint16_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint16_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint16_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::ocl_dev_ptr(const dev_ptr<uint16_t> &other) {
  const ocl_dev_ptr<uint16_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint16_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint16_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::ocl_dev_ptr(const dev_ptr<uint32_t> &other) {
  const ocl_dev_ptr<uint32_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint32_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint32_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::ocl_dev_ptr(const dev_ptr<uint32_t> &other) {
  const ocl_dev_ptr<uint32_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint32_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint32_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::ocl_dev_ptr(const dev_ptr<uint64_t> &other) {
  const ocl_dev_ptr<uint64_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint64_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint64_t>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                 ::ocl_dev_ptr(const dev_ptr<uint64_t> &other) {
  const ocl_dev_ptr<uint64_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<uint64_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, uint64_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                         ::ocl_dev_ptr(const dev_ptr<half_fp> &other) {
  const ocl_dev_ptr<half_fp>& cast_other =
      dynamic_cast<const ocl_dev_ptr<half_fp>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, half_fp>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                         ::ocl_dev_ptr(const dev_ptr<half_fp> &other) {
  const ocl_dev_ptr<half_fp>& cast_other =
      dynamic_cast<const ocl_dev_ptr<half_fp>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, half_fp>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                    ::ocl_dev_ptr(const dev_ptr<float> &other) {
  const ocl_dev_ptr<float>& cast_other =
      dynamic_cast<const ocl_dev_ptr<float>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, float>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                    ::ocl_dev_ptr(const dev_ptr<float> &other) {
  const ocl_dev_ptr<float>& cast_other =
      dynamic_cast<const ocl_dev_ptr<float>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, float>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                   ::ocl_dev_ptr(const dev_ptr<double> &other) {
  const ocl_dev_ptr<double>& cast_other =
      dynamic_cast<const ocl_dev_ptr<double>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, double>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                   ::ocl_dev_ptr(const dev_ptr<double> &other) {
  const ocl_dev_ptr<double>& cast_other =
      dynamic_cast<const ocl_dev_ptr<double>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, double>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                     ::ocl_dev_ptr(const dev_ptr<void> &other) {
  const ocl_dev_ptr<void>& cast_other =
      dynamic_cast<const ocl_dev_ptr<void>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, void>(cast_other);
}
template<typename Dtype>
ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                     ::ocl_dev_ptr(const dev_ptr<void> &other) {
  const ocl_dev_ptr<void>& cast_other =
      dynamic_cast<const ocl_dev_ptr<void>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, void>(cast_other);
}


template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                               ::ocl_dev_ptr(const dev_ptr<const bool> &other) {
  const ocl_dev_ptr<const bool>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const bool>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const bool>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                               ::ocl_dev_ptr(const dev_ptr<const char> &other) {
  const ocl_dev_ptr<const bool>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const char>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const char>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                             ::ocl_dev_ptr(const dev_ptr<const int8_t> &other) {
  const ocl_dev_ptr<const int8_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const int8_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const int8_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                            ::ocl_dev_ptr(const dev_ptr<const int16_t> &other) {
  const ocl_dev_ptr<const int16_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const int16_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const int16_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                            ::ocl_dev_ptr(const dev_ptr<const int32_t> &other) {
  const ocl_dev_ptr<const int32_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const int32_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const int32_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                            ::ocl_dev_ptr(const dev_ptr<const int64_t> &other) {
  const ocl_dev_ptr<const int64_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const int64_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const int64_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                            ::ocl_dev_ptr(const dev_ptr<const uint8_t> &other) {
  const ocl_dev_ptr<const uint8_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const uint8_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const uint8_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::ocl_dev_ptr(const dev_ptr<const uint16_t> &other) {
  const ocl_dev_ptr<const uint16_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const uint16_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const uint16_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::ocl_dev_ptr(const dev_ptr<const uint32_t> &other) {
  const ocl_dev_ptr<const uint32_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const uint32_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const uint32_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                           ::ocl_dev_ptr(const dev_ptr<const uint64_t> &other) {
  const ocl_dev_ptr<const uint64_t>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const uint64_t>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const uint64_t>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                   ::ocl_dev_ptr(const dev_ptr<const half_fp> &other) {
  const ocl_dev_ptr<const half_fp>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const half_fp>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const half_fp>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                              ::ocl_dev_ptr(const dev_ptr<const float> &other) {
  const ocl_dev_ptr<const float>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const float>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const float>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                             ::ocl_dev_ptr(const dev_ptr<const double> &other) {
  const ocl_dev_ptr<const double>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const double>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const double>(cast_other);
}

template<typename Dtype>
ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                               ::ocl_dev_ptr(const dev_ptr<const void> &other) {
  const ocl_dev_ptr<const void>& cast_other =
      dynamic_cast<const ocl_dev_ptr<const void>&>(other);
  ocl_mem_ = cast_other.get_ocl_mem();
  off_ = compute_off<Dtype, const void>(cast_other);
}


template<typename Dtype>
shared_ptr<dev_ptr<bool> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(bool* dummy) {
  return make_shared<ocl_dev_ptr<bool> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<char> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(char* dummy) {
  return make_shared<ocl_dev_ptr<char> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int8_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int8_t* dummy) {
  return make_shared<ocl_dev_ptr<int8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int16_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int16_t* dummy) {
  return make_shared<ocl_dev_ptr<int16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int32_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int32_t* dummy) {
  return make_shared<ocl_dev_ptr<int32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<int64_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(int64_t* dummy) {
  return make_shared<ocl_dev_ptr<int64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint8_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint8_t* dummy) {
  return make_shared<ocl_dev_ptr<uint8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint16_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint16_t* dummy) {
  return make_shared<ocl_dev_ptr<uint16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint32_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint32_t* dummy) {
  return make_shared<ocl_dev_ptr<uint32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<uint64_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(uint64_t* dummy) {
  return make_shared<ocl_dev_ptr<uint64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<half_fp> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(
      half_fp* dummy) {
  return make_shared<ocl_dev_ptr<half_fp> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<float> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(float* dummy) {
  return make_shared<ocl_dev_ptr<float> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<double> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(double* dummy) {
  return make_shared<ocl_dev_ptr<double> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<void> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(void* dummy) {
  return make_shared<ocl_dev_ptr<void> >(*this);
}


template<typename Dtype>
shared_ptr<dev_ptr<const bool> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const bool* dummy) {
  return make_shared<ocl_dev_ptr<const bool> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const bool> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const bool* dummy) {
  return make_shared<ocl_dev_ptr<const bool> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const char> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const char* dummy) {
  return make_shared<ocl_dev_ptr<const char> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const char> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const char* dummy) {
  return make_shared<ocl_dev_ptr<const char> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int8_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int8_t* dummy) {
  return make_shared<ocl_dev_ptr<const int8_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int8_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int8_t* dummy) {
  return make_shared<ocl_dev_ptr<const int8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int16_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int16_t* dummy) {
  return make_shared<ocl_dev_ptr<const int16_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int16_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int16_t* dummy) {
  return make_shared<ocl_dev_ptr<const int16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int32_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int32_t* dummy) {
  return make_shared<ocl_dev_ptr<const int32_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int32_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int32_t* dummy) {
  return make_shared<ocl_dev_ptr<const int32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const int64_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const int64_t* dummy) {
  return make_shared<ocl_dev_ptr<const int64_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const int64_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const int64_t* dummy) {
  return make_shared<ocl_dev_ptr<const int64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint8_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint8_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint8_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint8_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint8_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint8_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint16_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint16_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint16_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint16_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint16_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint16_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint32_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint32_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint32_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint32_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint32_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint32_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const uint64_t> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const uint64_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint64_t> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const uint64_t> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const uint64_t* dummy) {
  return make_shared<ocl_dev_ptr<const uint64_t> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const half_fp> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(
      const half_fp* dummy) {
  return make_shared<ocl_dev_ptr<const half_fp> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const half_fp> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(
      const half_fp* dummy) {
  return make_shared<ocl_dev_ptr<const half_fp> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const float> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const float* dummy) {
  return make_shared<ocl_dev_ptr<const float> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const float> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const float* dummy) {
  return make_shared<ocl_dev_ptr<const float> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const double> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const double* dummy) {
  return make_shared<ocl_dev_ptr<const double> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const double> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const double* dummy) {
  return make_shared<ocl_dev_ptr<const double> >(*this);
}

template<typename Dtype>
shared_ptr<dev_ptr<const void> > ocl_dev_ptr<Dtype,
  typename const_enable_if<Dtype>::type>::instance(const void* dummy) {
  return make_shared<ocl_dev_ptr<const void> >(*this);
}
template<typename Dtype>
shared_ptr<dev_ptr<const void> > ocl_dev_ptr<Dtype,
  typename non_const_enable_if<Dtype>::type>::instance(const void* dummy) {
  return make_shared<ocl_dev_ptr<const void> >(*this);
}


template<typename Dtype>
void ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                                 ::increment() {
  ++off_;
}
template<typename Dtype>
void ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                                 ::increment() {
  ++off_;
}
template<typename Dtype>
void ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                                 ::decrement() {
  --off_;
}
template<typename Dtype>
void ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                                 ::decrement() {
  --off_;
}

template<typename Dtype>
void ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                       ::increment(int_tp val) {
  off_ += val;
}
template<typename Dtype>
void ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                       ::increment(int_tp val) {
  off_ += val;
}

template<typename Dtype>
void ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                       ::decrement(int_tp val) {
  off_ -= val;
}
template<typename Dtype>
void ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                       ::decrement(int_tp val) {
  off_ -= val;
}

template<typename Dtype>
cl_mem  ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                         ::get_ocl_mem() const {
  return ocl_mem_;
}
template<typename Dtype>
cl_mem  ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                         ::get_ocl_mem() const {
  return ocl_mem_;
}

template<typename Dtype>
uint_tp ocl_dev_ptr<Dtype, typename const_enable_if<Dtype>::type>
                                                             ::get_off() const {
  return off_;
}
template<typename Dtype>
uint_tp ocl_dev_ptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                             ::get_off() const {
  return off_;
}


INSTANTIATE_POINTER_CLASS(ocl_dev_ptr)

#endif  // USE_OPENCL

}  // namespace caffe

