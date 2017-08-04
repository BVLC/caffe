#include <memory>

#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

namespace caffe {

template<typename Mtype>
vptr<Mtype>::vptr()
  : dev_ptr_(std::shared_ptr<dev_ptr<Mtype>>(nullptr)) {

}

template<typename Mtype>
template<typename Otype>
vptr<Mtype>::vptr(const vptr<Otype> &other)
  : dev_ptr_(std::shared_ptr<dev_ptr<Mtype>>(nullptr)) {
  instance(other);
}

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<int8_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<uint8_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<int16_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<uint16_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<int32_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<uint32_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<int64_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<uint64_t> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<half_float::half> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<float> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<double> &other);

template<typename Mtype>
template<>
vptr<Mtype>::vptr(const vptr<void> &other);


template<typename Mtype>
dev_ptr<Mtype>* vptr<Mtype>::get() {
  return dev_ptr_.get();
}

#ifdef USE_CUDA
template<typename Mtype>
Mtype* vptr<Mtype>::get_cuda_ptr() {
  return dynamic_cast<cuda_dev_ptr<Mtype> >(dev_ptr_.get())->get_cuda_ptr();
}
#endif

#ifdef USE_OPENCL
template<typename Mtype>
cl_mem vptr<Mtype>::get_ocl_mem() {
  return dynamic_cast<ocl_dev_ptr<Mtype> >(dev_ptr_.get())->get_ocl_mem();
}
template<typename Mtype>
uint_tp vptr<Mtype>::get_ocl_off() {
  return dynamic_cast<ocl_dev_ptr<Mtype> >(dev_ptr_.get())->get_off();
}
#endif


template<typename Mtype>
template<typename Otype>
vptr<Mtype> vptr<Mtype>::instance(const vptr<Otype> &other) {
#ifdef USE_CUDA
  if (dynamic_cast<cuda_dev_ptr<Otype>>(other.get())) {
    dev_ptr_ = std::make_shared<dev_ptr<Mtype> >(
        cuda_dev_ptr<Mtype>(dynamic_cast<cuda_dev_ptr<Otype>>(other.get())));
  }
#endif
#ifdef USE_OPENCL
  if (dynamic_cast<ocl_dev_ptr<Otype>*>(other.get())) {
    dev_ptr_ = std::make_shared<dev_ptr<Mtype> >(
        cl_dev_ptr<Mtype>(dynamic_cast<ocl_dev_ptr<Otype>*>(other.get())));
  }
#endif
  return this;
}

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<int8_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<uint8_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<int16_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<uint16_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<int32_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<uint32_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<int64_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<uint64_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<half_float::half> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<float> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<double> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::instance(const vptr<void> &other);



template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator++() {
  instance(this);
  dev_ptr_.get()->increment(1);
  return this;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator++(uint_tp val) {
  vptr<Mtype> old_vptr(this);
  instance(this);
  dev_ptr_.get()->increment(1);
  return old_vptr;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator--() {
  instance(this);
  dev_ptr_.get()->decrement(1);
  return this;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator--(uint_tp val) {
  vptr<Mtype> old_vptr(this);
  instance(this);
  dev_ptr_.get()->decrement(1);
  return old_vptr;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator+(uint_tp val) {
  vptr<Mtype> new_vptr(this);
  new_vptr.get()->increment(val);
  return new_vptr;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator-(uint_tp val) {
  vptr<Mtype> new_vptr(this);
  new_vptr.get()->decrement(val);
  return new_vptr;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator+=(uint_tp val) {
  dev_ptr_.get()->increment(val);
  return this;
}

template<typename Mtype>
vptr<Mtype> vptr<Mtype>::operator-=(uint_tp val) {
  dev_ptr_.get()->decrement(val);
  return this;
}


template<typename Mtype>
template<typename Otype>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<Otype> &other) {
  return instance(other);
}

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<int8_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<uint8_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<int16_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<uint16_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<int32_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<uint32_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<int64_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<uint64_t> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<half_float::half> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<float> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<double> &other);

template<typename Mtype>
template<>
vptr<Mtype> vptr<Mtype>::operator=(const vptr<void> &other);


INSTANTIATE_POINTER_CLASS(vptr)

}
