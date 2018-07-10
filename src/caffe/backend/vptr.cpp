#include <memory>

#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

namespace caffe {

template<typename Dtype>
void* vptr<Dtype, typename const_enable_if<Dtype>::type>::nullptr_ref_ =
    nullptr;

template<typename Dtype>
void* vptr<Dtype, typename non_const_enable_if<Dtype>::type>::nullptr_ref_ =
    nullptr;

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>::vptr()
  : dev_ptr_(nullptr) {}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>::vptr()
  : dev_ptr_(nullptr) {}


template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(shared_ptr<dev_ptr<Dtype> > ptr) : dev_ptr_(ptr) {}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(shared_ptr<dev_ptr<Dtype> > ptr) : dev_ptr_(ptr) {}


template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<bool> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<bool> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<char> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<char> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<int8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<int8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<int16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<int16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<int32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<int32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<int64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<int64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<uint64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<half_fp> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<half_fp> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<float> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<float> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<double> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<double> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}

template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<void> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename non_const_enable_if<Dtype>::type>
  ::vptr(const vptr<void> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}


template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const bool> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const char> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const int8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const int16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const int32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const int64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const uint8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const uint16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const uint32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const uint64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const half_fp> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const float> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const double> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}
template<typename Dtype>
vptr<Dtype, typename const_enable_if<Dtype>::type>
  ::vptr(const vptr<const void> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
}


template<typename Dtype>
dev_ptr<Dtype>* vptr<Dtype, typename const_enable_if<Dtype>::type>::get()
    const {
  return dev_ptr_.get();
}
template<typename Dtype>
dev_ptr<Dtype>* vptr<Dtype, typename non_const_enable_if<Dtype>::type>::get()
    const {
  return dev_ptr_.get();
}

template<typename Dtype>
shared_ptr<dev_ptr<Dtype> > vptr<Dtype,
                   typename const_enable_if<Dtype>::type>::get_dev_ptr() const {
  return dev_ptr_;
}
template<typename Dtype>
shared_ptr<dev_ptr<Dtype> > vptr<Dtype,
               typename non_const_enable_if<Dtype>::type>::get_dev_ptr() const {
  return dev_ptr_;
}

#ifdef USE_CUDA
template<typename Dtype>
Dtype* vptr<Dtype, typename const_enable_if<Dtype>::type>::get_cuda_ptr()
    const {
  if (dev_ptr_ == nullptr) {
    return nullptr;
  }
  return (dynamic_cast<cuda_dev_ptr<Dtype>*>(dev_ptr_.get()))->get();
}
template<typename Dtype>
Dtype* vptr<Dtype, typename non_const_enable_if<Dtype>::type>::get_cuda_ptr()
    const {
  if (dev_ptr_ == nullptr) {
    return nullptr;
  }
  return (dynamic_cast<cuda_dev_ptr<Dtype>*>(dev_ptr_.get()))->get();
}
template<typename Dtype>
void* vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                    ::get_cuda_ptr_ptr() const {
  if (dev_ptr_ == nullptr) {
    // Needs a pointer to a null-pointer (static class member)
    nullptr_ref_ = nullptr;
    return &nullptr_ref_;
  }
  return (dynamic_cast<cuda_dev_ptr<Dtype>*>(dev_ptr_.get()))->get_ptr();
}
template<typename Dtype>
void* vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                    ::get_cuda_ptr_ptr() const {
  if (dev_ptr_ == nullptr) {
    // Needs a pointer to a null-pointer (static class member)
    nullptr_ref_ = nullptr;
    return &nullptr_ref_;
  }
  return (dynamic_cast<cuda_dev_ptr<Dtype>*>(dev_ptr_.get()))->get_ptr();
}
#endif  // USE_CUDA

#ifdef USE_OPENCL
template<typename Dtype>
cl_mem vptr<Dtype, typename const_enable_if<Dtype>::type>::get_ocl_mem()
    const {
  if (dev_ptr_ == nullptr)
    return nullptr;
  return (dynamic_cast<ocl_dev_ptr<Dtype>*>(dev_ptr_.get()))->get_ocl_mem();
}
template<typename Dtype>
cl_mem vptr<Dtype, typename non_const_enable_if<Dtype>::type>::get_ocl_mem()
    const {
  if (dev_ptr_ == nullptr)
    return nullptr;
  return (dynamic_cast<ocl_dev_ptr<Dtype>*>(dev_ptr_.get()))->get_ocl_mem();
}
template<typename Dtype>
uint_tp vptr<Dtype, typename const_enable_if<Dtype>::type>::get_ocl_off()
    const {
  if (dev_ptr_ == nullptr)
    return 0;
  return (dynamic_cast<ocl_dev_ptr<Dtype>*>(dev_ptr_.get()))->get_off();
}
template<typename Dtype>
uint_tp vptr<Dtype, typename non_const_enable_if<Dtype>::type>::get_ocl_off()
    const {
  if (dev_ptr_ == nullptr)
    return 0;
  return (dynamic_cast<ocl_dev_ptr<Dtype>*>(dev_ptr_.get()))->get_off();
}
#endif  // USE_OPENCL


template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                                ::operator++() {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->increment(1);
  return *this;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                                ::operator++() {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->increment(1);
  return *this;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                         ::operator++(int val) {
  vptr<Dtype> old_vptr(*this);
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->increment(1);
  return old_vptr;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                         ::operator++(int val) {
  vptr<Dtype> old_vptr(*this);
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->increment(1);
  return old_vptr;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                                ::operator--() {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->decrement(1);
  return *this;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                                ::operator--() {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->decrement(1);
  return *this;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                         ::operator--(int val) {
  vptr<Dtype> old_vptr(*this);
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_.get()->decrement(1);
  return old_vptr;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                         ::operator--(int val) {
  vptr<Dtype> old_vptr(*this);
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->decrement(1);
  return old_vptr;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                      ::operator+(uint_tp val) {
  vptr<Dtype> new_vptr(*this);
  new_vptr += val;
  return new_vptr;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                      ::operator+(uint_tp val) {
  vptr<Dtype> new_vptr(*this);
  new_vptr += val;
  return new_vptr;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                      ::operator-(uint_tp val) {
  vptr<Dtype> new_vptr(*this);
  new_vptr -= val;
  return new_vptr;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                      ::operator-(uint_tp val) {
  vptr<Dtype> new_vptr(*this);
  new_vptr -= val;
  return new_vptr;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                     ::operator+=(uint_tp val) {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->increment(val);
  return *this;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                     ::operator+=(uint_tp val) {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->increment(val);
  return *this;
}

template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename const_enable_if<Dtype>::type>
                                                     ::operator-=(uint_tp val) {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->decrement(val);
  return *this;
}
template<typename Dtype>
vptr<Dtype> vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                                     ::operator-=(uint_tp val) {
  dev_ptr_ = dev_ptr_->instance((Dtype*)nullptr);
  dev_ptr_->decrement(val);
  return *this;
}


template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                          ::operator=(const vptr<bool> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                          ::operator=(const vptr<bool> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                          ::operator=(const vptr<char> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                          ::operator=(const vptr<char> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                        ::operator=(const vptr<int8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                        ::operator=(const vptr<int8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<int16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<int16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<int32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<int32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<int64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<int64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<uint8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                       ::operator=(const vptr<uint8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                      ::operator=(const vptr<uint16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                      ::operator=(const vptr<uint16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                      ::operator=(const vptr<uint32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                      ::operator=(const vptr<uint32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                      ::operator=(const vptr<uint64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                      ::operator=(const vptr<uint64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                              ::operator=(const vptr<half_fp> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                              ::operator=(const vptr<half_fp> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                         ::operator=(const vptr<float> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                         ::operator=(const vptr<float> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  } return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                        ::operator=(const vptr<double> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                        ::operator=(const vptr<double> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}

template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                          ::operator=(const vptr<void> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                                          ::operator=(const vptr<void> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}


template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                    ::operator=(const vptr<const bool> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                    ::operator=(const vptr<const char> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::operator=(const vptr<const int8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::operator=(const vptr<const int16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::operator=(const vptr<const int32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::operator=(const vptr<const int64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                 ::operator=(const vptr<const uint8_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                ::operator=(const vptr<const uint16_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                ::operator=(const vptr<const uint32_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                ::operator=(const vptr<const uint64_t> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                        ::operator=(const vptr<const half_fp> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                   ::operator=(const vptr<const float> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::operator=(const vptr<const double> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}
template<typename Dtype>
vptr<Dtype>& vptr<Dtype, typename const_enable_if<Dtype>::type>
                                  ::operator=(const vptr<const void> &other) {
  if (other.get_dev_ptr() != nullptr) {
    dev_ptr_ = other.get_dev_ptr()->instance((Dtype*)nullptr);
  } else {
    dev_ptr_ = nullptr;
  }
  return *this;
}


template<typename Dtype>
bool vptr<Dtype, typename const_enable_if<Dtype>::type>
                              ::is_valid() const {
  return dev_ptr_ != nullptr;
}

template<typename Dtype>
bool vptr<Dtype, typename non_const_enable_if<Dtype>::type>
                              ::is_valid() const {
  return dev_ptr_ != nullptr;
}

INSTANTIATE_POINTER_CLASS(vptr);

}
