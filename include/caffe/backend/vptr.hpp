#ifndef CAFFE_BACKEND_VPTR_HPP_
#define CAFFE_BACKEND_VPTR_HPP_

#include <cstddef>
#include "caffe/definitions.hpp"
#include "caffe/backend/dev_ptr.hpp"

namespace caffe {

template<typename Mtype>
class vptr {
  explicit vptr();

  template<typename Otype>
  explicit vptr(const vptr<Otype> &other);

 public:
  dev_ptr<Mtype>* get();

#ifdef USE_CUDA
  Mtype* get_cuda_ptr();
#endif

#ifdef USE_OPENCL
  cl_mem get_ocl_mem();
  uint_tp get_ocl_off();
#endif

  template<typename Otype>
  vptr<Mtype> operator=(const vptr<Otype> &other);

  vptr<Mtype> operator++();
  vptr<Mtype> operator--();
  vptr<Mtype> operator++(uint_tp val);
  vptr<Mtype> operator--(uint_tp val);
  vptr<Mtype> operator+(uint_tp val);
  vptr<Mtype> operator-(uint_tp val);
  vptr<Mtype> operator+=(uint_tp val);
  vptr<Mtype> operator-=(uint_tp val);

 private:
  template<typename Otype>
  vptr<Mtype> instance(const vptr<Otype> &other);

  std::shared_ptr<dev_ptr<Mtype> > dev_ptr_;
};

}  // namespace caffe

#endif  // CAFFE_BACKEND_VPTR_HPP_
