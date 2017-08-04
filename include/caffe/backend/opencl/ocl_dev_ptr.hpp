#ifndef CAFFE_BACKEND_OPENCL_OCL_DEV_PTR_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_DEV_PTR_HPP_

#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/dev_ptr.hpp"

#ifdef USE_OPENCL

namespace caffe {

template<typename Mtype>
class ocl_dev_ptr : public dev_ptr<Mtype> {
 public:
  explicit ocl_dev_ptr(cl_mem ocl_mem);
  template<typename Otype>
  explicit ocl_dev_ptr(dev_ptr<Otype> *other);

  cl_mem get_ocl_mem();
  uint_tp get_off();

  void increment(int_tp val);
  void decrement(int_tp val);

 private:
  cl_mem ocl_mem_;
  uint_tp off_;
};

}  // namespace caffe

#endif  // USE_OPENCL

#endif  // CAFFE_BACKEND_OPENCL_OCL_DEV_PTR_HPP_
