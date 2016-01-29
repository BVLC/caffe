#ifndef CAFFE_OCL_DEV_PTR_HPP_
#define CAFFE_OCL_DEV_PTR_HPP_

#ifdef USE_GREENTEA

#include "caffe/dev_ptr.hpp"
#ifndef __APPLE__
#include "CL/cl.h"
#else
#include "OpenCL/cl.h"
#endif

namespace caffe {

template<typename Type> class ocl_dev_ptr : public dev_ptr<Type> {
 public:
  explicit ocl_dev_ptr(cl_mem ocl_mem);
  Type* get();
  std::size_t off();

 private:
  cl_mem ocl_mem_;
  std::size_t off_;
};

}  // namespace caffe

#endif  // USE_GREENTEA

#endif /* CAFFE_OCL_DEV_PTR_HPP_ */
