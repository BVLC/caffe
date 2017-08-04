#ifndef CAFFE_BACKEND_OPENCL_OCL_DEVICE_PROGRAM_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_DEVICE_PROGRAM_HPP_

#include "caffe/backend/device_program.hpp"

#ifdef USE_OPENCL
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#endif  // USE_OPENCL

namespace caffe {

#ifdef USE_OPENCL

class ocl_device_program : public device_program {
 public:
  explicit ocl_device_program();

  virtual void compile();
  virtual void launch_kernel(std::string name);

 private:
  viennacl::ocl::program ocl_program_;
};

#endif  // USE_OPENCL

}



#endif  // CAFFE_BACKEND_OPENCL_OCL_DEVICE_PROGRAM_HPP_
