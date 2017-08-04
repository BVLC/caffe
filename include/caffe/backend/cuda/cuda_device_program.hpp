#ifndef CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_
#define CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_

#include "caffe/backend/device_program.hpp"

#ifdef USE_CUDA
#include "cuda.h"
#include "nvrtc.h"
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA
class cuda_device_program : public device_program {
 public:
  explicit cuda_device_program();

  virtual void compile();
  virtual void launch_kernel(std::string name, );


 private:
  nvrtcProgram cuda_program_;
  CUmodule cuda_module_;

};

#endif // USE_CUDA

}

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_
