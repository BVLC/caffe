#include "caffe/backend/cuda/cuda_device_program.hpp"

namespace caffe {

#ifdef USE_CUDA

void cuda_device_program::compile() {


}

void cuda_device_program::launch(std::string kernel_name,
                                 std::vector<uint_tp> &local,
                                 std::vector<uint_tp> &group,
                                 ) {
  CUfunction kernel;
  cuModuleGetFunction(&kernel, cuda_module_, kernel_name.c_str());

  void *args[] = { &bottom_data, &weight, &bias, &top_data };
  cuLaunchKernel(kernel,
                 group[0], group[1], group[2],      // Group
                 local[0], local[1], local[2],      // Local
                 0, NULL, args, 0);                 // Arguments
  cuCtxSynchronize();
}


#endif  // USE_CUDA



}

