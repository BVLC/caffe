#include <algorithm>

#include "caffe/backend/cuda/cuda_device_kernel.hpp"
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"

namespace caffe {

#ifdef USE_CUDA

CudaDeviceKernel::CudaDeviceKernel(Device* dev,
                                   string name,
                                   shared_ptr<CUfunction> cuda_kernel,
                                   KernelArgs args) :
                                       DeviceKernel(dev, name, args),
                                   cuda_args_(0),
                                   cuda_kernel_(cuda_kernel) {
}

void CudaDeviceKernel::Execute(vector<size_t> group,
                               vector<size_t> local) {
  vector<size_t> group_ws(3);
  vector<size_t> local_ws(3);

  for (uint_tp i = 0; i < 3; ++i) {
    group_ws[i] = group.size() > i ? group[i] : 1;
    local_ws[i] = local.size() > i ? local[i] : 1;
  }

  /*
  for (int_tp i = 0; i < 3; ++i) {
    std::cout << "Global: " << group_ws[i] << std::endl;
    std::cout << "Local: " << local_ws[i] << std::endl;
  }
  */

  void **cuargs = &cuda_args_[0];

  CUresult result = cuLaunchKernel(*cuda_kernel_.get(),
                                   // Group
                                   group_ws[0], group_ws[1], group_ws[2],
                                   // Local
                                   local_ws[0], local_ws[1], local_ws[2],
                                   // Arguments
                                   0, NULL, cuargs, NULL);
  cuCtxSynchronize();
  if (result != CUDA_SUCCESS) {
    LOG(FATAL) << "Kernel launch failed (" << cudaGetErrorString(result) << ")"
               << " (" << this->name_ << ")" << std::endl;
  }
  CUDA_POST_KERNEL_CHECK;

  // Reset kernel arguments
  this->arg_idx_ = 0;
  cuda_args_ = vector<void*>();
}

void CudaDeviceKernel::set_arg(uint_tp idx, const bool *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const char *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const int8_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const int16_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const int32_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const int64_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const uint8_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const uint16_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const uint32_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const uint64_t *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const half_fp *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const float *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, const double *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, (void*) arg);
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}

void CudaDeviceKernel::set_arg(uint_tp idx, vptr<char> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<int8_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<int16_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<int32_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<int64_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<uint8_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<uint16_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<uint32_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<uint64_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<half_fp> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<float> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<double> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<void> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}

void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const char> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const int8_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const int16_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const int32_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const int64_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const uint8_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const uint16_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const uint32_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const uint64_t> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const half_fp> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const float> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const double> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}
void CudaDeviceKernel::set_arg(uint_tp idx, vptr<const void> *arg) {
  cuda_args_.insert(cuda_args_.begin() + idx, arg->get_cuda_ptr_ptr());
  this->arg_idx_ = std::max(idx + 1, this->arg_idx_);
}

#endif  // USE_CUDA

}
