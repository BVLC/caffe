#include "caffe/backend/cuda/cuda_device.hpp"

namespace caffe {

#ifdef USE_CUDA

cuda_device::cuda_device(uint_tp id, uint_tp list_id)
    : current_queue_id_(0), workgroup_sizes_(3, 0), id_(id), list_id_(list_id),
      backend_(BACKEND_CUDA), memory_usage_(0), peak_memory_usage_(0),
      host_unified_(false), name_("") {
}

std::string cuda_device::name() {
  if (name_ == "") {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, id_);
    std::string extsstr(&prop.name[0]);
    std::replace(extsstr.begin(), extsstr.end(), ' ', '_');
    name_ = extsstr;
  }
  return name_;
}

uint_tp cuda_device::num_queues() {
  return 1;
}

void cuda_device::Init() {
    workgroup_sizes_[0] = CAFFE_CUDA_NUM_THREADS;
    workgroup_sizes_[1] = CAFFE_CUDA_NUM_THREADS;
    workgroup_sizes_[2] = CAFFE_CUDA_NUM_THREADS;
}

bool cuda_device::CheckVendor(std::string vendor) {
  if (vendor.compare("NVIDIA") == 0) {
      return true;
  }
  return false;
}

bool cuda_device::CheckCapability(std::string cap) {
  if (cap == "cl_khr_int32_base_atomics" ||
      cap == "cl_khr_int64_base_atomics" ||
      cap == "cl_khr_global_int32_base_atomics") {
    return true;
  }
  return false;
}

bool cuda_device::CheckType(std::string type) {
  if (backend_ == BACKEND_CUDA) {
    if (type.compare("GPU") == 0)
      return true;
  }
  return false;
}

void device::SwitchQueue(uint_tp id) {
  return;
}

void device::FinishQueues() {
  return;
}

#endif  // USE_CUDA

}  // namespace caffe
