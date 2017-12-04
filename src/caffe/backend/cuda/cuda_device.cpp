#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/backend/cuda/cuda_device_program.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

namespace caffe {

#ifdef USE_CUDA

CudaDevice::CudaDevice(uint_tp id, uint_tp list_id) {
  current_queue_id_ = 0;
  max_local_sizes_ = vector<size_t>(3, 0);
  max_group_sizes_ = vector<size_t>(3, 0);
  id_ = id;
  list_id_ = list_id;
  backend_ = BACKEND_CUDA;
  memory_usage_ = 0;
  peak_memory_usage_ = 0;
  host_unified_ = false;
  name_ = "";
}

void CudaDevice::Init() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, id_));
  max_local_sizes_[0] = prop.maxThreadsDim[0];
  max_local_sizes_[1] = prop.maxThreadsDim[1];
  max_local_sizes_[2] = prop.maxThreadsDim[2];
  max_group_sizes_[0] = prop.maxGridSize[0];
  max_group_sizes_[1] = prop.maxGridSize[1];
  max_group_sizes_[2] = prop.maxGridSize[2];
  max_local_size_ = prop.maxThreadsPerBlock;

  this->CreateMathProgram();
  this->CreateIm2ColProgram();
}

string CudaDevice::name() {
  if (name_ == "") {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, id_);
    string extsstr(&prop.name[0]);
    std::replace(extsstr.begin(), extsstr.end(), ' ', '_');
    name_ = extsstr;
  }
  return name_;
}

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.

void CudaDevice::MallocMemHost(void** ptr, uint_tp size) {
  CUDA_CHECK(cudaMallocHost(ptr, size));
}

void CudaDevice::FreeMemHost(void* ptr) {
  cudaFreeHost(ptr);
}

vptr<void> CudaDevice::MallocMemDevice(uint_tp size, void** ptr,
                                        bool zero_copy) {
  CHECK_GT(size, 0) << "Illegal allocation of size 0.";

  int initial_device;
  cudaGetDevice(&initial_device);
  cudaSetDevice(this->id());
  void* gpu_ptr;
  CUDA_CHECK(cudaMalloc(&gpu_ptr, size));
  cudaSetDevice(initial_device);
  CHECK_NE(zero_copy, true) << "Zero-copy not supported on CUDA.";
  return vptr<void>(std::make_shared<cuda_dev_ptr<void> >(gpu_ptr));
}

void CudaDevice::FreeMemDevice(vptr<void> ptr) {
  int initial_device;
  cudaGetDevice(&initial_device);
  cudaSetDevice(this->id());
  cudaFree(ptr.get_cuda_ptr());
  cudaSetDevice(initial_device);
}

bool CheckZeroCopy(vptr<void> gpu_ptr, void* cpu_ptr, uint_tp size) {
  return false;
}

uint_tp CudaDevice::num_queues() {
  return 1;
}

bool CudaDevice::is_host_unified() {
  return false;
}

void CudaDevice::get_threads(const vector<size_t>* work_size,
                             vector<size_t>* group, vector<size_t>* local,
                             DeviceKernel* kernel, bool auto_select) {

  for(uint_tp i = 0; i < work_size->size(); ++i) {
    local->insert(local->begin() + i, 1);
    group->insert(group->begin() + i, 1);
  }

  bool done = false;
  while (!done) {
    for (uint_tp i = 0; i < work_size->size(); ++i) {
      if (!done
          && ((*local)[i] < (*work_size)[i])
          && ((*local)[i] * 2 < max_local_sizes_[i])) {
        (*local)[i] *= 2;
      }
      size_t total_local_size = 1;
      for (uint_tp j = 0; j < work_size->size(); ++j) {
        total_local_size *= (*local)[j];
      }
      if (total_local_size > max_local_size_) {
        (*local)[i] /= 2;
        done = true;
      }
    }
  }

  for (uint_tp i = 0; i < work_size->size(); ++i) {
    (*group)[i] = ((*work_size)[i] - 1) / ((*local)[i]) + 1;
  }
}

shared_ptr<DeviceProgram> CudaDevice::CreateProgram() {
  return std::make_shared<CudaDeviceProgram>(this);
}

bool CudaDevice::CheckVendor(string vendor) {
  if (vendor.compare("NVIDIA") == 0) {
      return true;
  }
  return false;
}

bool CudaDevice::CheckCapability(string cap) {
  if (cap == "cl_khr_int32_base_atomics" ||
      cap == "cl_khr_int64_base_atomics" ||
      cap == "cl_khr_global_int32_base_atomics") {
    return true;
  }
  return false;
}

bool CudaDevice::CheckType(string type) {
  if (backend_ == BACKEND_CUDA) {
    if (type.compare("GPU") == 0)
      return true;
  }
  return false;
}

void Device::SwitchQueue(uint_tp id) {
  return;
}

void Device::FinishQueues() {
  return;
}

#endif  // USE_CUDA

}  // namespace caffe
