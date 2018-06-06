#ifdef USE_CUDA
#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/backend/cuda/cuda_device_program.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#include "caffe/cuda_nvrtc_headers.hpp"

namespace caffe {

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

CudaDevice::~CudaDevice() {
  for (size_t i = 0; i < cuda_headers_.size(); ++i) {
    free(cuda_headers_[i]);
    free(cuda_header_sources_[i]);
  }
  buffers_.clear();
}

void CudaDevice::Init() {
  // Force initialize CUDA context
  cudaSetDevice(id_);
  cudaFree(0);

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, id_));
  max_local_sizes_[0] = prop.maxThreadsDim[0];
  max_local_sizes_[1] = prop.maxThreadsDim[1];
  max_local_sizes_[2] = prop.maxThreadsDim[2];
  max_group_sizes_[0] = prop.maxGridSize[0];
  max_group_sizes_[1] = prop.maxGridSize[1];
  max_group_sizes_[2] = prop.maxGridSize[2];
  max_local_size_ = prop.maxThreadsPerBlock;
  cuda_major_ = prop.major;
  cuda_minor_ = prop.minor;

  this->preferred_vector_widths_[safe_type_name<char>()] = 4;
  this->preferred_vector_widths_[safe_type_name<int8_t>()] = 4;
  this->preferred_vector_widths_[safe_type_name<uint8_t>()] = 4;
  this->preferred_vector_widths_[safe_type_name<short>()] = 2;
  this->preferred_vector_widths_[safe_type_name<int16_t>()] = 2;
  this->preferred_vector_widths_[safe_type_name<uint16_t>()] = 2;
  this->preferred_vector_widths_[safe_type_name<int32_t>()] = 1;
  this->preferred_vector_widths_[safe_type_name<uint32_t>()] = 1;
  this->preferred_vector_widths_[safe_type_name<int64_t>()] = 1;
  this->preferred_vector_widths_[safe_type_name<uint64_t>()] = 1;
  this->preferred_vector_widths_[safe_type_name<half_fp>()] = 2;
  this->preferred_vector_widths_[safe_type_name<float>()] = 1;
  this->preferred_vector_widths_[safe_type_name<double>()] = 1;


  ReadHeaders();
  Device::Init();

  this->CreateMathProgram();
  this->CreateIm2ColProgram();
}

void CudaDevice::ReadHeaders() {
  map<string, string> cuda_headers = get_cuda_nvrtc_headers();
  for (map<string, string>::iterator it = cuda_headers.begin();
      it != cuda_headers.end(); ++it) {
    char* header_name = new char[it->first.length() + 1];
    strcpy(header_name, it->first.c_str());
    cuda_headers_.push_back(header_name);
    char* header_source = new char[it->second.length() + 1];
    strcpy(header_source, it->second.c_str());
    cuda_header_sources_.push_back(header_source);
  }
}

int_tp CudaDevice::get_header_count() {
  return cuda_headers_.size();
}

char** CudaDevice::get_header_names() {
  return &cuda_headers_[0];
}

char** CudaDevice::get_header_sources() {
  return &cuda_header_sources_[0];
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

void CudaDevice::MallocMemHost(uint_tp size, void** ptr) {
  CUDA_CHECK(cudaMallocHost(ptr, size));
}

void CudaDevice::FreeMemHost(void* ptr) {
  cudaFreeHost(ptr);
}

vptr<void> CudaDevice::MallocMemDevice(uint_tp size, void** ptr,
                                        bool zero_copy) {
  CHECK_GT(size, 0) << "Illegal allocation of size 0.";

  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  CUDA_CHECK(cudaSetDevice(this->id()));
  void* gpu_ptr;
  CUDA_CHECK(cudaMalloc(&gpu_ptr, size));
  CUDA_CHECK(cudaSetDevice(initial_device));
  CHECK_NE(zero_copy, true) << "Zero-copy not supported on CUDA.";
  return vptr<void>(make_shared<cuda_dev_ptr<void> >(gpu_ptr));
}

void CudaDevice::FreeMemDevice(vptr<void> ptr) {
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  CUDA_CHECK(cudaSetDevice(this->id()));
  CUDA_CHECK(cudaFree(ptr.get_cuda_ptr()));
  CUDA_CHECK(cudaSetDevice(initial_device));
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
  CHECK(work_size);
  CHECK(local);
  CHECK(group);
  CHECK(kernel);

  for(uint_tp i = 0; i < work_size->size(); ++i) {
    local->insert(local->begin() + i, 1);
    group->insert(group->begin() + i, 1);
  }

  bool done = false;
  vector<bool> local_done(work_size->size(), false);
  while (!done) {
    done = true;
    for (uint_tp i = 0; i < work_size->size(); ++i) {
      done = done && local_done[i];
    }
    for (uint_tp i = 0; i < work_size->size(); ++i) {
      if (!done
          && ((*local)[i] <= (*work_size)[i])
          && ((*local)[i] * 2 <= max_local_sizes_[i])) {
        (*local)[i] *= 2;
      } else {
        local_done[i] = true;
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
  return make_shared<CudaDeviceProgram>(this);
}

bool CudaDevice::CheckVendor(string vendor) {
  if (vendor.compare("NVIDIA") == 0) {
      return true;
  }
  return false;
}

bool CudaDevice::CheckCapability(DeviceCapability cap) {
  switch(cap) {
    case DEVICE_FP16_SUPPORT:
      return true;
    case DEVICE_FP32_SUPPORT:
      return true;
    case DEVICE_FP64_SUPPORT:
      return true;
    case DEVICE_INT32_LOCAL_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT64_LOCAL_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT32_LOCAL_EXTENDED_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT64_LOCAL_EXTENDED_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT32_GLOBAL_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT32_GLOBAL_EXTENDED_ATOMICS_SUPPORT:
      return true;
    case DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT:
      return true;
    case DEVICE_32_BIT_ADDRESS:
      // Report host pointer size
      return sizeof(void*) == 4;
    case DEVICE_64_BIT_ADDRESS:
      // Report host pointer size
      return sizeof(void*) == 8;
    case DEVICE_CUDA_DP4A_SUPPORT:
      return cuda_major_ >= 6 && cuda_minor_ >= 1;
    default:
      return false;
  }
}

bool CudaDevice::CheckType(string type) {
  return (type.compare("GPU") == 0);
}

bool CudaDevice::CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                                       uint_tp size) {
  return false;
}

void CudaDevice::SwitchQueue(uint_tp id) { }

void CudaDevice::FinishQueues() { }

}  // namespace caffe

#endif  // USE_CUDA
