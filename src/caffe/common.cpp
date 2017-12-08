#include <cmath>
#include <cstdio>
#include <ctime>
#include <glog/logging.h>
#include <memory>
#ifdef _WIN32
#include <process.h>
#endif

#include <deepir/allocator/buddy_pool.hpp>

#include "caffe/common.hpp"

namespace caffe {

// Make sure each thread can have different values.
// static boost::thread_specific_ptr<Caffe> thread_instance_;
static thread_local std::unique_ptr<Caffe> thread_instance_;

Caffe &Caffe::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
}

#ifdef CPU_ONLY // CPU-only Caffe.

Caffe::Caffe() : mode_(Caffe::CPU), device_id_(-1) {}

Caffe::~Caffe() {}

void Caffe::set_device(int device_id) {
  if (device_id >= 0) {
    NO_GPU;
    return;
  }
  Get().mode_ = CPU;
}

#else // Normal GPU + CPU Caffe.

Caffe::Caffe() : cublas_handle_(NULL), mode_(Caffe::CPU), device_id_(-1) {}

Caffe::~Caffe() {
  if (cublas_handle_)
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
}

void Caffe::set_device(int device_id) {
  if (device_id < 0) {
    Get().mode_ = CPU;
    return;
  }

  if (Get().device_id_ >= 0) {
    if (Get().device_id_ == device_id) {
      return;
    }
    std::cout << "device_id_=" << Get().device_id_ << std::endl;
    throw std::runtime_error("caffe thread has binded device");
  }

  CUDA_CHECK(cudaSetDevice(device_id));
  Get().device_id_ = device_id;
  Get().mode_ = GPU;
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&Get().cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    Get().cublas_handle_ = nullptr;
  } else {
    CUBLAS_CHECK(cublasSetStream(Get().cublas_handle_, cudaStreamPerThread));
  }

  Get().host_pool_ = std::make_shared<deepir::allocator::buddy_pool>(
      deepir::allocator::buddy_pool::alloc_location::host);
  Get().device_pool_ = std::make_shared<deepir::allocator::buddy_pool>(
      deepir::allocator::buddy_pool::alloc_location::device);
}

const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

#endif // CPU_ONLY

} // namespace caffe
