#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#ifdef USE_OPENCL
#include "caffe/util/OpenCL/OpenCLSupport.hpp"
#endif

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

#ifdef USE_CUDA
	if (gpu_ptr_) {
		CUDA_CHECK(cudaFree(gpu_ptr_));
	}
#endif

#ifdef USE_OPENCL
	if (gpu_ptr_) {
		BOOL_CHECK(caffe::OpenCL::clFree(gpu_ptr_));
		DLOG(INFO)<<"gpu_ptr_ = "<<gpu_ptr_<<" going out of scope.";
		gpu_ptr_ = NULL;
	}
#endif
}

inline void SyncedMemory::to_cpu() {

	std::string function = __func__;
	std::ostringstream oss;

  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
		if ( cpu_ptr_ == NULL ) {
			LOG(ERROR) << "failed to allocate "<<size_<<" Byte in main memory";
			return;
		}
		oss << "CPU@" << cpu_ptr_;
		memoryTag[cpu_ptr_] = oss.str();
		memoryCount++;
		//DLOG(INFO) << "malloc() " << getMemoryTag(cpu_ptr_).c_str() << " for " << size() << " Byte";

    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
	case HEAD_AT_GPU:
#if defined(USE_CUDA)
		if (cpu_ptr_ == NULL) {
			CaffeMallocHost(&cpu_ptr_, size_);
			own_cpu_data_ = true;
		}
		caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
		head_ = SYNCED;
#elif defined(USE_OPENCL)
		if (cpu_ptr_ == NULL) {
			CaffeMallocHost(&cpu_ptr_, size_);
			if ( cpu_ptr_ == NULL ) {
				LOG(ERROR) << "failed to allocate "<<size_<<" Byte in main memory";
				return;
			}
			oss << "CPU@" << cpu_ptr_;
			memoryTag[cpu_ptr_] = oss.str();
			memoryCount++;
			//DLOG(INFO) << "malloc() " << getMemoryTag(cpu_ptr_).c_str() << " for " << size() << " Byte";

			own_cpu_data_ = true;
		}
		caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_, caffe::OpenCL::COPY_GPU_TO_CPU);
		head_ = SYNCED;
#else
		NO_GPU;
#endif
		break;

	case HEAD_AT_CPU:

	case SYNCED:
		break;
  }
}

inline void SyncedMemory::to_gpu() {
	std::string function = __func__;

#if defined(USE_CUDA)
	switch (head_) {
		case UNINITIALIZED:
		CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
		caffe_gpu_memset(size_, 0, gpu_ptr_);
		head_ = HEAD_AT_GPU;
		break;
		case HEAD_AT_CPU:
		if (gpu_ptr_ == NULL) {
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
		}
		caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
		head_ = SYNCED;
		break;
		case HEAD_AT_GPU:
		case SYNCED:
		break;
	}
#elif defined(USE_OPENCL)
	switch (head_) {

	case UNINITIALIZED:
		BOOL_CHECK(caffe::OpenCL::clMalloc(&gpu_ptr_, size_));
		caffe_gpu_memset(size_, (char) 0, gpu_ptr_);
		head_ = HEAD_AT_GPU;
		break;

	case HEAD_AT_CPU:
		if (gpu_ptr_ == NULL) {
			BOOL_CHECK(caffe::OpenCL::clMalloc(&gpu_ptr_, size_));
		}
		caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_, caffe::OpenCL::COPY_CPU_TO_GPU);
		head_ = SYNCED;
		break;
	case HEAD_AT_GPU:
	case SYNCED:
		break;
	}
#else
	NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#if defined(USE_CUDA) || defined(USE_OPENCL)
	to_gpu();
	return (const void*) gpu_ptr_;
#else
	NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#if defined(USE_CUDA) || defined(USE_OPENCL)
	to_gpu();
	head_ = HEAD_AT_GPU;
	return gpu_ptr_;
#else
	NO_GPU;
#endif
}

std::string SyncedMemory::getMemoryTag(const void* ptr) {

	if ( ptr == NULL ) {
		return "NULL";
	}
	if ( memoryTag.find(ptr) == memoryTag.end() ) {
		return "UNKNOWN";
	}
	return memoryTag[ptr];
}

}  // namespace caffe

