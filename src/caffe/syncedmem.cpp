#include <cxxabi.h>
#include <execinfo.h>
#include <unistd.h>

#include <string>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

bool SyncedMemory::debug_info_ = false;

void SyncedMemory::debug_alloc() {
  // first just print the size of the allocation
  std::ostringstream size_str;
  size_str << size_ << " bytes";
  if (size_ > (1 << 20)) {
    size_str << " (" << static_cast<float>(size_) / (1 << 20) << " MB)";
  }
  LOG(INFO) << "debug: gpu alloc " << size_str.str();

  // now try to figure out who is responsible
  // the call stack should always go:
  // original caller -> Blob::*_gpu_(type) -> SyncedMemory::*_gpu_data ->
  //  -> SyncedMemory::to_gpu -> this function, so we need index four
  //  in the stack frame
  // if this fails, we'll just give up
  void* bt_buffer[5];
  int bt_size = backtrace(bt_buffer, sizeof(bt_buffer) / sizeof(void*));
  char** bt_strings = backtrace_symbols(bt_buffer, bt_size);
  if (bt_strings == NULL) {
    // out of memory for the backtrace
    return;
  }
  // parse and demangle the calling function name
  std::string caller_frame(bt_strings[4]);
  free(bt_strings);
  int start = caller_frame.find("(");
  int end = caller_frame.find("+");
  if (start == std::string::npos || end == std::string::npos) {
    return;
  }
  std::string caller_name = caller_frame.substr(start + 1, end - start - 1);
  int status;
  char* demang = abi::__cxa_demangle(caller_name.c_str(), NULL, NULL, &status);
  if (status) {
    return;
  }
  std::string demang_str(demang);
  free(demang);
  // cut out the argument list, which can be very long and hard to read
  demang_str = demang_str.substr(0, demang_str.find("("));
  LOG(INFO) << "debug:  from " <<  demang_str;
}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
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
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    if (debug_info_) {
      debug_alloc();
    }
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      if (debug_info_) {
        debug_alloc();
      }
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
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
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
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
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
#endif
}


}  // namespace caffe

