#include <cstring>

#include "caffe/common.hpp"
#include "caffe/tempmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<bool gpu>
struct TemporaryMemoryAllocator {
};
#ifndef CPU_ONLY
// GPU allocator
template<>
struct TemporaryMemoryAllocator<true> {
  static void * calloc(size_t size) {
    void * p;
    CUDA_CHECK(cudaMalloc(&p, size));
    caffe_gpu_memset(size, 0, p);
    return p;
  }
  static void free(void * p) {
      cudaFree(p);
  }
};
#endif
// CPU allocator
template<>
struct TemporaryMemoryAllocator<false> {
  static void * calloc(size_t size) {
    void * p;
    CaffeMallocHost(&p, size);
    caffe_memset(size, 0, p);
    return p;
  }
  static void free(void * p) {
      CaffeFreeHost(p);
  }
};

template<bool gpu>
class GlobalTemporaryMemory {
 private:
  size_t size_;
  void * data_;
  bool is_locked_;

 public:
  GlobalTemporaryMemory():size_(0), data_(NULL), is_locked_(false) {}
  ~GlobalTemporaryMemory() {
    if (data_)
      TemporaryMemoryAllocator<gpu>::free(data_);
  }
  void * lock() {
    // Note: If we expect a concurrrent access, we might need to
    //       make this check atomic
    //       We might also want to keep a thread_local temp memory
    //       around.
    CHECK(!is_locked_) << "Temporary memory is already locked!";
    is_locked_ = true;
    if (!data_)  // We allocate here to make Travis happy
      data_ = TemporaryMemoryAllocator<gpu>::calloc(size_+1);
    return data_;
  }
  template<typename Dtype>
  Dtype * lock() {
    return static_cast<Dtype*>(lock());
  }
  void unlock() {
    is_locked_ = false;
  }
  void allocate(size_t size) {
    if (size_ < size) {
      CHECK(!is_locked_) << "Cannot allocate memory while locked!";
      size_ = size;
      if (data_) {
        TemporaryMemoryAllocator<gpu>::free(data_);
        data_ = TemporaryMemoryAllocator<gpu>::calloc(size_);
      }
    }
  }
};
#ifndef CPU_ONLY
static GlobalTemporaryMemory<true> gpu_memory_;
#endif
static GlobalTemporaryMemory<false> cpu_memory_;

template<typename Dtype>
TemporaryMemory<Dtype>::TemporaryMemory(size_t size):cpu_ptr_(NULL),
  gpu_ptr_(NULL), size_(0) {
  resize(size_);
}
template<typename Dtype>
TemporaryMemory<Dtype>::~TemporaryMemory() {
}

template<typename Dtype>
void TemporaryMemory<Dtype>::acquire_cpu() {
  cpu_ptr_ = cpu_memory_.lock<Dtype>();
}
template<typename Dtype>
void TemporaryMemory<Dtype>::acquire_gpu() {
#ifdef CPU_ONLY
  NO_GPU;
#else
  gpu_ptr_ = gpu_memory_.lock<Dtype>();
#endif
}
template<typename Dtype>
void TemporaryMemory<Dtype>::release_cpu() {
  cpu_memory_.unlock();
  cpu_ptr_ = NULL;
}
template<typename Dtype>
void TemporaryMemory<Dtype>::release_gpu() {
#ifdef CPU_ONLY
  NO_GPU;
#else
  gpu_memory_.unlock();
  gpu_ptr_ = NULL;
#endif
}
template<typename Dtype>
const Dtype* TemporaryMemory<Dtype>::cpu_data() const {
  CHECK(cpu_ptr_ != NULL) << "Need to allocate and acquire the data first";
  return cpu_ptr_;
}
template<typename Dtype>
const Dtype* TemporaryMemory<Dtype>::gpu_data() const {
  CHECK(gpu_ptr_ != NULL) << "Need to allocate and acquire the data first";
  return gpu_ptr_;
}
template<typename Dtype>
Dtype* TemporaryMemory<Dtype>::mutable_cpu_data() {
  CHECK(cpu_ptr_ != NULL) << "Need to allocate and acquire the data first";
  return cpu_ptr_;
}
template<typename Dtype>
Dtype* TemporaryMemory<Dtype>::mutable_gpu_data() {
  CHECK(gpu_ptr_ != NULL) << "Need to allocate and acquire the data first";
  return gpu_ptr_;
}
template<typename Dtype>
void TemporaryMemory<Dtype>::resize(size_t size) {
  size_ = size;
#ifndef CPU_ONLY
  gpu_memory_.allocate(size_*sizeof(Dtype));
#endif
  cpu_memory_.allocate(size_*sizeof(Dtype));
}

INSTANTIATE_CLASS(TemporaryMemory);
template class TemporaryMemory<int>;
template class TemporaryMemory<unsigned int>;

}  // namespace caffe
