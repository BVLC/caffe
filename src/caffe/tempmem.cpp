#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <cstring>
#include <vector>

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
  class Block{
   private:
    Block(const Block& o):data_(NULL), size_(0), is_locked_(false) {}

   public:
    void * data_;
    size_t size_;
    bool is_locked_;
    Block():data_(NULL), size_(0), is_locked_(false) {}
    ~Block() {
      if (data_)
        TemporaryMemoryAllocator<gpu>::free(data_);
    }
    void * try_lock(size_t max_size) {
      if (is_locked_) return NULL;
      is_locked_ = true;
      if (size_ < max_size) {
        size_ = max_size;
        if (data_)
          TemporaryMemoryAllocator<gpu>::free(data_);
        data_ = TemporaryMemoryAllocator<gpu>::calloc(size_);
      }
      return data_;
    }
    void unlock() {
      is_locked_ = false;
    }
  };
  std::vector< boost::shared_ptr<Block> > blocks_;
  size_t max_size_;
  boost::mutex mtx_;
  GlobalTemporaryMemory(const GlobalTemporaryMemory & o):max_size_(0) {}

 public:
  GlobalTemporaryMemory():max_size_(0) {}
  void * lock() {
    // Note: Currently concurrent accesses allocate duplicate memory of
    //       max_size_ in order to reduce the need to reallocate memory
    //       This might be a bit wasteful.
    boost::lock_guard<boost::mutex> guard(mtx_);
    for (int i = 0; i < blocks_.size(); i++) {
      void * r = blocks_[i]->try_lock(max_size_);
      if (r) return r;
    }
    blocks_.push_back(boost::make_shared<Block>());
    return blocks_.back()->try_lock(max_size_);
  }
  template<typename Dtype>
  Dtype * lock() {
    return static_cast<Dtype*>(lock());
  }
  void unlock(void * mem) {
    boost::lock_guard<boost::mutex> guard(mtx_);
    for (int i = 0; i < blocks_.size(); i++)
      if (blocks_[i]->is_locked_ && blocks_[i]->data_ == mem) {
        blocks_[i]->unlock();
        return;
      }
    LOG(WARNING) << "Unlock failed! Lost the memory block!";
  }
  void allocate(size_t size) {
    boost::lock_guard<boost::mutex> guard(mtx_);
    if (size > max_size_)
      max_size_ = size;
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
  CHECK(cpu_ptr_ != NULL) << "acquire failed!";
}
template<typename Dtype>
void TemporaryMemory<Dtype>::acquire_gpu() {
#ifdef CPU_ONLY
  NO_GPU;
#else
  gpu_ptr_ = gpu_memory_.lock<Dtype>();
  CHECK(gpu_ptr_ != NULL) << "acquire failed!";
#endif
}
template<typename Dtype>
void TemporaryMemory<Dtype>::release_cpu() {
  CHECK(cpu_ptr_ != NULL) << "Need to allocate and acquire the data first";
  cpu_memory_.unlock(cpu_ptr_);
  cpu_ptr_ = NULL;
}
template<typename Dtype>
void TemporaryMemory<Dtype>::release_gpu() {
#ifdef CPU_ONLY
  NO_GPU;
#else
  CHECK(gpu_ptr_ != NULL) << "Need to allocate and acquire the data first";
  gpu_memory_.unlock(gpu_ptr_);
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
