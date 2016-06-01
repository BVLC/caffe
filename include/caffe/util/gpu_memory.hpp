#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <vector>
#include "caffe/common.hpp"

namespace caffe {

class GPUMemoryManager {
 public:
  enum PoolMode {
    NO_POOL,  // Straight CUDA malloc/free (may be expensive)
    CUB_POOL,  // CUB caching allocator
#ifdef CPU_ONLY
    DEFAULT_POOL = NO_POOL
#else
    DEFAULT_POOL = CUB_POOL  // CUB pool is able to use unified memory properly
#endif
  };

  static const char* pool_name();
  static bool using_pool() {
    return mode_ != NO_POOL;
  }

  class Arena {
   public:
    Arena(const std::vector<int>& gpus, PoolMode m = DEFAULT_POOL, bool debug =
        false) {
      init(gpus, m, debug);
    }
    ~Arena() {
      destroy();
    }
  };

#ifndef CPU_ONLY
  class Buffer {
   public:
    // Construction/destruction
    Buffer() :
        ptr_(NULL), stream_(), size_(0) {
    }
    Buffer(size_t size, cudaStream_t s = cudaStreamDefault) :
        stream_(s) {
      reserve(size);
    }
    ~Buffer() {
      GPUMemoryManager::deallocate(ptr_, stream_);
    }

    // Accessors
    void* data() const {
      return ptr_;
    }
    size_t size() const {
      return size_;
    }

    // Memory allocation/release
    bool try_reserve(size_t size) {
      bool status = true;
      if (size > size_) {
        if (ptr_) {
          GPUMemoryManager::deallocate(ptr_, stream_);
        }
        status = GPUMemoryManager::try_allocate(&ptr_, size, stream_);
        if (status) {
          size_ = size;
        }
      }
      return status;
    }

    void reserve(size_t size) {
      CHECK(try_reserve(size));
    }

    /*
     * This method behaves differently depending on pool availability:
     * If pool is available, it returns memory to the pool and sets ptr to NULL
     * If pool is not available, it does nothing (retaining memory)
     */
    void release() {
      if (GPUMemoryManager::using_pool()) {
        GPUMemoryManager::deallocate(ptr_, stream_);
        ptr_ = NULL;
        size_ = 0;
      }
      // otherwise (no pool) - we retain memory in the buffer
    }

   private:
    void* ptr_;
    cudaStream_t stream_;
    size_t size_;
  };
  static void update_dev_info(int device);
#endif  // CPU_ONLY

 private:
  static void init(const std::vector<int>&, PoolMode, bool);
  static void destroy();

  static bool initialized_;
  static PoolMode mode_;
  static bool debug_;

#ifndef CPU_ONLY
  struct MemInfo {
    MemInfo() {
      free_ = total_ = flush_count_ = 0;
    }
    size_t free_;
    size_t total_;
    unsigned flush_count_;
  };
  static vector<MemInfo> dev_info_;

 public:
  typedef void* pointer;
  static bool try_allocate(pointer* ptr, size_t size, cudaStream_t stream =
      cudaStreamDefault);
  static void allocate(pointer* ptr, size_t size, cudaStream_t stream =
      cudaStreamDefault) {
    CHECK(try_allocate(ptr, size, stream));
  }
  static void deallocate(pointer ptr, cudaStream_t = cudaStreamDefault);
  static void GetInfo(size_t* free_mem, size_t* used_mem);

 private:
  static void InitMemory(const std::vector<int>& gpus, PoolMode m);
#endif
};

}  // namespace caffe

#endif
