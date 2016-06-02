#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <vector>
#include "caffe/common.hpp"
#ifndef CPU_ONLY

namespace cub {
  class CachingDeviceAllocator;
}

namespace caffe {

struct GPUMemory {

  static GPUMemory& Get();

  void GetInfo(size_t* free_mem, size_t* used_mem) {
    return mgr_.GetInfo(free_mem, used_mem);
  }

  template <class Any>
  void allocate(Any** ptr, size_t size, cudaStream_t s = cudaStreamDefault) {
    CHECK(try_allocate(reinterpret_cast<void **>(ptr), size, s));
  }

  void deallocate(void* ptr, cudaStream_t s = cudaStreamDefault) {
    mgr_.deallocate(ptr, s);
  }

  bool try_allocate(void** ptr, size_t size, cudaStream_t stream =
                    cudaStreamDefault) {
    return mgr_.try_allocate(ptr, size, stream);
  }

  enum Mode {
    CUDA_MALLOC,   // Straight CUDA malloc/free (may be expensive)
    CUB_ALLOCATOR  // CUB caching allocator
  };

 private:
  struct Manager {
    Manager();
    ~Manager();
    void GetInfo(size_t* free_mem, size_t* used_mem);
    void deallocate(void* ptr, cudaStream_t s);
    bool try_allocate(void** ptr, size_t size, cudaStream_t);
    const char* pool_name() const;
    bool using_pool() const { return mode_ != CUDA_MALLOC; }
    void init(const std::vector<int>&, Mode, bool);

    Mode mode_;
    bool debug_;

   private:
    struct DevInfo {
      DevInfo() {
        free_ = total_ = flush_count_ = 0;
      }
      size_t free_;
      size_t total_;
      unsigned flush_count_;
    };
    void update_dev_info(int device);
    vector<DevInfo> dev_info_;
    bool initialized_;
    cub::CachingDeviceAllocator* cub_allocator_;

    static unsigned int BIN_GROWTH;  ///< Geometric growth factor for bin-sizes
    static unsigned int MIN_BIN;  ///< Minimum bin
    static unsigned int MAX_BIN;  ///< Maximum bin
    static size_t MAX_CACHED_BYTES;  ///< Maximum aggregate cached bytes
  };

 public:
  struct Arena {
    Arena(const std::vector<int>& gpus, Mode m = CUB_ALLOCATOR,
          bool debug = false) {
      Get().mgr_.init(gpus, m, debug);
    }
  };

  // Buffer's release() functionality depends on global pool availability
  // If pool is available, it returns memory to the pool and sets ptr to NULL
  // If pool is not available, it retains memory.
  struct Buffer {
    Buffer() : ptr_(NULL), stream_(), size_(0) {}
    Buffer(size_t size, cudaStream_t s = cudaStreamDefault) : stream_(s) {
      reserve(size);
    }
    ~Buffer() { Get().mgr_.deallocate(ptr_, stream_); }

    void* data() const { return ptr_; }
    size_t size() const { return size_; }

    bool try_reserve(size_t size) {
      bool status = true;
      if (size > size_) {
        if (ptr_) {
          Get().mgr_.deallocate(ptr_, stream_);
        }
        status = Get().mgr_.try_allocate(&ptr_, size, stream_);
        if (status) {
          size_ = size;
        }
      }
      return status;
    }

    void reserve(size_t size) { CHECK(try_reserve(size)); }

    void release() {
      if (Get().mgr_.using_pool()) {
        Get().mgr_.deallocate(ptr_, stream_);
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

 private:
  GPUMemory() {}
  Manager mgr_;
};

}  // namespace caffe

#endif

#endif
