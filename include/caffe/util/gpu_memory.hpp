#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <vector>
#include "caffe/common.hpp"

namespace caffe {

class gpu_memory {
 public:
  enum PoolMode {
    NoPool,     // Straight CUDA malllc/free. May be very expensive
    CubPool,     // CUB caching allocator
#ifdef CPU_ONLY
    DefaultPool = NoPool
#else
    DefaultPool = CubPool     // CUB pool is able to use unified memory properly
#endif
  };

  static const char* getPoolName();
  static bool usingPool() {
    return mode_ != NoPool;
  }

  class arena {
   public:
    arena(const std::vector<int>& gpus,
          PoolMode m = DefaultPool, bool debug = false) {
      init(gpus, m, debug);
    }
    ~arena() {
      destroy();
     }
  };

#ifndef CPU_ONLY
  class buffer {
   public:
    // Construction/destruction
    buffer(): ptr_(NULL), stream_(), size_(0) {}
    buffer(size_t size, cudaStream_t s = cudaStreamDefault): stream_(s) {
      reserve(size);
    }
    ~buffer() { gpu_memory::deallocate(ptr_, stream_); }

    // Accessors
    void* data() const { return ptr_; }
    size_t size() const { return size_; }

    // Memory allocation/release
    void reserve(size_t size) {
      if (size > size_) {
        if (ptr_)
          gpu_memory::deallocate(ptr_, stream_);
        gpu_memory::allocate(&ptr_, size, stream_);
        size_ = size;
      }
    }

    /*
     * This method behaves differently depending on pool availability:
     * If pool is available, it returns memory to the pool and sets ptr to NULL
     * If pool is not available, it does nothing (retaining memory)
     */
    void release() {
      if (gpu_memory::usingPool()) {
        gpu_memory::deallocate(ptr_, stream_);
        ptr_ = NULL;
        size_ = 0;
      }
      // otherwise (no pool) - we retain memory in the buffer
    }

   private:
    void*         ptr_;
    cudaStream_t  stream_;
    size_t        size_;
  };
  static void update_dev_info(int device);

# endif

 private:
  static void init(const std::vector<int>&, PoolMode, bool);
  static void destroy();

  static bool             initialized_;
  static PoolMode         mode_;
  static bool             debug_;

#ifndef CPU_ONLY
  struct MemInfo {
    MemInfo()  {
      free = total = flush_count = 0;
    }
    size_t   free;
    size_t   total;
    unsigned flush_count;
  };

  static vector<MemInfo>  dev_info_;

 public:
  typedef void* pointer;

  static void allocate(pointer* ptr, size_t size,
                       cudaStream_t stream = cudaStreamDefault);
  static void deallocate(pointer ptr, cudaStream_t = cudaStreamDefault);

  static void getInfo(size_t *free_mem, size_t *used_mem);

 private:
  static void initMEM(const std::vector<int>& gpus, PoolMode m);

#endif
};

}  // namespace caffe

# endif
