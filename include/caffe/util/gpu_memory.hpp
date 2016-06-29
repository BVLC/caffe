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
  static void GetInfo(size_t* free_mem, size_t* used_mem) {
    return mgr_.GetInfo(free_mem, used_mem);
  }

  template <class Any>
  static void allocate(Any** ptr, size_t size,
      int device = INVALID_DEVICE,
      cudaStream_t stream = cudaStreamDefault) {
    if (!try_allocate(reinterpret_cast<void**>(ptr), size, device, stream)) {
      CUDA_CHECK(cudaGetDevice(&device));
      LOG(FATAL) << "Out of memory: failed to allocate " << size
          << " bytes on device " << device;
    }
  }

  static void deallocate(void* ptr, int device = INVALID_DEVICE,
      cudaStream_t stream = cudaStreamDefault) {
    mgr_.deallocate(ptr, device, stream);
  }

  static bool try_allocate(void** ptr, size_t size, int device = INVALID_DEVICE,
      cudaStream_t stream = cudaStreamDefault) {
    return mgr_.try_allocate(ptr, size, device, stream);
  }

  enum Mode {
    CUDA_MALLOC,   // Straight CUDA malloc/free (may be expensive)
    CUB_ALLOCATOR  // CUB caching allocator
  };

  // Scope initializes global Memory Manager for a given scope.
  // It's instantiated in test(), train() and time() Caffe brewing functions
  // as well as in unit tests main().
  struct Scope {
    Scope(const std::vector<int>& gpus, Mode m = CUB_ALLOCATOR,
          bool debug = false) {
      mgr_.init(gpus, m, debug);
    }
  };

  // Workspace's release() functionality depends on global pool availability
  // If pool is available, it returns memory to the pool and sets ptr to NULL
  // If pool is not available, it retains memory.
  // This is single GPU workspace. See MultiWorkspace for multi-GPU support.
  struct Workspace {
    Workspace()
      : ptr_(NULL), size_(0), device_(INVALID_DEVICE),
        stream_(cudaStreamDefault) {}
    Workspace(size_t size, int device = INVALID_DEVICE,
        cudaStream_t s = cudaStreamDefault)
      : ptr_(NULL), size_(0), device_(device), stream_(s) {
      reserve(size, device);
    }
    ~Workspace() { mgr_.deallocate(ptr_, device_, stream_); }

    void* data() const { return ptr_; }
    size_t size() const { return size_; }
    int device() const { return device_; }

    bool try_reserve(size_t size, int device = INVALID_DEVICE) {
      bool status = true;
      if (size > size_) {
        if (ptr_ != NULL) {
          mgr_.deallocate(ptr_, device_, stream_);
        }
        if (device != INVALID_DEVICE) {
          device_ = device;  // switch from default to specific one
        }
        status = mgr_.try_allocate(&ptr_, size, device_, stream_);
        if (status) {
          size_ = size;
        }
      }
      return status;
    }

    void reserve(size_t size, int device = INVALID_DEVICE) {
      if (!try_reserve(size, device)) {
        CUDA_CHECK(cudaGetDevice(&device));
        LOG(FATAL) << "Out of memory: failed to allocate " << size
            << " bytes on device " << device;
      }
    }

    void release() {
      if (mgr_.using_pool() && ptr_ != NULL) {
        mgr_.deallocate(ptr_, device_, stream_);
        ptr_ = NULL;
        size_ = 0;
      }
      // otherwise (no pool) - we retain memory in the buffer
    }

   private:
    void* ptr_;
    size_t size_;
    int device_;
    cudaStream_t stream_;

    DISABLE_COPY_AND_ASSIGN(Workspace);
  };

  // This implementation maintains workspaces on per-GPU basis.
  struct MultiWorkspace {
    bool try_reserve(size_t size) {
      return current_workspace()->try_reserve(size);
    }
    void reserve(size_t size) {
      current_workspace()->reserve(size);
    }
    void release() {
      current_workspace()->release();
    }
    void* data() const {
      return current_workspace()->data();
    }
    size_t size() const {
      return current_workspace()->size();
    }
    int device() const {
      return current_workspace()->device();
    }

   private:
    shared_ptr<Workspace> current_workspace() const;
    mutable vector<shared_ptr<Workspace> > ws_;
  };

 private:
  struct Manager {
    Manager();
    ~Manager();
    void GetInfo(size_t* free_mem, size_t* used_mem);
    void deallocate(void* ptr, int device, cudaStream_t stream);
    bool try_allocate(void** ptr, size_t size, int device, cudaStream_t);
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

    static const unsigned int BIN_GROWTH;  ///< Geometric growth factor
    static const unsigned int MIN_BIN;  ///< Minimum bin
    static const unsigned int MAX_BIN;  ///< Maximum bin
    static const size_t MAX_CACHED_BYTES;  ///< Maximum aggregate cached bytes
  };

  static const int INVALID_DEVICE;  ///< Default is invalid: CUB takes care

  static Manager mgr_;
};

}  // namespace caffe

#endif

#endif
