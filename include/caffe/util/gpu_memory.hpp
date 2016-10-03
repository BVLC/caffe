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

  static cudaStream_t device_stream(int device) {
    return mgr_.device_stream(device);
  }

  template <class Any>
  static void allocate(Any** ptr, size_t size, int device,
      cudaStream_t stream) {
    if (!try_allocate(reinterpret_cast<void**>(ptr), size, device, stream)) {
      LOG(FATAL) << "Out of memory: failed to allocate " << size
          << " bytes on device " << device;
    }
  }

  static void deallocate(void* ptr, int device, cudaStream_t stream) {
    mgr_.deallocate(ptr, device, stream);
  }

  static bool try_allocate(void** ptr, size_t size, int device,
      cudaStream_t stream) {
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
    ~Workspace() { mgr_.deallocate(ptr_, device_, stream_); }

    void* data() const {
      CHECK_NOTNULL(ptr_);
      return ptr_;
    }
    size_t size() const { return size_; }
    int device() const { return device_; }

    bool try_reserve(size_t size, int device, cudaStream_t stream) {
      bool status = true;
      if (size > size_ || ptr_ == NULL) {
        if (ptr_ != NULL) {
          mgr_.deallocate(ptr_, device_, stream_);
        }
        if (device != INVALID_DEVICE) {
          device_ = device;  // switch from default to specific one
        }
        status = mgr_.try_allocate(&ptr_, size, device_, stream);
        if (status) {
          CHECK_NOTNULL(ptr_);
          size_ = size;
          stream_ = stream;
        }
      }
      return status;
    }

    void reserve(size_t size, int device, cudaStream_t stream) {
      if (!try_reserve(size, device, stream)) {
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
      const int device = current_device();
      cudaStream_t stream = device_stream(device);
      return current_workspace(device)->try_reserve(size, device, stream);
    }
    void reserve(size_t size) {
      const int device = current_device();
      cudaStream_t stream = device_stream(device);
      current_workspace(device)->reserve(size, device, stream);
    }
    void release() {
      current_workspace(current_device())->release();
    }
    void* data() const {
      return current_workspace(current_device())->data();
    }
    size_t size() const {
      return current_workspace(current_device())->size();
    }

   private:
    shared_ptr<Workspace> current_workspace(int device) const;
    mutable vector<shared_ptr<Workspace> > workspaces_;
  };

 private:
  struct Manager {
    Manager();
    ~Manager();
    void lazy_init(int device);
    void GetInfo(size_t* free_mem, size_t* used_mem);
    void deallocate(void* ptr, int device, cudaStream_t stream);
    bool try_allocate(void** ptr, size_t size, int device, cudaStream_t);
    const char* pool_name() const;
    bool using_pool() const { return mode_ != CUDA_MALLOC; }
    void init(const std::vector<int>&, Mode, bool);
    cudaStream_t device_stream(int device);

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
    vector<cudaStream_t> device_streams_;

    static const unsigned int BIN_GROWTH;  ///< Geometric growth factor
    static const unsigned int MIN_BIN;  ///< Minimum bin
    static const unsigned int MAX_BIN;  ///< Maximum bin
    static const size_t MAX_CACHED_BYTES;  ///< Maximum aggregate cached bytes
  };

  static Manager mgr_;
  static const int INVALID_DEVICE;  ///< Default is invalid: CUB takes care

  static int current_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }
};

}  // namespace caffe

#endif

#endif
