#ifndef CAFFE_MEMORYHANDLER_HPP_
#define CAFFE_MEMORYHANDLER_HPP_

#include "common.hpp"

#ifdef USE_CNMEM
// cuMEM integration
#include <cnmem.h>
#endif

namespace caffe {

class MemoryHandler {
 public:
  static MemoryHandler& Get();
#ifndef CPU_ONLY
  static void mallocGPU(void **ptr, size_t size,
                        cudaStream_t stream = cudaStreamDefault);
  static void freeGPU(void *ptr, cudaStream_t = cudaStreamDefault);
  static void registerStream(cudaStream_t stream);
#endif

  static bool usingPool() {
#ifdef USE_CNMEM
    return using_pool_;
#else
    return false;
#endif
  }
  static void getInfo(size_t *free_mem, size_t *used_mem);

  ~MemoryHandler() { }

 private:
  MemoryHandler() {}

  static void usePool() { 
#ifdef USE_CNMEM
    using_pool_ = true; 
#endif
  }

  static void setGPUs(const std::vector<int>& gpus) { gpus_ = gpus; }
  static void Init();
  static void destroy();

#ifndef CPU_ONLY
  void allocate_memory(void **ptr, size_t size, cudaStream_t stream);
  void free_memory(void *ptr, cudaStream_t stream);
#endif
  DISABLE_COPY_AND_ASSIGN(MemoryHandler);
  friend class MemoryHandlerActivator;
  static bool using_pool_;
  static bool initialized_;
  static std::vector<int> gpus_;
};

class MemoryHandlerActivator {
 public:
  explicit MemoryHandlerActivator(const std::vector<int>& gpus)
            : using_pool_(false) {
    if (gpus.size() > 0) {
      using_pool_ = true;
#ifdef USE_CNMEM
      MemoryHandler::usePool();
#endif
      MemoryHandler::setGPUs(gpus);
      MemoryHandler::Init();
#ifndef CPU_ONLY
      void* temp;
      MemoryHandler::mallocGPU(&temp, 4);
      MemoryHandler::freeGPU(temp);
#endif
    }
  }
  ~MemoryHandlerActivator() {
    if (using_pool_) {
      MemoryHandler::destroy();
    }
  }
 private:
  int using_pool_;
};

}  // namespace caffe

# endif
