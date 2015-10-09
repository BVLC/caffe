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
#ifndef CPU_ONLY
  static void mallocGPU(void **ptr, size_t size,
                        cudaStream_t stream = cudaStreamDefault);
  static void freeGPU(void *ptr, cudaStream_t = cudaStreamDefault);
  static void registerStream(cudaStream_t stream);
#endif

  static bool usingPool() {
    return using_pool_;
  }

  static void getInfo(size_t *free_mem, size_t *used_mem);

 private:
  static void init(const std::vector<int>& gpus_, bool use_pool=true);
  static void destroy();

  friend class MemoryHandlerActivator;
  static bool using_pool_;
  static bool initialized_;


};

class MemoryHandlerActivator {
 public:
  explicit MemoryHandlerActivator(const std::vector<int>& gpus)
            : using_pool_(false) {
    if (gpus.size() > 0) {
#ifdef USE_CNMEM
      using_pool_ = true;
#endif
      MemoryHandler::init(gpus, using_pool_);
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
