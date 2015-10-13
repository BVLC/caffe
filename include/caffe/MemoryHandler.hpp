#ifndef CAFFE_MEMORYHANDLER_HPP_
#define CAFFE_MEMORYHANDLER_HPP_

#include "common.hpp"

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
  static void init(const std::vector<int>& gpus, bool use_pool=true);
  static void destroy();

  friend class MemoryHandlerActivator;
  static bool using_pool_;
  static bool initialized_;


};

class MemoryHandlerActivator {
 public:
  MemoryHandlerActivator(const std::vector<int>& gpus, 
			 bool use_pool = true) {
    MemoryHandler::init(gpus, use_pool && gpus.size() > 0);
  }
  ~MemoryHandlerActivator() {
      MemoryHandler::destroy();
  }
};

}  // namespace caffe

# endif
