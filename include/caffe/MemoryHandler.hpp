#ifndef CAFFE_MEMORYHANDLER_HPP_
#define CAFFE_MEMORYHANDLER_HPP_

#include <vector>
#include "common.hpp"

namespace caffe {

class MemoryHandler {
 public:
  enum PoolMode { NoPool, CnMemPool, CubPool };

#ifndef CPU_ONLY
  static void mallocGPU(void **ptr, size_t size,
                        cudaStream_t stream = cudaStreamDefault);
  static void freeGPU(void *ptr, cudaStream_t = cudaStreamDefault);
  static void registerStream(cudaStream_t stream);
#endif

  static const char* getName();
  static bool usingPool() {
    return mode_ != NoPool;
  }

  static void getInfo(size_t *free_mem, size_t *used_mem);

 private:
  static void initCNMEM(const std::vector<int>& gpus);
  static void init(const std::vector<int>&, PoolMode);
  static void destroy();

  friend class MemoryHandlerActivator;
  static bool using_pool_;
  static bool initialized_;
  static PoolMode mode_;
};

class MemoryHandlerActivator {
 public:
  MemoryHandlerActivator(const std::vector<int>& gpus,
                         MemoryHandler::PoolMode m = MemoryHandler::CnMemPool) {
    MemoryHandler::init(gpus, m);
  }

  ~MemoryHandlerActivator() {
      MemoryHandler::destroy();
  }
};

}  // namespace caffe

# endif
