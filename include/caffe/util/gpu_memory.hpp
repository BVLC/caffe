#ifndef CAFFE_UTIL_GPU_MEMORY_HPP_
#define CAFFE_UTIL_GPU_MEMORY_HPP_

#include <vector>
#include "caffe/common.hpp"

namespace caffe {

class gpu_memory {
 public:
  enum PoolMode { NoPool, CnMemPool, CubPool };

  static const char* getPoolName();
  static bool usingPool() {
    return mode_ != NoPool;
  }

  class arena {
   public:
    arena(const std::vector<int>& gpus, PoolMode m = CnMemPool) {
      init(gpus, m);
    }
    ~arena() {
      destroy();
     }
  };

 private:
    static void init(const std::vector<int>&, PoolMode);
    static void destroy();

    static bool initialized_;
    static PoolMode mode_;

#ifndef CPU_ONLY

 public:
  static void allocate(void **ptr, size_t size,
                       cudaStream_t stream = cudaStreamDefault);
  static void deallocate(void *ptr, cudaStream_t = cudaStreamDefault);
  static void registerStream(cudaStream_t stream);
  static void getInfo(size_t *free_mem, size_t *used_mem);

 private:
    static void initCNMEM(const std::vector<int>& gpus);
#endif
};

}  // namespace caffe

# endif
