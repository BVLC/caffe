#ifndef CAFFE_GPU_MEMORY_POOL_HPP_
#define CAFFE_GPU_MEMORY_POOL_HPP_

namespace caffe {
void set_gpu_memory_pool(size_t memory_bytes);
void set_cpu_memory_pool(size_t memory_bytes);
} // namespace caffe

#endif // CAFFE_GPU_MEMORY_POOL_HPP_
