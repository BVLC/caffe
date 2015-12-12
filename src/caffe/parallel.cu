#include <cuda_runtime.h>
#include <stdio.h>
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Dtype>
__global__
void sync_master_kernel(Dtype* gpu, Dtype** grds, size_t* offs, //
                        int batch_start, int batch_count) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for (int b = 0; b < batch_count; ++b) {
    // Index in queue
    int q = (batch_start + b) & (IBChannel::FRAMES - 1);
    gpu[offs[q] + i] += grds[q][i];
  }
}

template<typename Dtype>
void sync_master_kernel(Dtype* gpu, Dtype** grds, size_t* offs, //
                        int batch_start, int batch_count, //
                        const cudaStream_t& stream, size_t chunk) {
  int threadsPerBlock = 256;  // TODO bench
  int blocksPerGrid = chunk / threadsPerBlock;
  sync_master_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      gpu, grds, offs, batch_start, batch_count);
  CUDA_POST_KERNEL_CHECK;
}

template void sync_master_kernel<float>(float* gpu, float** grds, size_t* offs,
                                        int batch_start, int batch_count, //
                                        const cudaStream_t& stream, size_t chunk);
template void sync_master_kernel<double>(double* gpu, double** grds,
                                         size_t* offs, //
                                         int batch_start, int batch_count, //
                                         const cudaStream_t& stream, size_t chunk);

//

template<typename Dtype>
__global__
void sync_worker_kernel(Dtype* gpu, Dtype* last, Dtype** pos, size_t* offs,
                        Dtype** grads, uint8_t* get_grads,
                        int batch_start, int batch_count) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for (int b = 0; b < batch_count; ++b) {
    // Index in queue
    int q = (batch_start + b) & (IBChannel::FRAMES - 1);
    Dtype d = gpu[offs[q] + i] - last[offs[q] + i];
    if(get_grads[q]) {
      gpu[offs[q] + i] = last[offs[q] + i] = pos[q][i] + d;
      grads[q][i] = d;  // Warn: pos and grads can be same, keep assignment last
    } else {
      last[offs[q] + i] = pos[q][i];
      gpu[offs[q] + i] = pos[q][i] + d;
    }
  }
}

template<typename Dtype>
void sync_worker_kernel(Dtype* gpu, Dtype* last, Dtype** pos, size_t* offs,
                        Dtype** grads, uint8_t* get_grads,
                        int batch_start, int batch_count,
                        const cudaStream_t& stream, size_t chunk) {
  int threadsPerBlock = 64;  // TODO bench
  int blocksPerGrid = chunk / threadsPerBlock;
  sync_worker_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
      gpu, last, pos, offs, grads, get_grads, batch_start, batch_count);
  CUDA_POST_KERNEL_CHECK;
}

template void sync_worker_kernel<float>(float* gpu, float* last, float** pos,
                                        size_t* offs,
                                        float** grads, uint8_t* get_grads,
                                        int batch_start, int batch_count,
                                        const cudaStream_t& stream, size_t chunk);

template void sync_worker_kernel<double>(double* gpu, double* last,
                                         double** pos, size_t* offs,
                                         double** grads, uint8_t* get_grads,
                                         int batch_start, int batch_count,
                                         const cudaStream_t& stream, size_t chunk);

}
