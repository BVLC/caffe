#include <cuda_runtime.h>
#include <stdio.h>
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Dtype>
__global__
void GPUSyncKernel(Dtype* gpu, Dtype* chunk1, Dtype* chunk2, size_t off) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  Dtype d = gpu[off + i] - chunk1[i];
  gpu[off + i] = chunk2[i] + d;
  chunk2[i] = d;
}

template<typename Dtype>
void GPUSync_kernel(Dtype* gpu, Dtype* chunk1, Dtype* chunk2, size_t off) {
  int threadsPerBlock = 256; // TODO bench
  int blocksPerGrid = GPUSync<Dtype>::CHUNK / threadsPerBlock;
  GPUSyncKernel<<<blocksPerGrid, threadsPerBlock>>>(gpu, chunk1, chunk2, off);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation

template void GPUSync_kernel<float>(float* gpu, float* chunk1, float* chunk2, size_t off);
template void GPUSync_kernel<double>(double* gpu, double* chunk1, double* chunk2, size_t off);
}
