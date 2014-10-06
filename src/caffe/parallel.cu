#include <cuda_runtime.h>
#include <stdio.h>
#include "caffe/parallel.hpp"

namespace caffe {

template<typename Dtype>
__global__
void GPUSyncKernel(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  Dtype d = gpu[off + i] - last[off + i];
  gpu[off + i] = last[off + i] = chunk[i] + d;
  chunk[i] = d;
}

template<typename Dtype>
void GPUSync_kernel(Dtype* gpu, Dtype* last, Dtype* chunk, size_t off, cudaStream_t& stream) {
  int threadsPerBlock = 256; // TODO bench
  int blocksPerGrid = GPUSync<Dtype>::CHUNK / threadsPerBlock;
  GPUSyncKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(gpu, last, chunk, off);
  CUDA_POST_KERNEL_CHECK;
}

template void GPUSync_kernel<float>(float* gpu, float* last, float* chunk, size_t off, cudaStream_t& stream);
template void GPUSync_kernel<double>(double* gpu, double* last, double* chunk, size_t off, cudaStream_t& stream);
}
