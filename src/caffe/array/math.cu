#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/array/array.hpp"
#include "caffe/array/math.hpp"

namespace caffe {

template <typename T, typename F>
__global__ void unary_kernel(const int n, const T *x, T *y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = F::eval(x[index]);
  }
}
template<typename T, typename F>
void Unary<T, F>::eval_gpu(const Array<T> & a, Array<T> * c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(a.shape(), c->shape()) << "Shape does not match!";
  const int N = count(a.shape());
  const T * pa = a.gpu_data();
  T * pc = c->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  unary_kernel<T, F> <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, pa, pc);
  CUDA_POST_KERNEL_CHECK;
}


template <typename T, typename F>
__global__ void binary_kernel(const int n, const T *a, const T *b, T *r) {
  CUDA_KERNEL_LOOP(index, n) {
    r[index] = F::eval(a[index], b[index]);
  }
}
template <typename T, typename F>
__global__ void binary_kernel(const int n, T a, const T *b, T *r) {
  CUDA_KERNEL_LOOP(index, n) {
    r[index] = F::eval(a, b[index]);
  }
}
template <typename T, typename F>
__global__ void binary_kernel(const int n, const T *a, T b, T *r) {
  CUDA_KERNEL_LOOP(index, n) {
    r[index] = F::eval(a[index], b);
  }
}
template<typename T, typename F>
void Binary<T, F>::eval_gpu(const Array<T> &a, const Array<T> &b, Array<T> *c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(a.shape(), c->shape()) << "Shape does not match!";
  CHECK_EQ(b.shape(), c->shape()) << "Shape does not match!";
  const int N = count(a.shape());
  const T * pa = a.gpu_data(), * pb = b.gpu_data();
  T * pc = c->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  binary_kernel<T, F> <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, pa, pb, pc);
  CUDA_POST_KERNEL_CHECK;
}
template<typename T, typename F>
void Binary<T, F>::eval_gpu(T a, const Array<T> & b, Array<T> * c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(b.shape(), c->shape()) << "Shape does not match!";
  const int N = count(b.shape());
  const T * pb = b.gpu_data();
  T * pc = c->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  binary_kernel<T, F> <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, pb, pc);
  CUDA_POST_KERNEL_CHECK;
}
template<typename T, typename F>
void Binary<T, F>::eval_gpu(const Array<T> & a, T b, Array<T> * c) {
  CHECK(!!c) << "Output array does not exist!";
  CHECK_EQ(a.shape(), c->shape()) << "Shape does not match!";
  const int N = count(a.shape());
  const T * pa = a.gpu_data();
  T * pc = c->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  binary_kernel<T, F> <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, pa, b, pc);
  CUDA_POST_KERNEL_CHECK;
}
extern __shared__ char shared_data[];
template <typename T, typename F>
__global__ void reduction_kernel(const int n, const T *x, T *y) {
  // Reduce the data
  T * sdata = reinterpret_cast<T*>(shared_data);

  const int tid = threadIdx.x;
  const int i0 = blockIdx.x*blockDim.x + tid, D = blockDim.x*gridDim.x;
  sdata[tid] = i0 < n ? x[i0] : T(0);
  for (int i = i0 + D; i < n; i += D)
    sdata[tid] = F::eval(sdata[tid], x[i]);
  __syncthreads();

  // Reduce the block
  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s && i0 < n)
      sdata[tid] = F::eval(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0)
    y[blockIdx.x] = sdata[0];
}
template<typename T, typename F>
T Reduction<T, F>::eval_gpu(const Array<T> & a) {
  const int N = count(a.shape());
  CHECK_GT(N, 0) << "At least one element required for reduction";
  const T * pa = a.gpu_data();
  int NB = CAFFE_GET_BLOCKS(N), NT = CAFFE_CUDA_NUM_THREADS;
  if (NB > 32) NB = 32;
  Array<T> tmp(temporaryMemory((NB+10)*sizeof(T)), make_shape(NB));
  // NOLINT_NEXT_LINE(whitespace/operators)
  reduction_kernel<T, F> <<<NB, NT, (NT+32)*sizeof(T)>>>(
      N, pa, tmp.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  const T * ptmp = tmp.cpu_data();
  T r = ptmp[0];
  for (int i = 1; i < NB; i++)
    r = F::eval(ptmp[i], r);
  return r;
}

INSTANTIATE_ALL;

}  // namespace caffe
