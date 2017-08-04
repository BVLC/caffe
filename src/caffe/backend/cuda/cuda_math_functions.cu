#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#ifdef USE_CUDA

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA

void cuda_device::memcpy(const uint_tp N, vptr<void> X, vptr<void> Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y.get_cuda_ptr(), X.get_cuda_ptr(),
                          N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}


template<typename Dtype>
__global__ void set_kernel(const int_tp n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template<typename Dtype>
void cuda_device::set(const int_tp N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, alpha, Y);
}


template<typename Dtype>
__global__ void add_scalar_kernel(const int_tp n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template<>
void cuda_device::add_scalar(const int_tp N, const float alpha, vptr<float> Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N),
                                       CAFFE_CUDA_NUM_THREADS)(
      N, alpha, Y);
}

template<>
void cuda_device::add_scalar(const int_tp N, const double alpha, vptr<double> Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N),
                                        CAFFE_CUDA_NUM_THREADS)(
      N, alpha, Y);
}

template<typename Dtype>
__global__ void add_kernel(const int_tp n, const Dtype* a, const Dtype* b,
                           Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template<>
void cuda_device::add<float>(const int_tp N, vptr<float> a, vptr<float> b,
                          vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<>
void cuda_device::add<double>(const int_tp N, const vptr<double> a, const vptr<double> b,
                           vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<typename Dtype>
__global__ void sub_kernel(const int_tp n, const Dtype* a, const Dtype* b,
                           Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template<>
void cuda_device::sub<float>(const int_tp N, vptr<float> a, vptr<float> b,
                          vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<>
void cuda_device::sub<double>(const int_tp N, const vptr<double> a, const vptr<double> b,
                           vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<typename Dtype>
__global__ void mul_kernel(const int_tp n, const Dtype* a, const Dtype* b,
                           Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template<>
void cuda_device::mul<float>(const int_tp N, vptr<float> a, vptr<float> b,
                          vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<>
void cuda_device::mul<double>(const int_tp N, const vptr<double> a, const vptr<double> b,
                           vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<typename Dtype>
__global__ void div_kernel(const int_tp n, const Dtype* a, const Dtype* b,
                           Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template<>
void cuda_device::div<float>(const int_tp N, vptr<float> a, vptr<float> b,
                          vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<>
void cuda_device::div<double>(const int_tp N, const vptr<double> a, const vptr<double> b,
                           vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, b, y);
}

template<typename Dtype>
__global__ void abs_kernel(const int_tp n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template<>
void cuda_device::abs<float>(const int_tp N, vptr<float> a, vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template<>
void cuda_device::abs<double>(const int_tp N, const vptr<double> a, vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template<typename Dtype>
__global__ void exp_kernel(const int_tp n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template<>
void cuda_device::exp<float>(const int_tp N, vptr<float> a, vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template<>
void cuda_device::exp<double>(const int_tp N, const vptr<double> a, vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template<typename Dtype>
__global__ void log_kernel(const int_tp n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template<>
void cuda_device::log<float>(const int_tp N, vptr<float> a, vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template<>
void cuda_device::log<double>(const int_tp N, const vptr<double> a, vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template<typename Dtype>
__global__ void powx_kernel(const int_tp n, const Dtype* a, const Dtype alpha,
                            Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template<>
void cuda_device::powx<float>(const int_tp N, vptr<float> a, const float alpha,
                           vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, alpha, y);
}

template<>
void cuda_device::powx<double>(const int_tp N, const vptr<double> a, const double alpha,
                            vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double> CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int_tp n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void cuda_device::sqrt<float>(const int_tp N, vptr<float> a, vptr<float> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float>CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

template <>
void cuda_device::sqrt<double>(const int_tp N, const vptr<double> a, vptr<double> y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double>CUDA_KERNEL(CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS)(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));


void cuda_device::rng_uniform(const int_tp n, unsigned int* r) {  // NOLINT
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

void cuda_device::rng_uniform(const int_tp n, unsigned long long* r) {  // NOLINT
  CURAND_CHECK(curandGenerateLongLong(Caffe::curand_generator64(), r, n));
}

template<>
void cuda_device::rng_uniform<float>(const int_tp n, const float a, const float b,
                                  vptr<float> r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    cuda_device::scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    cuda_device::add_scalar(n, a, r);
  }
}

template<>
void cuda_device::rng_uniform<double>(const int_tp n, const double a,
                                   const double b, vptr<double> r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    cuda_device::scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    cuda_device::add_scalar(n, a, r);
  }
}

void cuda_device::rng_gaussian_float(const uint_tp n,
                      const float mu, const float sigma, vptr<float> r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r.get_cuda_ptr(),
                           n, mu, sigma));
}

template<>
void cuda_device::rng_gaussian(const int_tp n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}


#endif  // USE_CUDA

}  // namespace caffe
