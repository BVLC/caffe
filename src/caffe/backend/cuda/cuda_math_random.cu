#ifdef USE_CUDA

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "caffe/backend/cuda/cuda_device.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

namespace caffe {


void CudaDevice::rng_uniform(const uint_tp n, vptr<uint32_t> r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r.get_cuda_ptr(), n));
}

void CudaDevice::rng_uniform(const uint_tp n, vptr<uint64_t> r) {
  CURAND_CHECK(curandGenerateLongLong(Caffe::curand_generator64(),
                   reinterpret_cast<unsigned long long*>(r.get_cuda_ptr()), n));
}

#ifdef USE_SINGLE
void CudaDevice::rng_uniform_float(const uint_tp n, const float a,
                                    float b,
                                    vptr<float> r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(),
                                     r.get_cuda_ptr(), n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    CudaDevice::scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    CudaDevice::add_scalar(n, a, r);
  }
}
void CudaDevice::rng_gaussian_float(const uint_tp n, const float mu,
                                     const float sigma, vptr<float> r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r.get_cuda_ptr(),
                           n, mu, sigma));
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
void CudaDevice::rng_uniform_double(const uint_tp n, const double a,
                                    const double b,
                                    vptr<double> r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(),
                                           r.get_cuda_ptr(), n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    CudaDevice::scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    CudaDevice::add_scalar(n, a, r);
  }
}
void CudaDevice::rng_gaussian_double(const uint_tp n, const double mu,
                                      const double sigma, vptr<double> r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r.get_cuda_ptr(),
                                 n, mu, sigma));
}
#endif  // USE_DOUBLE


}  // namespace caffe
#endif  // USE_CUDA
