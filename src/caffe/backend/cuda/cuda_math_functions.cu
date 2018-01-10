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

#ifdef USE_CUDA
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA

void CudaDevice::memcpy(const uint_tp n, vptr<const void> x, vptr<void> y) {
  if (x.get_cuda_ptr() != y.get_cuda_ptr()) {
    CHECK(x.get_cuda_ptr());
    CHECK(y.get_cuda_ptr());
    CUDA_CHECK(cudaMemcpy(y.get_cuda_ptr(), x.get_cuda_ptr(),
                          n, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

void CudaDevice::memcpy(const uint_tp n, const void* x, vptr<void> y) {
  if (x != y.get_cuda_ptr()) {
    CHECK(x);
    CHECK(y.get_cuda_ptr());
    CUDA_CHECK(cudaMemcpy(y.get_cuda_ptr(), x,
                          n, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

void CudaDevice::memcpy(const uint_tp n, vptr<const void> x, void* y) {
  if (x.get_cuda_ptr() != y) {
    CHECK(x.get_cuda_ptr());
    CHECK(y);
    CUDA_CHECK(cudaMemcpy(y, x.get_cuda_ptr(),
                          n, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

#endif  // USE_CUDA

}  // namespace caffe
