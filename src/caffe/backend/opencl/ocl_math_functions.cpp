#include "caffe/backend/opencl/ocl_device.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/common.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/backend/dev_ptr.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

namespace caffe {

#ifdef USE_OPENCL

void OclDevice::memcpy(const uint_tp n, vptr<const void> x, vptr<void> y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  if (x.get_ocl_mem() != y.get_ocl_mem()) {
    clEnqueueCopyBuffer(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                        y.get_ocl_mem(), x.get_ocl_off(), y.get_ocl_off(), n, 0,
                        NULL, NULL);
  }
}

void OclDevice::memcpy(const uint_tp n, const void* x, vptr<void> y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  if (x != nullptr) {
    clEnqueueWriteBuffer(ctx.get_queue().handle().get(), y.get_ocl_mem(),
                         CL_TRUE, y.get_ocl_off(), n, x, 0, NULL, NULL);
  }
}

void OclDevice::memcpy(const uint_tp n, vptr<const void> x, void* y) {
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->id());
  if (y != NULL) {
    clEnqueueReadBuffer(ctx.get_queue().handle().get(), x.get_ocl_mem(),
                        CL_TRUE, x.get_ocl_off(), n, y, 0, NULL, NULL);
  }
}

void OclDevice::rng_uniform(const uint_tp n, vptr<uint32_t> r) {
  vector<uint32_t> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  this->memcpy(sizeof(uint32_t) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_uniform(const uint_tp n, vptr<uint64_t> r) {
  vector<uint64_t> random(n);  //NOLINT
  caffe_rng_uniform(n, &random[0]);
  this->memcpy(sizeof(uint64_t) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_uniform_half(const uint_tp n, const half_float::half a,
                                  const half_float::half b,
                                  vptr<half_float::half> r) {
  vector<half_float::half> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(half_float::half) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_uniform_float(const uint_tp n, const float a,
                                   const float b, vptr<float> r) {
  vector<float> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(float) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_uniform_double(const uint_tp n, const double a,
                                   const double b, vptr<double> r) {
  vector<double> random(n);  // NOLINT
  caffe_rng_uniform(n, a, b, &random[0]);
  this->memcpy(sizeof(double) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_gaussian_half(const uint_tp n, const half_float::half mu,
                                    const half_float::half sigma,
                                    vptr<half_float::half> r) {
  vector<half_float::half> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  this->memcpy(sizeof(half_float::half) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_gaussian_float(const uint_tp n, const float mu,
                                   const float sigma, vptr<float> r) {
  vector<float> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  this->memcpy(sizeof(float) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_gaussian_double(const uint_tp n, const double mu,
                                     const double sigma, vptr<double> r) {
  vector<double> random(n);  // NOLINT
  caffe_rng_gaussian(n, mu, sigma, &random[0]);
  this->memcpy(sizeof(double) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_bernoulli_half(const uint_tp n, const half_float::half p,
                                    vptr<int> r) {
  vector<half_float::half> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(half_float::half) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_bernoulli_float(const uint_tp n, const float p,
                                     vptr<int> r) {
  vector<float> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(float) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_bernoulli_double(const uint_tp n, const double p,
                                      vptr<int> r) {
  vector<double> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(double) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_bernoulli_half(const uint_tp n, const half_float::half p,
                                    vptr<unsigned int> r) {
  vector<half_float::half> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(half_float::half) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_bernoulli_float(const uint_tp n, const float p,
                                     vptr<unsigned int> r) {
  vector<float> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(float) * n, &random[0], vptr<void>(r));
}

void OclDevice::rng_bernoulli_double(const uint_tp n, const double p,
                                      vptr<unsigned int> r) {
  vector<double> random(n);  // NOLINT
  caffe_rng_bernoulli(n, p, &random[0]);
  this->memcpy(sizeof(double) * n, &random[0], vptr<void>(r));
}

#endif  // USE_OPENCL

}  // namespace caffe

