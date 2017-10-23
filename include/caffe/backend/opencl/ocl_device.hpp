#ifndef CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_
#define CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_

#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"

namespace caffe {

#ifdef USE_OPENCL

class OclDevice : public Device {
 public:
  explicit OclDevice(uint_tp id, uint_tp list_id);

  const char* clGetErrorString(cl_int error);
#ifdef USE_CLFFT
  const char* clfftGetErrorString(clfftStatus status);
#endif  // USE_CLFFT

  virtual void Init();
  virtual bool CheckCapability(string cap);
  virtual bool CheckVendor(string vendor);
  virtual bool CheckType(string type);
  virtual void SwitchQueue(uint_tp id);
  virtual uint_tp current_queue_id();
  virtual void get_threads(const vector<size_t>* work_size,
                           vector<size_t>* local,
                           vector<size_t>* group,
                           DeviceKernel* kernel,
                           bool auto_select);
  virtual void FinishQueues();
  virtual uint_tp num_queues();
  virtual bool is_host_unified();
  bool is_beignet();
  virtual string name();
  virtual shared_ptr<DeviceProgram> CreateProgram();

  virtual void MallocMemHost(void** ptr, uint_tp size);
  virtual void FreeMemHost(void* ptr);
  virtual vptr<void> MallocMemDevice(uint_tp size, void** ptr, bool zero_copy);
  virtual void FreeMemDevice(vptr<void> ptr);
  virtual bool CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                             uint_tp size);

  virtual void memcpy(const uint_tp n, vptr<const void> x, vptr<void> y);
  virtual void memcpy(const uint_tp n, const void* x, vptr<void> y);
  virtual void memcpy(const uint_tp n, vptr<const void> x, void* y);

  virtual void rng_uniform(const uint_tp n, vptr<uint32_t> r);
  virtual void rng_uniform(const uint_tp n, vptr<uint64_t> r);

  virtual void gemm_half
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const half_float::half alpha, vptr<const half_float::half> a,
                 vptr<const half_float::half> b,
                 const half_float::half beta,
                 vptr<half_float::half> c);

  virtual void gemm_float
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const float alpha, vptr<const float> a,
                 vptr<const float> b,
                 const float beta, vptr<float> c);

  virtual void gemm_double
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const double alpha, vptr<const double> a,
                 vptr<const double> b,
                 const double beta, vptr<double> c);

  virtual void gemv_half
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const half_float::half alpha,
                 vptr<const half_float::half> a,
                 vptr<const half_float::half> x, const half_float::half beta,
                 vptr<half_float::half> y);

  virtual void gemv_float
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const float alpha,
                 vptr<const float> a,
                 vptr<const float> x, const float beta,
                 vptr<float> y);

  virtual void gemv_double
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const double alpha,
                 vptr<const double> a,
                 vptr<const double> x, const double beta,
                 vptr<double> y);

  virtual void axpy_half(const uint_tp n,
                         const half_float::half alpha,
                         vptr<const half_float::half> x,
                         vptr<half_float::half> y);

  virtual void axpy_float(const uint_tp n, const float alpha,
                          vptr<const float> x, vptr<float> y);

  virtual void axpy_double(const uint_tp n, const double alpha,
                          vptr<const double> x, vptr<double> y);

  virtual void axpby_half(const uint_tp n, const half_float::half alpha,
                     vptr<const half_float::half> x,
                     const half_float::half beta, vptr<half_float::half> y);

  virtual void axpby_float(const uint_tp n, const float alpha,
                     vptr<const float> x, const float beta, vptr<float> y);

  virtual void axpby_double(const uint_tp n, const double alpha,
                     vptr<const double> x, const double beta, vptr<double> y);

  virtual void rng_uniform_half(const uint_tp n, const half_float::half a,
                        const half_float::half b, vptr<half_float::half> r);

  virtual void rng_uniform_float(const uint_tp n, const float a, const float b,
                                 vptr<float> r);

  virtual void rng_uniform_double(const uint_tp n, const double a,
                                  const double b, vptr<double> r);

  virtual void rng_gaussian_half(const uint_tp n, const half_float::half mu,
                    const half_float::half sigma, vptr<half_float::half> r);

  virtual void rng_gaussian_float(const uint_tp n, const float mu,
                                  const float sigma, vptr<float> r);

  virtual void rng_gaussian_double(const uint_tp n, const double mu,
                                   const double sigma, vptr<double> r);

  virtual void rng_bernoulli_half(const uint_tp n, const half_float::half p,
                                  vptr<int> r);

  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int> r);

  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int> r);

  virtual void rng_bernoulli_half(const uint_tp n, const half_float::half p,
                                  vptr<unsigned int> r);

  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<unsigned int> r);

  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<unsigned int> r);

  virtual void dot_half(const uint_tp n, vptr<const half_float::half> x,
                        vptr<const half_float::half> y, half_float::half *out);

  virtual void dot_float(const uint_tp n, vptr<const float> x,
                         vptr<const float> y, float *out);

  virtual void dot_double(const uint_tp n, vptr<const double> x,
                          vptr<const double> y, double *out);

  virtual void asum_half(const uint_tp n, vptr<const half_float::half> x,
                         half_float::half* y);

  virtual void asum_float(const uint_tp n, vptr<const float> x, float* y);

  virtual void asum_double(const uint_tp n, vptr<const double> x, double* y);

  virtual void scal_half(const uint_tp n, const half_float::half alpha,
                         vptr<half_float::half> x);

  virtual void scal_float(const uint_tp n, const float alpha, vptr<float> x);

  virtual void scal_double(const uint_tp n, const double alpha, vptr<double> x);

  virtual void scale_half(const uint_tp n, const half_float::half alpha,
                          vptr<const half_float::half> x,
                          vptr<half_float::half> y);

  virtual void scale_float(const uint_tp n, const float alpha,
                           vptr<const float> x, vptr<float> y);

  virtual void scale_double(const uint_tp n, const double alpha,
                            vptr<const double> x, vptr<double> y);
};


#endif  // USE_OPENCL

}  // namespace caffe

#endif  // CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_
