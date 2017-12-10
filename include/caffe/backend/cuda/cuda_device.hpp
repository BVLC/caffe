#ifndef CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_
#define CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_

#include "caffe/common.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/device.hpp"

#include <string>

namespace caffe {

#ifdef USE_CUDA

class CudaDevice : public Device {
 public:
  explicit CudaDevice(uint_tp id, uint_tp list_id);
  ~CudaDevice();

  template <typename Dtype>
  void scal_str(const int_tp n, const Dtype alpha, vptr<Dtype> x,
                      cudaStream_t str);

  int_tp get_header_count();
  char** get_header_names();
  char** get_header_sources();

  virtual void Init();
  virtual bool CheckCapability(DeviceCapability cap);
  virtual bool CheckVendor(string vendor);
  virtual bool CheckType(string type);
  virtual void SwitchQueue(uint_tp id);
  virtual void get_threads(const vector<size_t>* work_size,
                           vector<size_t>* group,
                           vector<size_t>* local,
                           DeviceKernel* kernel,
                           bool auto_select);
  virtual void FinishQueues();
  virtual uint_tp num_queues();
  virtual bool is_host_unified();
  virtual string name();
  virtual shared_ptr<DeviceProgram> CreateProgram();

  virtual void MallocMemHost(void** ptr, uint_tp size);
  virtual void FreeMemHost(void* ptr);
  virtual vptr<void> MallocMemDevice(uint_tp size, void** ptr, bool zero_copy);
  virtual void FreeMemDevice(vptr<void> ptr);
  virtual bool CheckZeroCopy(vptr<void const> gpu_ptr,
                             void* cpu_ptr, uint_tp size);

  virtual void memcpy(const uint_tp n, vptr<const void> x, vptr<void> y);
  virtual void memcpy(const uint_tp n, const void* x, vptr<void> y);
  virtual void memcpy(const uint_tp n, vptr<const void> x, void* y);

  virtual void rng_uniform(const uint_tp n, vptr<uint32_t> r);
  virtual void rng_uniform(const uint_tp n, vptr<uint64_t> r);

  virtual void gemm_half
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const half_fp alpha, vptr<const half_fp> a,
                 vptr<const half_fp> b,
                 const half_fp beta,
                 vptr<half_fp> c);

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
                 const uint_tp n, const half_fp alpha,
                 vptr<const half_fp> a,
                 vptr<const half_fp> x, const half_fp beta,
                 vptr<half_fp> y);

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
                         const half_fp alpha,
                         vptr<const half_fp> x,
                         vptr<half_fp> y);

  virtual void axpy_float(const uint_tp n, const float alpha,
                          vptr<const float> x, vptr<float> y);

  virtual void axpy_double(const uint_tp n, const double alpha,
                          vptr<const double> x, vptr<double> y);

  virtual void axpby_half(const uint_tp n, const half_fp alpha,
                     vptr<const half_fp> x,
                     const half_fp beta, vptr<half_fp> y);

  virtual void axpby_float(const uint_tp n, const float alpha,
                     vptr<const float> x, const float beta, vptr<float> y);

  virtual void axpby_double(const uint_tp n, const double alpha,
                     vptr<const double> x, const double beta, vptr<double> y);

  virtual void rng_uniform_float(const uint_tp n, const float a, const float b,
                                 vptr<float> r);

  virtual void rng_uniform_double(const uint_tp n, const double a,
                                  const double b, vptr<double> r);

  virtual void rng_gaussian_float(const uint_tp n, const float mu,
                                  const float sigma, vptr<float> r);

  virtual void rng_gaussian_double(const uint_tp n, const double mu,
                                   const double sigma, vptr<double> r);

  virtual void dot_half(const uint_tp n, vptr<const half_fp> x,
                      vptr<const half_fp> y, half_fp *out);

  virtual void dot_float(const uint_tp n, vptr<const float> x,
                         vptr<const float> y, float *out);

  virtual void dot_double(const uint_tp n, vptr<const double> x,
                          vptr<const double> y, double *out);

  virtual void asum_half(const uint_tp n, vptr<const half_fp> x,
                         half_fp* y);

  virtual void asum_float(const uint_tp n, vptr<const float> x, float* y);

  virtual void asum_double(const uint_tp n, vptr<const double> x, double* y);

  virtual void scal_half(const uint_tp n, const half_fp alpha,
                         vptr<half_fp> x);

  virtual void scal_float(const uint_tp n, const float alpha, vptr<float> x);

  virtual void scal_double(const uint_tp n, const double alpha, vptr<double> x);

  virtual void scale_half(const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> x,
                          vptr<half_fp> y);

  virtual void scale_float(const uint_tp n, const float alpha,
                           vptr<const float> x, vptr<float> y);

  virtual void scale_double(const uint_tp n, const double alpha,
                            vptr<const double> x, vptr<double> y);

 private:
  void ReadHeaders();

  vector<char*> cuda_headers_;
  vector<char*> cuda_header_sources_;
};

#endif  // USE_CUDA

}  // namespace caffe

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_
