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

  virtual void MallocMemHost(uint_tp size, void** ptr);
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

#ifdef USE_HALF
  virtual void gemm_half
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const half_fp alpha, vptr<const half_fp> a,
                 vptr<const half_fp> b,
                 const half_fp beta,
                 vptr<half_fp> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
#endif  // USE_HALF

#ifdef USE_SINGLE
  virtual void gemm_float
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const float alpha, vptr<const float> a,
                 vptr<const float> b,
                 const float beta, vptr<float> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
  virtual void gemv_float
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const float alpha,
                 vptr<const float> a,
                 vptr<const float> x, const float beta,
                 vptr<float> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_float(const uint_tp n, const float alpha,
                          vptr<const float> x, vptr<float> y,
                          const QuantizerValues* const alpha_quant = nullptr,
                          const QuantizerValues* const x_quant = nullptr,
                          const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_float(const uint_tp n, const float alpha,
                          vptr<const float> x, const float beta, vptr<float> y,
                          const QuantizerValues* const alpha_quant = nullptr,
                          const QuantizerValues* const x_quant = nullptr,
                          const QuantizerValues* const beta_quant = nullptr,
                          const QuantizerValues* const y_quant = nullptr);
  virtual void dot_float(const uint_tp n, vptr<const float> x,
                         vptr<const float> y, float *out);
  virtual void asum_float(const uint_tp n, vptr<const float> x, float* y);
  virtual void scale_float(const uint_tp n, const float alpha,
                          vptr<const float> x, vptr<float> y);
  virtual void scal_float(const uint_tp n, const float alpha, vptr<float> x);

  virtual void rng_uniform_float(const uint_tp n, const float a,
                                 const float b, vptr<float> r);
  virtual void rng_gaussian_float(const uint_tp n, const float mu,
                                  const float sigma, vptr<float> r);
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
  virtual void gemm_double
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const double alpha, vptr<const double> a,
                 vptr<const double> b,
                 const double beta, vptr<double> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
  virtual void gemv_double
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const double alpha,
                 vptr<const double> a,
                 vptr<const double> x, const double beta,
                 vptr<double> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_double(const uint_tp n, const double alpha,
                          vptr<const double> x, vptr<double> y,
                          const QuantizerValues* const alpha_quant = nullptr,
                          const QuantizerValues* const x_quant = nullptr,
                          const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_double(const uint_tp n, const double alpha,
                       vptr<const double> x, const double beta, vptr<double> y,
                       const QuantizerValues* const alpha_quant = nullptr,
                       const QuantizerValues* const x_quant = nullptr,
                       const QuantizerValues* const beta_quant = nullptr,
                       const QuantizerValues* const y_quant = nullptr);
  virtual void dot_double(const uint_tp n, vptr<const double> x,
                          vptr<const double> y, double *out);
  virtual void asum_double(const uint_tp n, vptr<const double> x, double* y);
  virtual void scale_double(const uint_tp n, const double alpha,
                            vptr<const double> x, vptr<double> y);
  virtual void scal_double(const uint_tp n, const double alpha,
                           vptr<double> x);

  virtual void rng_uniform_double(const uint_tp n, const double a,
                                  const double b, vptr<double> r);
  virtual void rng_gaussian_double(const uint_tp n, const double mu,
                                   const double sigma, vptr<double> r);
#endif  // USE_DOUBLE

 private:
  void ReadHeaders();

  vector<char*> cuda_headers_;
  vector<char*> cuda_header_sources_;
  int cuda_minor_;
  int cuda_major_;
};

#endif  // USE_CUDA

}  // namespace caffe

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_
