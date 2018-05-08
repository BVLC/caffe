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
  ~OclDevice();

  const char* clGetErrorString(cl_int error);
#ifdef USE_CLFFT
  const char* clfftGetErrorString(clfftStatus status);
#endif  // USE_CLFFT

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
  bool is_beignet();
  virtual string name();
  virtual shared_ptr<DeviceProgram> CreateProgram();

  virtual void unlock_buffer(int_tp* lock_id);

  virtual void MallocMemHost(uint_tp size, void** ptr);
  virtual void FreeMemHost(void* ptr);
  virtual vptr<void> MallocMemDevice(uint_tp size, void** ptr, bool zero_copy);
  virtual void FreeMemDevice(vptr<void> ptr);
  virtual bool CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                             uint_tp size);

  void ocl_null_kernel(float arg, cl_event* event);

  virtual void memcpy(const uint_tp n, vptr<const void> x, vptr<void> y);
  virtual void memcpy(const uint_tp n, const void* x, vptr<void> y);
  virtual void memcpy(const uint_tp n, vptr<const void> x, void* y);

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
  virtual void gemv_half
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const half_fp alpha,
                 vptr<const half_fp> a,
                 vptr<const half_fp> x, const half_fp beta,
                 vptr<half_fp> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_half(const uint_tp n,
                         const half_fp alpha,
                         vptr<const half_fp> x,
                         vptr<half_fp> y,
                         const QuantizerValues* const alpha_quant = nullptr,
                         const QuantizerValues* const x_quant = nullptr,
                         const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_half(const uint_tp n, const half_fp alpha,
                    vptr<const half_fp> x, const half_fp beta, vptr<half_fp> y,
                    const QuantizerValues* const alpha_quant = nullptr,
                    const QuantizerValues* const x_quant = nullptr,
                    const QuantizerValues* const beta_quant = nullptr,
                    const QuantizerValues* const y_quant = nullptr);
  virtual void dot_half(const uint_tp n, vptr<const half_fp> x,
                        vptr<const half_fp> y, half_fp *out);
  virtual void asum_half(const uint_tp n, vptr<const half_fp> x, half_fp* y);
  virtual void scale_half(const uint_tp n, const half_fp alpha,
                          vptr<const half_fp> x, vptr<half_fp> y);
  virtual void scal_half(const uint_tp n, const half_fp alpha, vptr<half_fp> x);
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
#endif  // USE_DOUBLE
};


#endif  // USE_OPENCL

}  // namespace caffe

#endif  // CAFFE_BACKEND_OPENCL_OCL_DEVICE_HPP_
