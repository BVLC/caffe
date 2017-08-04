#ifndef CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_
#define CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_

#include "caffe/backend/cuda/caffe_cuda.hpp"
#include "caffe/backend/device.hpp"

#include <string>

namespace caffe {

#ifdef USE_CUDA

class cuda_device : public device {
 public:
  explicit cuda_device(uint_tp id, uint_tp list_id);
  template <typename Dtype>
  void scal_str(const int_tp N, const Dtype alpha, vptr<Dtype> X,
                      cudaStream_t str);

  virtual void Init();
  virtual bool CheckCapability(std::string cap);
  virtual bool CheckVendor(std::string vendor);
  virtual bool CheckType(std::string type);
  virtual void SwitchQueue(uint_tp id);
  virtual uint_tp current_queue_id();
  virtual uint_tp workgroup_size(uint_tp id);
  virtual void FinishQueues();
  virtual uint_tp num_queues();
  virtual std::string name();

  virtual void memcpy(const uint_tp N, vptr<void> X, vptr<void> Y);
  virtual void memset(const uint_tp N, const int_tp alpha, vptr<void> X);
  virtual void rng_uniform(const uint_tp n, vptr<uint32_t>* r);
  virtual void rng_uniform(const uint_tp n, vptr<uint64_t>* r);

  virtual void gemm_half
                (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const half_float::half alpha, vptr<half_float::half> A,
                 vptr<half_float::half> B,
                 const half_float::half beta,
                 vptr<half_float::half> C);

  virtual void gemm_float
                (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const float alpha, vptr<float> A,
                 vptr<float> B,
                 const float beta, vptr<float> C);

  virtual void gemm_double
                (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const double alpha, vptr<double> A,
                 vptr<double> B,
                 const double beta, vptr<double> C);

  virtual void gemv_half
                (const CBLAS_TRANSPOSE TransA, const uint_tp M,
                 const uint_tp N, const half_float::half alpha,
                 vptr<half_float::half> A,
                 vptr<half_float::half> x, const half_float::half beta,
                 vptr<half_float::half> y);

  virtual void gemv_float
                (const CBLAS_TRANSPOSE TransA, const uint_tp M,
                 const uint_tp N, const float alpha,
                 vptr<float> A,
                 vptr<float> x, const float beta,
                 vptr<float> y);

  virtual void gemv_double
                (const CBLAS_TRANSPOSE TransA, const uint_tp M,
                 const uint_tp N, const double alpha,
                 vptr<double> A,
                 vptr<double> x, const double beta,
                 vptr<double> y);

  virtual void axpy_half(const uint_tp N,
                         const half_float::half alpha,
                         vptr<half_float::half> X,
                         vptr<half_float::half> Y);

  virtual void axpy_float(const uint_tp N, const float alpha,
                          vptr<float> X, vptr<float> Y);

  virtual void axpy_double(const uint_tp N, const double alpha,
                          vptr<double> X, vptr<double> Y);

  virtual void axpby_half(const uint_tp N, const half_float::half alpha,
                     vptr<half_float::half> X,
                     const half_float::half beta, vptr<half_float::half> Y);

  virtual void axpby_float(const uint_tp N, const float alpha,
                     vptr<float> X, const float beta, vptr<float> Y);

  virtual void axpby_double(const uint_tp N, const double alpha,
                     vptr<double> X, const double beta, vptr<double> Y);

  virtual void set_half(const uint_tp N, const half_float::half alpha,
                        vptr<half_float::half> X);

  virtual void set_float(const uint_tp N, const float alpha,
                         vptr<float> X);

  virtual void set_double(const uint_tp N, const double alpha,
                          vptr<double> X);

  virtual void add_scalar_half(const uint_tp N, const half_float::half alpha,
                          vptr<half_float::half> X);

  virtual void add_scalar_float(const uint_tp N, const float alpha,
                          vptr<float> X);

  virtual void add_scalar_double(const uint_tp N, const double alpha,
                          vptr<double> X);

  virtual void scal_half(const uint_tp N, const half_float::half alpha,
                    vptr<half_float::half> X);

  virtual void scal_float(const uint_tp N, const float alpha,
                          vptr<float> X);

  virtual void scal_double(const uint_tp N, const double alpha,
                           vptr<double> X);

  virtual void add_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y);

  virtual void add_float(const uint_tp N, vptr<float> a,
                         vptr<float> b, vptr<float> y);

  virtual void add_double(const uint_tp N, vptr<double> a,
                         vptr<double> b, vptr<double> y);

  virtual void sub_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y);

  virtual void sub_float(const uint_tp N, vptr<float> a, vptr<float> b,
                         vptr<float> y);

  virtual void sub_double(const uint_tp N, vptr<double> a, vptr<double> b,
                          vptr<double> y);

  virtual void mul_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y);

  virtual void mul_float(const uint_tp N, vptr<float> a,
                        vptr<float> b, vptr<float> y);

  virtual void mul_double(const uint_tp N, vptr<double> a,
                        vptr<double> b, vptr<double> y);

  virtual void div_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y);

  virtual void div_float(const uint_tp N, vptr<float> a, vptr<float> b,
                         vptr<float> y);

  virtual void div_double(const uint_tp N, vptr<double> a, vptr<double> b,
                          vptr<double> y);

  virtual void abs_half(const uint_tp n, vptr<half_float::half> a,
                   vptr<half_float::half> y);

  virtual void abs_float(const uint_tp n, vptr<float> a, vptr<float> y);

  virtual void abs_double(const uint_tp n, vptr<double> a, vptr<double> y);

  virtual void exp_half(const uint_tp n, vptr<half_float::half> a,
                        vptr<half_float::half> y);

  virtual void exp_float(const uint_tp n, vptr<float> a, vptr<float> y);

  virtual void exp_double(const uint_tp n, vptr<double> a, vptr<double> y);

  virtual void log_half(const uint_tp n, vptr<half_float::half> a,
                        vptr<half_float::half> y);

  virtual void log_float(const uint_tp n, vptr<float> a, vptr<float> y);

  virtual void log_double(const uint_tp n, vptr<double> a, vptr<double> y);

  virtual void powx_half(const uint_tp n, vptr<half_float::half> a,
                         const half_float::half b,
                         vptr<half_float::half> y);

  virtual void powx_float(const uint_tp n, vptr<float> a, const float b,
                          vptr<float> y);

  virtual void powx_double(const uint_tp n, vptr<double> a, const double b,
                           vptr<double> y);

  virtual void sqrt_half(const uint_tp n, vptr<half_float::half> a,
                         vptr<half_float::half> y);

  virtual void sqrt_float(const uint_tp n, vptr<float> a, vptr<float> y);

  virtual void sqrt_double(const uint_tp n, vptr<double> a, vptr<double> y);

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
                                  vptr<int_tp> r);

  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int_tp> r);

  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int_tp> r);

  virtual void dot_half(const uint_tp n, vptr<half_float::half> x,
                      vptr<half_float::half> y, half_float::half *out);

  virtual void dot_float(const uint_tp n, vptr<float> x, vptr<float> y,
                         float *out);

  virtual void dot_double(const uint_tp n, vptr<double> x, vptr<double> y,
                          double *out);

  virtual void asum_half(const uint_tp n, vptr<half_float::half> x,
                         half_float::half* y);

  virtual void asum_float(const uint_tp n, vptr<float> x, float* y);

  virtual void asum_double(const uint_tp n, vptr<double> x, double* y);

  virtual void sign_half(const uint_tp n, vptr<half_float::half> x,
                         vptr<half_float::half> y);

  virtual void sign_float(const uint_tp n, vptr<float> x,
                         vptr<float> y);

  virtual void sign_double(const uint_tp n, vptr<double> x,
                         vptr<double> y);

  virtual void sgnbit_half(const uint_tp n, vptr<half_float::half> x,
                           vptr<half_float::half> y);

  virtual void sgnbit_float(const uint_tp n, vptr<float> x, vptr<float> y);

  virtual void sgnbit_double(const uint_tp n, vptr<double> x,
                             vptr<double> y);

  virtual void fabs_half(const uint_tp n, vptr<half_float::half> x,
                         vptr<half_float::half> y);

  virtual void fabs_float(const uint_tp n, vptr<float> x, vptr<float> y);

  virtual void fabs_double(const uint_tp n, vptr<double> x, vptr<double> y);


  virtual void scale_half(const uint_tp n, const half_float::half alpha,
                     vptr<half_float::half> x, vptr<half_float::half> y);

  virtual void scale_float(const uint_tp n, const float alpha,
                           vptr<float> x, vptr<float> y);

  virtual void scale_double(const uint_tp n, const double alpha,
                            vptr<double> x, vptr<double> y);

};

}

#endif  // USE_CUDA

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_HPP_
