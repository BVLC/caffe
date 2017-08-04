#ifndef CAFFE_BACKEND_DEVICE_HPP_
#define CAFFE_BACKEND_DEVICE_HPP_

#ifdef CMAKE_BUILD
#include "caffe_config.h"
#endif

#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/vptr.hpp"

namespace caffe {

class device {
 public:
  Backend backend() const;
  uint_tp id() const;
  uint_tp list_id() const;

  template<typename Dtype, typename Mtype>
  std::shared_ptr<Blob<Dtype, Mtype> > Buffer(uint_tp id);

  uint_tp memory_usage();
  uint_tp peak_memory_usage();
  void IncreaseMemoryUsage(uint_tp bytes);
  void DecreaseMemoryUsage(uint_tp bytes);
  void ResetPeakMemoryUsage();

  virtual void Init() = 0;
  virtual bool CheckCapability(std::string cap) = 0;
  virtual bool CheckVendor(std::string vendor) = 0;
  virtual bool CheckType(std::string type) = 0;
  virtual void SwitchQueue(uint_tp id) = 0;
  virtual uint_tp current_queue_id() = 0;
  virtual uint_tp workgroup_size(uint_tp id) = 0;
  virtual void FinishQueues() = 0;
  virtual uint_tp num_queues() = 0;
  virtual std::string name() = 0;

  template<typename Dtype>
  void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                      const uint_tp M, const uint_tp N, const uint_tp K,
                      const Dtype alpha, vptr<Dtype> A,
                      vptr<Dtype> B,
                      const Dtype beta, vptr<Dtype> C);

  template<typename Dtype>
  void gemv(const CBLAS_TRANSPOSE TransA, const uint_tp M,
                      const uint_tp N, const Dtype alpha,
                      vptr<Dtype> A,
                      vptr<Dtype> x, const Dtype beta,
                      vptr<Dtype> y);

  template<typename Dtype>
  void axpy(const uint_tp N, const Dtype alpha, vptr<Dtype> X, vptr<Dtype> Y);

  template<typename Dtype>
  void axpby(const uint_tp N, const Dtype alpha, vptr<Dtype> X,
             const Dtype beta, vptr<Dtype> Y);

  virtual void memcpy(const uint_tp N, vptr<void> X, vptr<void> Y) = 0;

  template<typename Dtype>
  void set(const uint_tp N, const Dtype alpha, vptr<Dtype> X);

  virtual void memset(const uint_tp N, const int_tp alpha, vptr<void> X) = 0;

  template<typename Dtype>
  void add_scalar(const uint_tp N, const Dtype alpha, vptr<Dtype> X);

  template<typename Dtype>
  void scal(const uint_tp N, const Dtype alpha, vptr<Dtype> X);

  template<typename Dtype>
  void add(const uint_tp N, vptr<Dtype> a, vptr<Dtype> b, vptr<Dtype> y);

  template<typename Dtype>
  void sub(const uint_tp N, vptr<Dtype> a, vptr<Dtype> b, vptr<Dtype> y);

  template<typename Dtype>
  void mul(const uint_tp N, vptr<Dtype> a, vptr<Dtype> b, vptr<Dtype> y);

  template<typename Dtype>
  void div(const uint_tp N, vptr<Dtype> a, vptr<Dtype> b, vptr<Dtype> y);

  template<typename Dtype>
  void abs(const uint_tp n, vptr<Dtype> a, vptr<Dtype> y);

  template<typename Dtype>
  void exp(const uint_tp n, vptr<Dtype> a, vptr<Dtype> y);

  template<typename Dtype>
  void log(const uint_tp n, vptr<Dtype> a, vptr<Dtype> y);

  template<typename Dtype>
  void powx(const uint_tp n, vptr<Dtype> a, const Dtype b, vptr<Dtype> y);

  template <typename Dtype>
  void sqrt(const uint_tp n, vptr<Dtype> a, vptr<Dtype> y);

  // rng_uniform with two arguments generates integers in the range
  // [0, UINT_MAX].
  virtual void rng_uniform(const uint_tp n, vptr<uint32_t>* r) = 0;  // NOLINT
  virtual void rng_uniform(const uint_tp n, vptr<uint64_t>* r) = 0;  // NOLINT

  // rng_uniform with four arguments generates floats in the range
  // (a, b] (strictly greater than a, less than or equal to b)
  template<typename Dtype>
  void rng_uniform(const uint_tp n, const Dtype a, const Dtype b, vptr<Dtype> r);

  template<typename Dtype>
  void rng_gaussian(const uint_tp n, const Dtype mu, const Dtype sigma,
                    vptr<Dtype> r);

  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<int_tp> r);

  template<typename Dtype>
  void dot(const uint_tp n, vptr<Dtype> x, vptr<Dtype> y, Dtype* out);

  template<typename Dtype>
  void asum(const uint_tp n, vptr<Dtype> x, Dtype* y);

  template<typename Dtype>
  void sign(const uint_tp n, vptr<Dtype> x, vptr<Dtype> y);

  template<typename Dtype>
  void sgnbit(const uint_tp n, vptr<Dtype> x, vptr<Dtype> y);

  template<typename Dtype>
  void fabs(const uint_tp n, vptr<Dtype> x, vptr<Dtype> y);

  template<typename Dtype>
  void scale(const uint_tp n, const Dtype alpha, vptr<Dtype> x, vptr<Dtype> y);

 protected:

  virtual void gemm_half
                (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const half_float::half alpha, vptr<half_float::half> A,
                 vptr<half_float::half> B,
                 const half_float::half beta,
                 vptr<half_float::half> C) = 0;

  virtual void gemm_float
                (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const float alpha, vptr<float> A,
                 vptr<float> B,
                 const float beta, vptr<float> C) = 0;

  virtual void gemm_double
                (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const uint_tp M, const uint_tp N, const uint_tp K,
                 const double alpha, vptr<double> A,
                 vptr<double> B,
                 const double beta, vptr<double> C) = 0;

  virtual void gemv_half
                (const CBLAS_TRANSPOSE TransA, const uint_tp M,
                 const uint_tp N, const half_float::half alpha,
                 vptr<half_float::half> A,
                 vptr<half_float::half> x, const half_float::half beta,
                 vptr<half_float::half> y) = 0;

  virtual void gemv_float
                (const CBLAS_TRANSPOSE TransA, const uint_tp M,
                 const uint_tp N, const float alpha,
                 vptr<float> A,
                 vptr<float> x, const float beta,
                 vptr<float> y) = 0;

  virtual void gemv_double
                (const CBLAS_TRANSPOSE TransA, const uint_tp M,
                 const uint_tp N, const double alpha,
                 vptr<double> A,
                 vptr<double> x, const double beta,
                 vptr<double> y) = 0;

  virtual void axpy_half(const uint_tp N,
                         const half_float::half alpha,
                         vptr<half_float::half> X,
                         vptr<half_float::half> Y) = 0;

  virtual void axpy_float(const uint_tp N, const float alpha,
                          vptr<float> X, vptr<float> Y) = 0;

  virtual void axpy_double(const uint_tp N, const double alpha,
                          vptr<double> X, vptr<double> Y) = 0;

  virtual void axpby_half(const uint_tp N, const half_float::half alpha,
                     vptr<half_float::half> X,
                     const half_float::half beta, vptr<half_float::half> Y) = 0;

  virtual void axpby_float(const uint_tp N, const float alpha,
                     vptr<float> X, const float beta, vptr<float> Y) = 0;

  virtual void axpby_double(const uint_tp N, const double alpha,
                     vptr<double> X, const double beta, vptr<double> Y) = 0;

  virtual void set_half(const uint_tp N, const half_float::half alpha,
                        vptr<half_float::half> X) = 0;

  virtual void set_float(const uint_tp N, const float alpha,
                         vptr<float> X) = 0;

  virtual void set_double(const uint_tp N, const double alpha,
                          vptr<double> X) = 0;

  virtual void add_scalar_half(const uint_tp N, const half_float::half alpha,
                          vptr<half_float::half> X) = 0;

  virtual void add_scalar_float(const uint_tp N, const float alpha,
                          vptr<float> X) = 0;

  virtual void add_scalar_double(const uint_tp N, const double alpha,
                          vptr<double> X) = 0;

  virtual void scal_half(const uint_tp N, const half_float::half alpha,
                    vptr<half_float::half> X) = 0;

  virtual void scal_float(const uint_tp N, const float alpha,
                          vptr<float> X) = 0;

  virtual void scal_double(const uint_tp N, const double alpha,
                           vptr<double> X) = 0;

  virtual void add_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y) = 0;

  virtual void add_float(const uint_tp N, vptr<float> a,
                         vptr<float> b, vptr<float> y) = 0;

  virtual void add_double(const uint_tp N, vptr<double> a,
                         vptr<double> b, vptr<double> y) = 0;

  virtual void sub_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y) = 0;

  virtual void sub_float(const uint_tp N, vptr<float> a, vptr<float> b,
                         vptr<float> y) = 0;

  virtual void sub_double(const uint_tp N, vptr<double> a, vptr<double> b,
                          vptr<double> y) = 0;

  virtual void mul_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y) = 0;

  virtual void mul_float(const uint_tp N, vptr<float> a,
                        vptr<float> b, vptr<float> y) = 0;

  virtual void mul_double(const uint_tp N, vptr<double> a,
                        vptr<double> b, vptr<double> y) = 0;

  virtual void div_half(const uint_tp N, vptr<half_float::half> a,
                        vptr<half_float::half> b, vptr<half_float::half> y) = 0;

  virtual void div_float(const uint_tp N, vptr<float> a, vptr<float> b,
                         vptr<float> y) = 0;

  virtual void div_double(const uint_tp N, vptr<double> a, vptr<double> b,
                          vptr<double> y) = 0;

  virtual void abs_half(const uint_tp n, vptr<half_float::half> a,
                   vptr<half_float::half> y) = 0;

  virtual void abs_float(const uint_tp n, vptr<float> a, vptr<float> y) = 0;

  virtual void abs_double(const uint_tp n, vptr<double> a, vptr<double> y) = 0;

  virtual void exp_half(const uint_tp n, vptr<half_float::half> a,
                        vptr<half_float::half> y) = 0;

  virtual void exp_float(const uint_tp n, vptr<float> a, vptr<float> y) = 0;

  virtual void exp_double(const uint_tp n, vptr<double> a, vptr<double> y) = 0;

  virtual void log_half(const uint_tp n, vptr<half_float::half> a,
                        vptr<half_float::half> y) = 0;

  virtual void log_float(const uint_tp n, vptr<float> a, vptr<float> y) = 0;

  virtual void log_double(const uint_tp n, vptr<double> a, vptr<double> y) = 0;

  virtual void powx_half(const uint_tp n, vptr<half_float::half> a,
                         const half_float::half b,
                         vptr<half_float::half> y) = 0;

  virtual void powx_float(const uint_tp n, vptr<float> a, const float b,
                          vptr<float> y) = 0;

  virtual void powx_double(const uint_tp n, vptr<double> a, const double b,
                           vptr<double> y) = 0;

  virtual void sqrt_half(const uint_tp n, vptr<half_float::half> a,
                         vptr<half_float::half> y) = 0;

  virtual void sqrt_float(const uint_tp n, vptr<float> a, vptr<float> y) = 0;

  virtual void sqrt_double(const uint_tp n, vptr<double> a, vptr<double> y) = 0;

  virtual void rng_uniform_half(const uint_tp n, const half_float::half a,
                        const half_float::half b, vptr<half_float::half> r) = 0;

  virtual void rng_uniform_float(const uint_tp n, const float a, const float b,
                                 vptr<float> r) = 0;

  virtual void rng_uniform_double(const uint_tp n, const double a,
                                  const double b, vptr<double> r) = 0;

  virtual void rng_gaussian_half(const uint_tp n, const half_float::half mu,
                    const half_float::half sigma, vptr<half_float::half> r) = 0;

  virtual void rng_gaussian_float(const uint_tp n, const float mu,
                                  const float sigma, vptr<float> r) = 0;

  virtual void rng_gaussian_double(const uint_tp n, const double mu,
                                   const double sigma, vptr<double> r) = 0;

  virtual void rng_bernoulli_half(const uint_tp n, const half_float::half p,
                                  vptr<int_tp> r) = 0;

  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int_tp> r) = 0;

  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int_tp> r) = 0;

  virtual void dot_half(const uint_tp n, vptr<half_float::half> x,
                      vptr<half_float::half> y, half_float::half *out) = 0;

  virtual void dot_float(const uint_tp n, vptr<float> x, vptr<float> y,
                         float *out) = 0;

  virtual void dot_double(const uint_tp n, vptr<double> x, vptr<double> y,
                          double *out) = 0;

  virtual void asum_half(const uint_tp n, vptr<half_float::half> x,
                         half_float::half* y) = 0;

  virtual void asum_float(const uint_tp n, vptr<float> x, float* y) = 0;

  virtual void asum_double(const uint_tp n, vptr<double> x, double* y) = 0;

  virtual void sign_half(const uint_tp n, vptr<half_float::half> x,
                         vptr<half_float::half> y) = 0;

  virtual void sign_float(const uint_tp n, vptr<float> x,
                         vptr<float> y) = 0;

  virtual void sign_double(const uint_tp n, vptr<double> x,
                         vptr<double> y) = 0;

  virtual void sgnbit_half(const uint_tp n, vptr<half_float::half> x,
                           vptr<half_float::half> y) = 0;

  virtual void sgnbit_float(const uint_tp n, vptr<float> x, vptr<float> y) = 0;

  virtual void sgnbit_double(const uint_tp n, vptr<double> x,
                             vptr<double> y) = 0;

  virtual void fabs_half(const uint_tp n, vptr<half_float::half> x,
                         vptr<half_float::half> y) = 0;

  virtual void fabs_float(const uint_tp n, vptr<float> x, vptr<float> y) = 0;

  virtual void fabs_double(const uint_tp n, vptr<double> x, vptr<double> y) = 0;


  virtual void scale_half(const uint_tp n, const half_float::half alpha,
                     vptr<half_float::half> x, vptr<half_float::half> y) = 0;

  virtual void scale_float(const uint_tp n, const float alpha,
                           vptr<float> x, vptr<float> y) = 0;

  virtual void scale_double(const uint_tp n, const double alpha,
                            vptr<double> x, vptr<double> y) = 0;

  int current_queue_id_;
  std::vector<int> workgroup_sizes_;
  int id_;
  int list_id_;
  Backend backend_;
  uint_tp memory_usage_;
  uint_tp peak_memory_usage_;
  std::vector<std::shared_ptr<Blob<half_float::half, half_float::half> > > buff_h_;
  std::vector<std::shared_ptr<Blob<float, float> > > buff_f_;
  std::vector<std::shared_ptr<Blob<double, double> > > buff_d_;
  bool host_unified_;
  std::string name_;
};
}  // namespace caffe

#endif  // CAFFE_BACKEND_DEVICE_HPP_
