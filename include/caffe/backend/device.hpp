#ifndef CAFFE_BACKEND_DEVICE_HPP_
#define CAFFE_BACKEND_DEVICE_HPP_

#ifdef CMAKE_BUILD
#include "caffe_config.h"
#endif

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/device_program.hpp"
#include "caffe/backend/device_kernel.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/trait_helper.hpp"
#include "caffe/util/type_utils.hpp"
#include "caffe/trait_helper.hpp"

#ifdef USE_SQLITE
#include "caffe/util/db_sqlite.hpp"
#endif

#ifdef USE_LIBDNN
#include <unordered_map>
#endif

namespace caffe {

enum DeviceCapability {
  DEVICE_FP16_SUPPORT,
  DEVICE_FP32_SUPPORT,
  DEVICE_FP64_SUPPORT,
  DEVICE_INT32_LOCAL_ATOMICS_SUPPORT,
  DEVICE_INT64_LOCAL_ATOMICS_SUPPORT,
  DEVICE_INT32_LOCAL_EXTENDED_ATOMICS_SUPPORT,
  DEVICE_INT64_LOCAL_EXTENDED_ATOMICS_SUPPORT,
  DEVICE_INT32_GLOBAL_ATOMICS_SUPPORT,
  DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT,
  DEVICE_INT32_GLOBAL_EXTENDED_ATOMICS_SUPPORT,
  DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT,
  DEVICE_32_BIT_ADDRESS,
  DEVICE_64_BIT_ADDRESS,
  DEVICE_CUDA_DP4A_SUPPORT
};

#ifdef USE_LIBDNN
// Forward declare LibDNN
class LibDNNBase;
template<typename MItype, typename MOtype>
class LibDNN;
template<typename MItype, typename MOtype>
class LibDNNBlas;
#endif  // USE_LIBDNN

class Device {
 public:
  explicit Device();

#ifdef USE_SQLITE
  shared_ptr<SQLiteHelper> get_database();
#endif  // USE_SQLITE

  Backend backend() const;
  uint_tp id() const;
  uint_tp list_id() const;

#ifdef USE_LIBDNN
  template<typename MItype, typename MOtype>
  typename std::enable_if<proto_type_is_same<MItype>::value
                       && proto_type_is_same<MOtype>::value,
    shared_ptr<LibDNNBlas<MItype, MOtype> > >::type
  GetLibDNNBlas() {
    libdnn_lock_.lock();
    size_t data_id = data_type_index<MItype>()
                   + data_type_index<MOtype>() * (PROTO_DATA_INDEX_MAX + 1);
    std::unordered_map<size_t, size_t>::iterator it =
                                                 libdnn_blas_map_.find(data_id);
    if (it == libdnn_blas_map_.end()) {
      size_t id = libdnn_blas_.size();
      libdnn_blas_map_[data_id] = id;
      libdnn_blas_.push_back(
          make_shared<LibDNNBlas<MItype, MOtype> >(this));
    }
    libdnn_lock_.unlock();
    return static_pointer_cast<LibDNNBlas<MItype, MOtype> >(
      libdnn_blas_[libdnn_blas_map_[data_id]]);
  }
#endif  // USE_LIBDNN

  template<typename Dtype>
  shared_ptr<Blob<Dtype> > Buffer(vector<int_tp> shape, int_tp* lock_id);
  virtual void unlock_buffer(int_tp* lock_id);

  uint_tp memory_usage();
  uint_tp peak_memory_usage();
  void increase_memory_usage(uint_tp bytes);
  void decrease_memory_usage(uint_tp bytes);
  void reset_peak_memory_usage();

  virtual void Init();
  virtual bool CheckCapability(DeviceCapability cap);
  virtual bool CheckVendor(string vendor);
  virtual bool CheckType(string type);
  virtual void SwitchQueue(uint_tp id);
  uint_tp current_queue_id();
  size_t workgroup_size(uint_tp id);
  template<typename Dtype>
  size_t preferred_vector_width();
  virtual void get_threads(const vector<size_t>* work_size,
                           vector<size_t>* group,
                           vector<size_t>* local,
                           DeviceKernel* kernel,
                           bool auto_select);
  virtual void FinishQueues();
  virtual uint_tp num_queues();
  virtual bool is_host_unified();
  bool is_fast_unsafe_math() const;
  virtual string name();
  virtual shared_ptr<DeviceProgram> CreateProgram();

  virtual void MallocMemHost(uint_tp size, void** ptr);
  virtual void FreeMemHost(void* ptr);
  virtual vptr<void> MallocMemDevice(uint_tp size, void** ptr,
                                     bool zero_copy);
  virtual void FreeMemDevice(vptr<void> ptr);
  virtual bool CheckZeroCopy(vptr<const void> gpu_ptr, void* cpu_ptr,
                             uint_tp size);

  void null_kernel(float arg);

  template<typename Dtype>
  void im2col(vptr<const Dtype> data_im, const int_tp channels,
              const int_tp height, const int_tp width, const int_tp kernel_h,
              const int_tp kernel_w, const int_tp pad_h, const int_tp pad_w,
              const int_tp stride_h, const int_tp stride_w,
              const int_tp dilation_h, const int_tp dilation_w,
              vptr<Dtype> data_col,
              const QuantizerValues* const data_quant= nullptr);

  template<typename Dtype>
  void col2im(vptr<const Dtype> data_col, const int_tp channels,
              const int_tp height, const int_tp width, const int_tp kernel_h,
              const int_tp kernel_w, const int_tp pad_h, const int_tp pad_w,
              const int_tp stride_h, const int_tp stride_w,
              const int_tp dilation_h, const int_tp dilation_w,
              vptr<Dtype> data_im);

  template<typename Dtype>
  void im2col_nd(vptr<const Dtype> data_im, const int_tp num_spatial_axes,
                 const int_tp num_kernels, vptr<const int_tp> im_shape,
                 vptr<const int_tp> col_shape, vptr<const int_tp> kernel_shape,
                 vptr<const int_tp> pad, vptr<const int_tp> stride,
                 vptr<const int_tp> dilation, vptr<Dtype> data_col,
                 const QuantizerValues* const data_quant= nullptr);

  template<typename Dtype>
  void col2im_nd(vptr<const Dtype> data_col, const int_tp num_spatial_axes,
                 const int_tp im_size, vptr<const int_tp> im_shape,
                 vptr<const int_tp> col_shape, vptr<const int_tp> kernel_shape,
                 vptr<const int_tp> pad, vptr<const int_tp> stride,
                 vptr<const int_tp> dilation, vptr<Dtype> data_im);


  template<typename Dtype>
  void copy(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y);

  template<typename Dtype>
  void copy(const uint_tp n, const Dtype* x, vptr<Dtype> y);

  template<typename Dtype>
  void copy(const uint_tp n, vptr<const Dtype> x, Dtype* y);

  template<typename Dtype>
  void gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
            const uint_tp m, const uint_tp n, const uint_tp k,
            const Dtype alpha, vptr<const Dtype> a,
            vptr<const Dtype> b,
            const Dtype beta, vptr<Dtype> c,
            const QuantizerValues* const alpha_quant = nullptr,
            const QuantizerValues* const a_quant = nullptr,
            const QuantizerValues* const b_quant = nullptr,
            const QuantizerValues* const beta_quant = nullptr,
            const QuantizerValues* const c_quant = nullptr);

  template<typename Dtype>
  void gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m,
            const uint_tp n, const Dtype alpha,
            vptr<const Dtype> a,
            vptr<const Dtype> x, const Dtype beta,
            vptr<Dtype> y,
            const QuantizerValues* const alpha_quant = nullptr,
            const QuantizerValues* const a_quant = nullptr,
            const QuantizerValues* const x_quant = nullptr,
            const QuantizerValues* const beta_quant = nullptr,
            const QuantizerValues* const y_quant = nullptr);

  template<typename Dtype>
  void axpy(const uint_tp n, const Dtype alpha, vptr<const Dtype> x,
            vptr<Dtype> y,
            const QuantizerValues* const alpha_quant = nullptr,
            const QuantizerValues* const x_quant = nullptr,
            const QuantizerValues* const y_quant = nullptr);

  template<typename Dtype>
  void axpby(const uint_tp n, const Dtype alpha, vptr<const Dtype> x,
             const Dtype beta, vptr<Dtype> y,
             const QuantizerValues* const alpha_quant = nullptr,
             const QuantizerValues* const x_quant = nullptr,
             const QuantizerValues* const beta_quant = nullptr,
             const QuantizerValues* const y_quant = nullptr);

  virtual void memcpy(const uint_tp n, vptr<const void> x, vptr<void> y);
  virtual void memcpy(const uint_tp n, const void* x, vptr<void> y);
  virtual void memcpy(const uint_tp n, vptr<const void> x, void* y);

  template<typename Dtype>
  void set(const uint_tp n, const Dtype alpha, vptr<Dtype> x);

  void memset(const uint_tp n, const char alpha, vptr<char> x);

  template<typename Dtype>
  void add_scalar(const uint_tp n, const Dtype alpha, vptr<Dtype> x);

  template<typename Dtype>
  void scal(const uint_tp n, const Dtype alpha, vptr<Dtype> x);

  template<typename Dtype>
  void add(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
           vptr<Dtype> y);

  template<typename Dtype>
  void sub(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
           vptr<Dtype> y);

  template<typename Dtype>
  void mul(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
           vptr<Dtype> y);

  template<typename Dtype>
  void div(const uint_tp n, vptr<const Dtype> a, vptr<const Dtype> b,
           vptr<Dtype> y);

  template<typename Dtype>
  void abs(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y);

  template<typename Dtype>
  void exp(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y);

  template<typename Dtype>
  void log(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y);

  template<typename Dtype>
  void powx(const uint_tp n, vptr<const Dtype> a, const Dtype b, vptr<Dtype> y);

  template <typename Dtype>
  void sqrt(const uint_tp n, vptr<const Dtype> a, vptr<Dtype> y);

  // rng_uniform with two arguments generates integers in the range
  // [0, UINT_MAX].
  virtual void rng_uniform(const uint_tp n, vptr<uint8_t> r);
  virtual void rng_uniform(const uint_tp n, vptr<uint16_t> r);
  virtual void rng_uniform(const uint_tp n, vptr<uint32_t> r);
  virtual void rng_uniform(const uint_tp n, vptr<uint64_t> r);

  // rng_uniform with four arguments generates floats in the range
  // (a, b] (strictly greater than a, less than or equal to b)
  template<typename Dtype>
  void rng_uniform(const uint_tp n, const Dtype a, const Dtype b,
                   vptr<Dtype> r);

  template<typename Dtype>
  void rng_gaussian(const uint_tp n, const Dtype mu, const Dtype sigma,
                    vptr<Dtype> r);

  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<int8_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<int16_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<int32_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<int64_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<uint8_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<uint16_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<uint32_t> r);
  template<typename Dtype>
  void rng_bernoulli(const uint_tp n, const Dtype p, vptr<uint64_t> r);

  template<typename Dtype>
  void dot(const uint_tp n, vptr<const Dtype> x, vptr<const Dtype> y,
           Dtype* out);

  template<typename Dtype>
  void asum(const uint_tp n, vptr<const Dtype> x, Dtype* y);

  template<typename Dtype>
  void sign(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y);

  template<typename Dtype>
  void sgnbit(const uint_tp n, vptr<const Dtype> x, vptr<Dtype> y);

  template<typename Dtype>
  void scale(const uint_tp n, const Dtype alpha, vptr<const Dtype> x,
             vptr<Dtype> y);

 protected:
  void CreateMathProgram();
  void CreateIm2ColProgram();

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

  virtual void rng_uniform_half(const uint_tp n, const half_fp a,
                                const half_fp b, vptr<half_fp> r);
  virtual void rng_gaussian_half(const uint_tp n, const half_fp mu,
                                 const half_fp sigma, vptr<half_fp> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<int8_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<int16_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<int32_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<int64_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<uint8_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<uint16_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<uint32_t> r);
  virtual void rng_bernoulli_half(const uint_tp n, const half_fp p,
                                  vptr<uint64_t> r);
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
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int8_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int16_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int32_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<int64_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<uint8_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<uint16_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<uint32_t> r);
  virtual void rng_bernoulli_float(const uint_tp n, const float p,
                                   vptr<uint64_t> r);
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
                          vptr<const double> x, vptr<double> y);
  virtual void axpby_double(const uint_tp n, const double alpha,
                       vptr<const double> x, const double beta, vptr<double> y);
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
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int8_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int16_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int32_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<int64_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint8_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint16_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint32_t> r);
  virtual void rng_bernoulli_double(const uint_tp n, const double p,
                                    vptr<uint64_t> r);
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
  virtual void gemm_uint8
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const uint8_t alpha, vptr<const uint8_t> a,
                 vptr<const uint8_t> b,
                 const uint8_t beta,
                 vptr<uint8_t> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
  virtual void gemv_uint8
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const uint8_t alpha,
                 vptr<const uint8_t> a,
                 vptr<const uint8_t> x, const uint8_t beta,
                 vptr<uint8_t> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_uint8(const uint_tp n,
                         const uint8_t alpha,
                         vptr<const uint8_t> x,
                         vptr<uint8_t> y,
                         const QuantizerValues* const alpha_quant = nullptr,
                         const QuantizerValues* const x_quant = nullptr,
                         const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_uint8(const uint_tp n, const uint8_t alpha,
                    vptr<const uint8_t> x, const uint8_t beta, vptr<uint8_t> y,
                    const QuantizerValues* const alpha_quant = nullptr,
                    const QuantizerValues* const x_quant = nullptr,
                    const QuantizerValues* const beta_quant = nullptr,
                    const QuantizerValues* const y_quant = nullptr);
  virtual void dot_uint8(const uint_tp n, vptr<const uint8_t> x,
                        vptr<const uint8_t> y, uint8_t *out);
  virtual void asum_uint8(const uint_tp n, vptr<const uint8_t> x, uint8_t* y);
  virtual void scale_uint8(const uint_tp n, const uint8_t alpha,
                          vptr<const uint8_t> x, vptr<uint8_t> y);
  virtual void scal_uint8(const uint_tp n, const uint8_t alpha, vptr<uint8_t> x);
  virtual void rng_uniform_uint8(const uint_tp n, const uint8_t a,
                                 const uint8_t b, vptr<uint8_t> r);

#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
  virtual void gemm_uint16
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const uint16_t alpha, vptr<const uint16_t> a,
                 vptr<const uint16_t> b,
                 const uint16_t beta,
                 vptr<uint16_t> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
  virtual void gemv_uint16
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const uint16_t alpha,
                 vptr<const uint16_t> a,
                 vptr<const uint16_t> x, const uint16_t beta,
                 vptr<uint16_t> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_uint16(const uint_tp n,
                         const uint16_t alpha,
                         vptr<const uint16_t> x,
                         vptr<uint16_t> y,
                         const QuantizerValues* const alpha_quant = nullptr,
                         const QuantizerValues* const x_quant = nullptr,
                         const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_uint16(const uint_tp n, const uint16_t alpha,
                    vptr<const uint16_t> x, const uint16_t beta,
                    vptr<uint16_t> y,
                    const QuantizerValues* const alpha_quant = nullptr,
                    const QuantizerValues* const x_quant = nullptr,
                    const QuantizerValues* const beta_quant = nullptr,
                    const QuantizerValues* const y_quant = nullptr);
  virtual void dot_uint16(const uint_tp n, vptr<const uint16_t> x,
                        vptr<const uint16_t> y, uint16_t *out);
  virtual void asum_uint16(const uint_tp n, vptr<const uint16_t> x, uint16_t* y);
  virtual void scale_uint16(const uint_tp n, const uint16_t alpha,
                          vptr<const uint16_t> x, vptr<uint16_t> y);
  virtual void scal_uint16(const uint_tp n, const uint16_t alpha,
                          vptr<uint16_t> x);
  virtual void rng_uniform_uint16(const uint_tp n, const uint16_t a,
                                 const uint16_t b, vptr<uint16_t> r);
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
  virtual void gemm_uint32
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const uint32_t alpha, vptr<const uint32_t> a,
                 vptr<const uint32_t> b,
                 const uint32_t beta,
                 vptr<uint32_t> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
  virtual void gemv_uint32
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const uint32_t alpha,
                 vptr<const uint32_t> a,
                 vptr<const uint32_t> x, const uint32_t beta,
                 vptr<uint32_t> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_uint32(const uint_tp n,
                         const uint32_t alpha,
                         vptr<const uint32_t> x,
                         vptr<uint32_t> y,
                         const QuantizerValues* const alpha_quant = nullptr,
                         const QuantizerValues* const x_quant = nullptr,
                         const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_uint32(const uint_tp n, const uint32_t alpha,
                    vptr<const uint32_t> x, const uint32_t beta,
                    vptr<uint32_t> y,
                    const QuantizerValues* const alpha_quant = nullptr,
                    const QuantizerValues* const x_quant = nullptr,
                    const QuantizerValues* const beta_quant = nullptr,
                    const QuantizerValues* const y_quant = nullptr);
  virtual void dot_uint32(const uint_tp n, vptr<const uint32_t> x,
                        vptr<const uint32_t> y, uint32_t *out);
  virtual void asum_uint32(const uint_tp n, vptr<const uint32_t> x,
                           uint32_t* y);
  virtual void scale_uint32(const uint_tp n, const uint32_t alpha,
                          vptr<const uint32_t> x, vptr<uint32_t> y);
  virtual void scal_uint32(const uint_tp n, const uint32_t alpha,
                          vptr<uint32_t> x);
  virtual void rng_uniform_uint32(const uint_tp n, const uint32_t a,
                                 const uint32_t b, vptr<uint32_t> r);
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
  virtual void gemm_uint64
                (const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
                 const uint_tp m, const uint_tp n, const uint_tp k,
                 const uint64_t alpha, vptr<const uint64_t> a,
                 vptr<const uint64_t> b,
                 const uint64_t beta,
                 vptr<uint64_t> c,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const b_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const c_quant = nullptr);
  virtual void gemv_uint64
                (const CBLAS_TRANSPOSE trans_a, const uint_tp m,
                 const uint_tp n, const uint64_t alpha,
                 vptr<const uint64_t> a,
                 vptr<const uint64_t> x, const uint64_t beta,
                 vptr<uint64_t> y,
                 const QuantizerValues* const alpha_quant = nullptr,
                 const QuantizerValues* const a_quant = nullptr,
                 const QuantizerValues* const x_quant = nullptr,
                 const QuantizerValues* const beta_quant = nullptr,
                 const QuantizerValues* const y_quant = nullptr);
  virtual void axpy_uint64(const uint_tp n,
                         const uint64_t alpha,
                         vptr<const uint64_t> x,
                         vptr<uint64_t> y,
                         const QuantizerValues* const alpha_quant = nullptr,
                         const QuantizerValues* const x_quant = nullptr,
                         const QuantizerValues* const y_quant = nullptr);
  virtual void axpby_uint64(const uint_tp n, const uint64_t alpha,
                    vptr<const uint64_t> x, const uint64_t beta,
                    vptr<uint64_t> y,
                    const QuantizerValues* const alpha_quant = nullptr,
                    const QuantizerValues* const x_quant = nullptr,
                    const QuantizerValues* const beta_quant = nullptr,
                    const QuantizerValues* const y_quant = nullptr);
  virtual void dot_uint64(const uint_tp n, vptr<const uint64_t> x,
                        vptr<const uint64_t> y, uint64_t *out);
  virtual void asum_uint64(const uint_tp n, vptr<const uint64_t> x,
                           uint64_t* y);
  virtual void scale_uint64(const uint_tp n, const uint64_t alpha,
                          vptr<const uint64_t> x, vptr<uint64_t> y);
  virtual void scal_uint64(const uint_tp n, const uint64_t alpha,
                          vptr<uint64_t> x);
  virtual void rng_uniform_uint64(const uint_tp n, const uint64_t a,
                                 const uint64_t b, vptr<uint64_t> r);
#endif  // USE_INT_QUANT_64


  int_tp current_queue_id_;
  size_t max_local_size_;
  vector<size_t> max_local_sizes_;
  vector<size_t> max_group_sizes_;
  std::map<string, size_t> preferred_vector_widths_;
  int_tp id_;
  int_tp list_id_;
  Backend backend_;
  uint_tp memory_usage_;
  uint_tp peak_memory_usage_;
  vector<shared_ptr<Blob<uint8_t> > > buffers_;
  std::mutex buffer_vec_mutex_;
  vector<shared_ptr<std::atomic<bool> > > buffer_flags_;
  bool host_unified_;
  bool fast_unsafe_math_;
  string name_;
  vector<shared_ptr<DeviceProgram> > math_programs_;
  vector<shared_ptr<DeviceProgram> > im2col_programs_;
#ifdef USE_LIBDNN
  std::mutex libdnn_lock_;
  vector<shared_ptr<LibDNNBase> > libdnn_blas_;
  std::unordered_map<size_t, size_t> libdnn_blas_map_;
#endif  // USE_LIBDNN

#ifdef USE_SQLITE
  shared_ptr<SQLiteHelper> database_;
#endif
};

}  // namespace caffe

#endif  // CAFFE_BACKEND_DEVICE_HPP_
