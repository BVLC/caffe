#include <boost/math/special_functions/next.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_axpby<half_fp>(const int_tp n,
                        const half_fp alpha, const half_fp* X,
                        const half_fp beta, half_fp* Y) {
  cblas_haxpby(n, alpha, X, 1, beta, Y, 1);
}

void vhAdd(const int_tp n, const half_fp* a, const half_fp* b,
                     half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] + b[i];
  }
}

template<>
void caffe_add<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhAdd(n, a, b, Y);
}

void vhSub(const int_tp n, const half_fp* a, const half_fp* b,
           half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] - b[i];
  }
}

template<>
void caffe_sub<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhSub(n, a, b, Y);
}

void vhMul(const int_tp n, const half_fp* a, const half_fp* b,
           half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] * b[i];
  }
}

template<>
void caffe_mul<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhMul(n, a, b, Y);
}

void vhDiv(const int_tp n, const half_fp* a, const half_fp* b,
           half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = a[i] / b[i];
  }
}

template<>
void caffe_div<half_fp>(const int_tp n, const half_fp* a,
                               const half_fp* b, half_fp* Y) {
  vhDiv(n, a, b, Y);
}

void vhPowx(const int_tp n, const half_fp*a, const half_fp b,
            half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = pow(a[i], b);
  }
}

template<>
void caffe_powx<half_fp>(const int_tp n, const half_fp* a,
                                const half_fp b, half_fp* Y) {
  vhPowx(n, a, b, Y);
}

void vhSqr(const int_tp n, const half_fp *a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = sqrt(a[i]);
  }
}

template<>
void caffe_sqr<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhSqr(n, a, Y);
}

void vhExp(const int_tp n, const half_fp* a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = exp(a[i]);
  }
}

template<>
void caffe_exp<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhExp(n, a, Y);
}

void vhLn(const int_tp n, const half_fp* a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = log(a[i]);
  }
}

template<>
void caffe_log<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhLn(n, a, Y);
}

void vhAbs(const int_tp n, const half_fp *a, half_fp* Y) {
  for (int i = 0; i < n; i++) {
    Y[i] = fabs(a[i]);
  }
}

template<>
void caffe_abs<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* Y) {
  vhAbs(n, a, Y);
}

void vsHqrt(const int_tp n, const half_fp* a, half_fp* Y) {
  for (int_tp i = 0; i < n; i++) {
    Y[i] = sqrt(a[i]);
  }
}
template <>
void caffe_sqrt<half_fp>(const int_tp n, const half_fp* a,
                                  half_fp* Y) {
  vsHqrt(n, a, Y);
}

template<>
void caffe_scale<half_fp>(const int_tp n,
                        const half_fp alpha, const half_fp *X,
                        half_fp* Y) {
  for (int_tp i = 0; i < n; i++)
    Y[i] = X[i];
  caffe_scal(n, alpha, Y);
}

template<>
half_fp caffe_strided_dot<half_fp>(const int_tp n,
                                const half_fp* X, const int_tp incx,
                                const half_fp* Y, const int_tp incy) {
  half_fp sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += X[i * incx] * Y[i * incy];
  return sum;
}


// AXPY
template<>
void caffe_axpy<float>(const int_tp n, const float alpha, const float* X,
                       float* Y) {
  cblas_saxpy(n, alpha, X, 1, Y, 1);
}
template<>
void caffe_axpy<double>(const int_tp n, const double alpha, const double* X,
                        double* Y) {
  cblas_daxpy(n, alpha, X, 1, Y, 1);
}
template<typename Dtype>
void caffe_axpy(const int_tp n, const Dtype alpha,
                         const Dtype* X, Dtype* Y) {
#pragma omp parallel for
  for (int_tp i = 0; i < n; ++i) {
    Y[i] += alpha * X[i];
  }
}
template void caffe_axpy<half_fp>(const int_tp n, const half_fp alpha,
                         const half_fp* X, half_fp* Y);
template void caffe_axpy<int8_t>(const int_tp n, const int8_t alpha,
                         const int8_t* X, int8_t* Y);
template void caffe_axpy<int16_t>(const int_tp n, const int16_t alpha,
                         const int16_t* X, int16_t* Y);
template void caffe_axpy<int32_t>(const int_tp n, const int32_t alpha,
                         const int32_t* X, int32_t* Y);
template void caffe_axpy<int64_t>(const int_tp n, const int64_t alpha,
                         const int64_t* X, int64_t* Y);

// SET
template<typename Dtype>
void caffe_set(const int_tp n, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int_tp i = 0; i < n; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int8_t>(const int_tp n, const int8_t alpha,
                                int8_t* Y);
template void caffe_set<uint8_t>(const int_tp n, const uint8_t alpha,
                                 uint8_t* Y);
template void caffe_set<int16_t>(const int_tp n, const int16_t alpha,
                                 int16_t* Y);
template void caffe_set<uint16_t>(const int_tp n, const uint16_t alpha,
                                  uint16_t* Y);
template void caffe_set<int32_t>(const int_tp n, const int32_t alpha,
                                 int32_t* Y);
template void caffe_set<uint32_t>(const int_tp n, const uint32_t alpha,
                                  uint32_t* Y);
template void caffe_set<int64_t>(const int_tp n, const int64_t alpha,
                                 int64_t* Y);
template void caffe_set<uint64_t>(const int_tp n, const uint64_t alpha,
                                  uint64_t* Y);
template void caffe_set<half_fp>(const int_tp n,
                             const half_fp alpha, half_fp* Y);
template void caffe_set<float>(const int_tp n, const float alpha, float* Y);
template void caffe_set<double>(const int_tp n, const double alpha, double* Y);

// ADD SCALAR
template<typename Dtype>
void caffe_add_scalar(const int_tp n, const Dtype alpha, Dtype* y) {
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    y[i] += alpha;
  }
}

template void caffe_add_scalar(const int_tp n, const half_fp alpha, half_fp* y);
template void caffe_add_scalar(const int_tp n, const float alpha, float* y);
template void caffe_add_scalar(const int_tp n, const double alpha, double* y);
template void caffe_add_scalar(const int_tp n, const int8_t alpha, int8_t* y);
template void caffe_add_scalar(const int_tp n, const int16_t alpha, int16_t* y);
template void caffe_add_scalar(const int_tp n, const int32_t alpha, int32_t* y);
template void caffe_add_scalar(const int_tp n, const int64_t alpha, int64_t* y);

// COPY
template<typename Dtype>
void caffe_copy(const int_tp n, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
  }
}

template void caffe_copy<int8_t>(const int_tp n, const int8_t* X,
                                     int8_t* Y);
template void caffe_copy<int16_t>(const int_tp n, const int16_t* X,
                                     int16_t* Y);
template void caffe_copy<int32_t>(const int_tp n, const int32_t* X,
                                     int32_t* Y);
template void caffe_copy<int64_t>(const int_tp n, const int64_t* X,
                                     int64_t* Y);
template void caffe_copy<uint8_t>(const int_tp n, const uint8_t* X,
                                     uint8_t* Y);
template void caffe_copy<uint16_t>(const int_tp n, const uint16_t* X,
                                     uint16_t* Y);
template void caffe_copy<uint32_t>(const int_tp n, const uint32_t* X,
                                     uint32_t* Y);
template void caffe_copy<uint64_t>(const int_tp n, const uint64_t* X,
                                     uint64_t* Y);
template void caffe_copy<half_fp>(const int_tp n,
                                      const half_fp* X, half_fp* Y);
template void caffe_copy<float>(const int_tp n, const float* X,
                                    float* Y);
template void caffe_copy<double>(const int_tp n, const double* X,
                                     double* Y);


// SCAL
template<>
void caffe_scal<float>(const int_tp n, const float alpha, float *X) {
  cblas_sscal(n, alpha, X, 1);
}
template<>
void caffe_scal<double>(const int_tp n, const double alpha, double *X) {
  cblas_dscal(n, alpha, X, 1);
}
template<typename Dtype>
void caffe_scal(const int_tp n, const Dtype alpha, Dtype *X) {
#pragma omp parallel for
  for (int_tp i = 0; i < n; ++i) {
    X[i] *= alpha;
  }
}
template void caffe_scal<half_fp>(const int_tp n, const half_fp alpha,
                                  half_fp *X);
template void caffe_scal<int8_t>(const int_tp n, const int8_t alpha,
                                 int8_t *X);
template void caffe_scal<int16_t>(const int_tp n, const int16_t alpha,
                                  int16_t *X);
template void caffe_scal<int32_t>(const int_tp n, const int32_t alpha,
                                  int32_t *X);
template void caffe_scal<int64_t>(const int_tp n, const int64_t alpha,
                                  int64_t *X);

// AXPY
template<>
void caffe_axpby<float>(const int_tp n, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(n, alpha, X, 1, beta, Y, 1);
}
template<>
void caffe_axpby<double>(const int_tp n, const double alpha,
                             const double* X, const double beta, double* Y) {
  cblas_daxpby(n, alpha, X, 1, beta, Y, 1);
}

// ADD
template<>
void caffe_add<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsAdd(n, a, b, Y);
}

template<>
void caffe_add<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdAdd(n, a, b, Y);
}

// SUB
template<>
void caffe_sub<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsSub(n, a, b, Y);
}
template<>
void caffe_sub<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdSub(n, a, b, Y);
}

// MUL
template<>
void caffe_mul<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsMul(n, a, b, Y);
}
template<>
void caffe_mul<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdMul(n, a, b, Y);
}

template<>
void caffe_div<float>(const int_tp n, const float* a, const float* b,
                      float* Y) {
  vsDiv(n, a, b, Y);
}

template<>
void caffe_div<double>(const int_tp n, const double* a, const double* b,
                       double* Y) {
  vdDiv(n, a, b, Y);
}

template<>
void caffe_powx<float>(const int_tp n, const float* a, const float b,
                       float* Y) {
  vsPowx(n, a, b, Y);
}

template<>
void caffe_powx<double>(const int_tp n, const double* a, const double b,
                        double* Y) {
  vdPowx(n, a, b, Y);
}

template<>
void caffe_sqr<float>(const int_tp n, const float* a, float* Y) {
  vsSqr(n, a, Y);
}

template<>
void caffe_sqr<double>(const int_tp n, const double* a, double* Y) {
  vdSqr(n, a, Y);
}

template <>
void caffe_sqrt<float>(const int_tp n, const float* a, float* Y) {
  vsSqrt(n, a, Y);
}

template <>
void caffe_sqrt<double>(const int_tp n, const double* a, double* Y) {
  vdSqrt(n, a, Y);
}

template <>
void caffe_exp<float>(const int_tp n, const float* a, float* Y) {
  vsExp(n, a, Y);
}

template<>
void caffe_exp<double>(const int_tp n, const double* a, double* Y) {
  vdExp(n, a, Y);
}

template<>
void caffe_log<float>(const int_tp n, const float* a, float* Y) {
  vsLn(n, a, Y);
}

template<>
void caffe_log<double>(const int_tp n, const double* a, double* Y) {
  vdLn(n, a, Y);
}

template<>
void caffe_abs<float>(const int_tp n, const float* a, float* Y) {
  vsAbs(n, a, Y);
}

template<>
void caffe_abs<double>(const int_tp n, const double* a, double* Y) {
  vdAbs(n, a, Y);
}

template<>
float caffe_strided_dot<float>(const int_tp n, const float* X,
                                   const int_tp incx, const float* Y,
                                   const int_tp incy) {
  return cblas_sdot(n, X, incx, Y, incy);
}

template<>
double caffe_strided_dot<double>(const int_tp n, const double* X,
                                     const int_tp incx, const double* Y,
                                     const int_tp incy) {
  return cblas_ddot(n, X, incx, Y, incy);
}

template<typename Dtype>
typename std::enable_if<float_is_same<Dtype>::value, Dtype>::type
caffe_dot(const int_tp n, const Dtype* X, const Dtype* Y) {
  return caffe_strided_dot(n, X, 1, Y, 1);
}

template
half_fp caffe_dot<half_fp>(const int_tp n,
                               const half_fp* X, const half_fp* Y);
template
float caffe_dot<float>(const int_tp n, const float* X, const float* Y);
template
double caffe_dot<double>(const int_tp n, const double* X, const double* Y);

template<typename Dtype>
typename std::enable_if<signed_integer_is_same<Dtype>::value, Dtype>::type
caffe_dot(const int_tp n, const Dtype* X, const Dtype* Y) {
  Dtype r;
  #pragma omp for reduction(+ : r)
  for (size_t i = 0; i < n; i++) {
    r += X[i] * Y[i];
  }
  return r;
}

template int8_t caffe_dot<int8_t>(const int_tp n, const int8_t* X,
                              const int8_t* Y);
template int16_t caffe_dot<int16_t>(const int_tp n, const int16_t* X,
                              const int16_t* Y);
template int32_t caffe_dot<int32_t>(const int_tp n, const int32_t* X,
                              const int32_t* Y);
template int64_t caffe_dot<int64_t>(const int_tp n, const int64_t* X,
                              const int64_t* Y);


// ASUM
template<>
half_fp caffe_asum<half_fp>(const int_tp n, const half_fp* x) {
  half_fp sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += fabs(x[i]);
  return sum;
}
template<>
float caffe_asum<float>(const int_tp n, const float* x) {
  return cblas_sasum(n, x, 1);
}
template<>
double caffe_asum<double>(const int_tp n, const double* x) {
  return cblas_dasum(n, x, 1);
}
template<typename Dtype>
Dtype caffe_asum(const int_tp n, const Dtype* x) {
  Dtype sum = 0;
  for (int_tp i = 0; i < n; i++)
    sum += abs(x[i]);
  return sum;
}
template int8_t caffe_asum<int8_t>(const int_tp n, const int8_t* x);
template int16_t caffe_asum<int16_t>(const int_tp n, const int16_t* x);
template int32_t caffe_asum<int32_t>(const int_tp n, const int32_t* x);
template int64_t caffe_asum<int64_t>(const int_tp n, const int64_t* x);


template<>
void caffe_scale<float>(const int_tp n, const float alpha, const float *X,
                            float* Y) {
  cblas_scopy(n, X, 1, Y, 1);
  cblas_sscal(n, alpha, Y, 1);
}

template<>
void caffe_scale<double>(const int_tp n, const double alpha,
                             const double *X, double* Y) {
  cblas_dcopy(n, X, 1, Y, 1);
  cblas_dscal(n, alpha, Y, 1);
}

}  // namespace caffe
