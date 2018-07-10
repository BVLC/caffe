#include <boost/math/special_functions/next.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// SET
template<typename Dtype>
void caffe_set(const int_tp n, const Dtype alpha, Dtype* y) {
  if (alpha == 0) {
    memset(y, 0, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int_tp i = 0; i < n; ++i) {
    y[i] = alpha;
  }
}

template void caffe_set<int8_t>(const int_tp n, const int8_t alpha,
                                 int8_t* y);
template void caffe_set<uint8_t>(const int_tp n, const uint8_t alpha,
                                uint8_t* y);
template void caffe_set<int16_t>(const int_tp n, const int16_t alpha,
                                 int16_t* y);
template void caffe_set<uint16_t>(const int_tp n, const uint16_t alpha,
                                  uint16_t* y);
template void caffe_set<int32_t>(const int_tp n, const int32_t alpha,
                                 int32_t* y);
template void caffe_set<uint32_t>(const int_tp n, const uint32_t alpha,
                                  uint32_t* y);
template void caffe_set<int64_t>(const int_tp n, const int64_t alpha,
                                 int64_t* y);
template void caffe_set<uint64_t>(const int_tp n, const uint64_t alpha,
                                  uint64_t* y);
template void caffe_set<half_fp>(const int_tp n,
                             const half_fp alpha, half_fp* y);
template void caffe_set<float>(const int_tp n, const float alpha, float* y);
template void caffe_set<double>(const int_tp n, const double alpha, double* y);

// ADD SCALAR
template<typename Dtype>
void caffe_add_scalar(const int_tp n, const Dtype alpha, Dtype* y) {
  for (size_t i = 0; i < n; ++i) {
    y[i] += alpha;
  }
}

template void caffe_add_scalar(const int_tp n, const half_fp alpha, half_fp* y);
template void caffe_add_scalar(const int_tp n, const float alpha, float* y);
template void caffe_add_scalar(const int_tp n, const double alpha, double* y);
template void caffe_add_scalar(const int_tp n, const uint8_t alpha, uint8_t* y);
template void caffe_add_scalar(const int_tp n, const uint16_t alpha, uint16_t* y);
template void caffe_add_scalar(const int_tp n, const uint32_t alpha, uint32_t* y);
template void caffe_add_scalar(const int_tp n, const uint64_t alpha, uint64_t* y);

// COPy
template<typename Dtype>
void caffe_copy(const int_tp n, const Dtype* x, Dtype* y) {
  if (x != y) {
    memcpy(y, x, sizeof(Dtype) * n);  // NOLINT(caffe/alt_fn)
  }
}

template void caffe_copy<int8_t>(const int_tp n, const int8_t* x,
                                     int8_t* y);
template void caffe_copy<int16_t>(const int_tp n, const int16_t* x,
                                     int16_t* y);
template void caffe_copy<int32_t>(const int_tp n, const int32_t* x,
                                     int32_t* y);
template void caffe_copy<int64_t>(const int_tp n, const int64_t* x,
                                     int64_t* y);
template void caffe_copy<uint8_t>(const int_tp n, const uint8_t* x,
                                     uint8_t* y);
template void caffe_copy<uint16_t>(const int_tp n, const uint16_t* x,
                                     uint16_t* y);
template void caffe_copy<uint32_t>(const int_tp n, const uint32_t* x,
                                     uint32_t* y);
template void caffe_copy<uint64_t>(const int_tp n, const uint64_t* x,
                                     uint64_t* y);
template void caffe_copy<half_fp>(const int_tp n,
                                      const half_fp* x, half_fp* y);
template void caffe_copy<float>(const int_tp n, const float* x,
                                    float* y);
template void caffe_copy<double>(const int_tp n, const double* x,
                                     double* y);


// ADD
template<typename Dtype>
void caffe_add(const int_tp n, const Dtype* a,
               const Dtype* b, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}

template void caffe_add<half_fp>(const int_tp n, const half_fp* a,
                                 const half_fp* b, half_fp* y);
template void caffe_add<uint8_t>(const int_tp n, const uint8_t* a,
                                 const uint8_t* b, uint8_t* y);
template void caffe_add<uint16_t>(const int_tp n, const uint16_t* a,
                                 const uint16_t* b, uint16_t* y);
template void caffe_add<uint32_t>(const int_tp n, const uint32_t* a,
                                 const uint32_t* b, uint32_t* y);
template void caffe_add<uint64_t>(const int_tp n, const uint64_t* a,
                                 const uint64_t* b, uint64_t* y);

template<>
void caffe_add<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsAdd(n, a, b, y);
}

template<>
void caffe_add<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdAdd(n, a, b, y);
}

// SUB
template<typename Dtype>
void caffe_sub(const int_tp n, const Dtype* a,
               const Dtype* b, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = a[i] - b[i];
  }
}

template void caffe_sub<half_fp>(const int_tp n, const half_fp* a,
                                 const half_fp* b, half_fp* y);
template void caffe_sub<uint8_t>(const int_tp n, const uint8_t* a,
                                 const uint8_t* b, uint8_t* y);
template void caffe_sub<uint16_t>(const int_tp n, const uint16_t* a,
                                 const uint16_t* b, uint16_t* y);
template void caffe_sub<uint32_t>(const int_tp n, const uint32_t* a,
                                 const uint32_t* b, uint32_t* y);
template void caffe_sub<uint64_t>(const int_tp n, const uint64_t* a,
                                 const uint64_t* b, uint64_t* y);

template<>
void caffe_sub<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsSub(n, a, b, y);
}
template<>
void caffe_sub<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdSub(n, a, b, y);
}


// MUL
template<typename Dtype>
void caffe_mul(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = a[i] * b[i];
  }
}

template void caffe_mul<half_fp>(const int_tp n, const half_fp* a,
                                 const half_fp* b, half_fp* y);
template void caffe_mul<uint8_t>(const int_tp n, const uint8_t* a,
                                 const uint8_t* b, uint8_t* y);
template void caffe_mul<uint16_t>(const int_tp n, const uint16_t* a,
                                 const uint16_t* b, uint16_t* y);
template void caffe_mul<uint32_t>(const int_tp n, const uint32_t* a,
                                 const uint32_t* b, uint32_t* y);
template void caffe_mul<uint64_t>(const int_tp n, const uint64_t* a,
                                 const uint64_t* b, uint64_t* y);

template<>
void caffe_mul<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsMul(n, a, b, y);
}
template<>
void caffe_mul<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdMul(n, a, b, y);
}

// DIV
template<typename Dtype>
void caffe_div(const int_tp n, const Dtype* a, const Dtype* b, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = a[i] / b[i];
  }
}

template void caffe_div<half_fp>(const int_tp n, const half_fp* a,
                                 const half_fp* b, half_fp* y);
template void caffe_div<uint8_t>(const int_tp n, const uint8_t* a,
                                 const uint8_t* b, uint8_t* y);
template void caffe_div<uint16_t>(const int_tp n, const uint16_t* a,
                                 const uint16_t* b, uint16_t* y);
template void caffe_div<uint32_t>(const int_tp n, const uint32_t* a,
                                 const uint32_t* b, uint32_t* y);
template void caffe_div<uint64_t>(const int_tp n, const uint64_t* a,
                                 const uint64_t* b, uint64_t* y);

template<>
void caffe_div<float>(const int_tp n, const float* a, const float* b,
                      float* y) {
  vsDiv(n, a, b, y);
}
template<>
void caffe_div<double>(const int_tp n, const double* a, const double* b,
                       double* y) {
  vdDiv(n, a, b, y);
}


// POWX
template<typename Dtype>
void caffe_powx(const int_tp n, const Dtype* a, const Dtype b, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = pow(a[i], b);
  }
}

template void caffe_powx<half_fp>(const int_tp n, const half_fp* a,
                                const half_fp b, half_fp* y);
template void caffe_powx<uint8_t>(const int_tp n, const uint8_t* a,
                                const uint8_t b, uint8_t* y);
template void caffe_powx<uint16_t>(const int_tp n, const uint16_t* a,
                                const uint16_t b, uint16_t* y);
template void caffe_powx<uint32_t>(const int_tp n, const uint32_t* a,
                                const uint32_t b, uint32_t* y);
template void caffe_powx<uint64_t>(const int_tp n, const uint64_t* a,
                                const uint64_t b, uint64_t* y);

template<>
void caffe_powx<float>(const int_tp n, const float* a, const float b,
                       float* y) {
  vsPowx(n, a, b, y);
}
template<>
void caffe_powx<double>(const int_tp n, const double* a, const double b,
                        double* y) {
  vdPowx(n, a, b, y);
}

// SQR
template<typename Dtype>
void caffe_sqr(const int_tp n, const Dtype* a, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = a[i] * a[i];
  }
}

template void caffe_sqr<half_fp>(const int_tp n, const half_fp* a, half_fp* y);
template void caffe_sqr<uint8_t>(const int_tp n, const uint8_t* a, uint8_t* y);
template void caffe_sqr<uint16_t>(const int_tp n, const uint16_t* a, uint16_t* y);
template void caffe_sqr<uint32_t>(const int_tp n, const uint32_t* a, uint32_t* y);
template void caffe_sqr<uint64_t>(const int_tp n, const uint64_t* a, uint64_t* y);


template<>
void caffe_sqr<float>(const int_tp n, const float* a, float* y) {
  vsSqr(n, a, y);
}
template<>
void caffe_sqr<double>(const int_tp n, const double* a, double* y) {
  vdSqr(n, a, y);
}

// SQRT
template<typename Dtype>
void caffe_sqrt(const int_tp n, const Dtype* a, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = std::sqrt(a[i]);
  }
}

template void caffe_sqrt(const int_tp n, const half_fp* a, half_fp* y);
template void caffe_sqrt(const int_tp n, const uint8_t* a, uint8_t* y);
template void caffe_sqrt(const int_tp n, const uint16_t* a, uint16_t* y);
template void caffe_sqrt(const int_tp n, const uint32_t* a, uint32_t* y);
template void caffe_sqrt(const int_tp n, const uint64_t* a, uint64_t* y);

template<>
void caffe_sqrt<float>(const int_tp n, const float* a, float* y) {
  vsSqrt(n, a, y);
}
template<>
void caffe_sqrt<double>(const int_tp n, const double* a, double* y) {
  vdSqrt(n, a, y);
}


// EXP
template<typename Dtype>
void caffe_exp(const int_tp n, const Dtype* a, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = std::exp(a[i]);
  }
}

template void caffe_exp<half_fp>(const int_tp n, const half_fp* a, half_fp* y);
template void caffe_exp<uint8_t>(const int_tp n, const uint8_t* a, uint8_t* y);
template void caffe_exp<uint16_t>(const int_tp n, const uint16_t* a, uint16_t* y);
template void caffe_exp<uint32_t>(const int_tp n, const uint32_t* a, uint32_t* y);
template void caffe_exp<uint64_t>(const int_tp n, const uint64_t* a, uint64_t* y);

template<>
void caffe_exp<float>(const int_tp n, const float* a, float* y) {
  vsExp(n, a, y);
}
template<>
void caffe_exp<double>(const int_tp n, const double* a, double* y) {
  vdExp(n, a, y);
}


// LOG
template<typename Dtype>
void caffe_log(const int_tp n, const Dtype* a, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = log(a[i]);
  }
}

template void caffe_log<half_fp>(const int_tp n, const half_fp* a, half_fp* y);
template void caffe_log<uint8_t>(const int_tp n, const uint8_t* a, uint8_t* y);
template void caffe_log<uint16_t>(const int_tp n, const uint16_t* a, uint16_t* y);
template void caffe_log<uint32_t>(const int_tp n, const uint32_t* a, uint32_t* y);
template void caffe_log<uint64_t>(const int_tp n, const uint64_t* a, uint64_t* y);

template<>
void caffe_log<float>(const int_tp n, const float* a, float* y) {
  vsLn(n, a, y);
}
template<>
void caffe_log<double>(const int_tp n, const double* a, double* y) {
  vdLn(n, a, y);
}


// ABS
template<typename Dtype>
void caffe_abs(const int_tp n, const Dtype* a, Dtype* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = fabs(a[i]);
  }
}

template void caffe_abs<uint8_t>(const int_tp n, const uint8_t* a,
                                 uint8_t* y);
template void caffe_abs<uint16_t>(const int_tp n, const uint16_t* a,
                                  uint16_t* y);
template void caffe_abs<uint32_t>(const int_tp n, const uint32_t* a,
                                  uint32_t* y);
template void caffe_abs<uint64_t>(const int_tp n, const uint64_t* a,
                                  uint64_t* y);

template<>
void caffe_abs<half_fp>(const int_tp n, const half_fp* a,
                                 half_fp* y) {
  for (int_tp i = 0; i < n; ++i) {
    y[i] = fabs(a[i]);
  }
}
template<>
void caffe_abs<float>(const int_tp n, const float* a, float* y) {
  vsAbs(n, a, y);
}
template<>
void caffe_abs<double>(const int_tp n, const double* a, double* y) {
  vdAbs(n, a, y);
}



}  // namespace caffe
