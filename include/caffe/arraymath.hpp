#ifndef CAFFE_ARRAYMATH_HPP_
#define CAFFE_ARRAYMATH_HPP_

#include "caffe/array.hpp"

namespace caffe {

// Define all math functions
template<typename T>
struct ARMath {
  typedef Expression<T> PE;
  // Unary operations
  static PE abs(const ArrayBase<T> &a);
  static PE exp(const ArrayBase<T> &a);
  static PE log(const ArrayBase<T> &a);
  static PE negate(const ArrayBase<T> &a);
  static PE sign(const ArrayBase<T> &a);
  static PE sqrt(const ArrayBase<T> &a);

  // Binary operations
#define DECLARE_BINARY_OP(name)\
  static PE name(const ArrayBase<T> &a, const ArrayBase<T> &b);\
  static PE name(T a, const ArrayBase<T> &b);\
  static PE name(const ArrayBase<T> &a, T b)

  DECLARE_BINARY_OP(add);
  DECLARE_BINARY_OP(div);
  DECLARE_BINARY_OP(maximum);
  DECLARE_BINARY_OP(minimum);
  DECLARE_BINARY_OP(mul);
  DECLARE_BINARY_OP(pow);
  DECLARE_BINARY_OP(sub);

  // TODO: Reductions and partial reductions

  // Matrix operations
  // gemm: compute alpha*op( A )*op( B ) + beta*C (A, B and C are in row major)
  static void gemm(bool transA, bool transB, T alpha, const Array<T> & a,
                   const Array<T> & b, T beta, Array<T> * c);
  static void im2col(const Array<T> & data, int kernel_h, int kernel_w,
                     int pad_h, int pad_w, int stride_h, int stride_w,
                     Array<T> * data_col);
  static void col2im(const Array<T> & data_col, int patch_h, int patch_w,
                     int pad_h, int pad_w, int stride_h, int stride_w,
                     Array<T> * data);
  static void conv(const Array<T> & im, const Array<T> & kernel, int pad_h,
                   int pad_w, int stride_h, int stride_w, Array<T> * out);
};
}  // namespace caffe

#endif  // CAFFE_ARRAYMATH_HPP_
