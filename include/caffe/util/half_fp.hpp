#ifndef CAFFE_UTIL_HALF_FP_HPP_
#define CAFFE_UTIL_HALF_FP_HPP_

#include <memory>
#include "3rdparty/half/half.hpp"

namespace caffe {

#define HALF_DIG 3
#define HALF_MANT_DIG 11
#define HALF_MAX_10_EXP 4
#define HALF_MAX_EXP 16
#define HALF_MIN_10_EXP -4
#define HALF_MIN_EXP -13
#define HALF_RADIX 2
#define HALF_MAX 65504.f // 0x1.ffcp15
#define HALF_MIN 6.10352e-5f // 0x1.0p-14
#define HALF_EPSILON 0x1.0p-10f

inline float fixup_arg_type(float v) {
  return v;
}

inline float fixup_arg_type(half_fp v) {
  return float(v);
}

inline double fixup_arg_type(double v) {
  return v;
}

inline int fixup_arg_type(int v) {
  return v;
}

inline unsigned int fixup_arg_type(unsigned int v) {
  return v;
}

inline long long fixup_arg_type(long long v) {
  return v;
}

inline unsigned long long fixup_arg_type(unsigned long long v) {
  return v;
}

inline long fixup_arg_type(long v) {
  return v;
}

inline unsigned long fixup_arg_type(unsigned long v) {
  return v;
}

inline float fixup_arg_type(const caffe::detail::expr& expr) {
  return float(expr);
}

inline const void * fixup_arg_type(const std::shared_ptr<void>& share_ptr) {
  return (const void*)share_ptr.get();
}

}

#endif   // CAFFE_UTIL_HALF_FP_HPP_
