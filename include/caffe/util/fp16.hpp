#ifndef CAFFE_UTIL_FP16_H_
#define CAFFE_UTIL_FP16_H_

#include <boost/shared_ptr.hpp>
#include "caffe/3rdparty/half/half.hpp"
using half_float::half;

#ifdef _MSC_VER
	#define HALF_MAX	65503
	#define HALF_MIN	0.00061036
#else
	#define HALF_MAX	0x1.ffcp15
	#define HALF_MIN	0x1.0p-14
#endif

#include <boost/shared_ptr.hpp>

inline float fixup_arg_type(float v) {
  return v;
}

inline float fixup_arg_type(half_float::half v) {
  return static_cast<float>(v);
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

inline float fixup_arg_type(const half_float::detail::expr& expr) {
  return static_cast<float>(expr);
}

inline const void * fixup_arg_type(const boost::shared_ptr<void>& share_ptr) {
  return (const void*)share_ptr.get();
}

#endif   // CAFFE_UTIL_HDF5_H_
