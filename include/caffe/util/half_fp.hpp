#ifndef CAFFE_UTIL_HALF_FP_HPP_
#define CAFFE_UTIL_HALF_FP_HPP_

#include "caffe/definitions.hpp"
#include "3rdparty/half/half.hpp"

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

typedef half_float::half half_fp;


namespace caffe {


}

#endif   // CAFFE_UTIL_HALF_FP_HPP_
