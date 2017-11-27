#ifndef CAFFE_BACKEND_BACKEND_HPP_
#define CAFFE_BACKEND_BACKEND_HPP_

#include <cstddef>
#include <type_traits>

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/util/half_fp.hpp"

namespace caffe {

enum Backend {
  BACKEND_CUDA,
  BACKEND_OPENCL,
  BACKEND_HIP,
  BACKEND_CPU
};

typedef tuple<string, string, uint64_t>              KernelArg;
typedef vector<KernelArg>                            KernelArgs;

const uint64_t KERNEL_ARG_NONE          =            0ULL       ;
const uint64_t KERNEL_ARG_CONST         =            1ULL <<   0;
const uint64_t KERNEL_ARG_GLOBAL_MEM    =            1ULL <<   1;
const uint64_t KERNEL_ARG_LOCAL_MEM     =            1ULL <<   2;
const uint64_t KERNEL_ARG_MEM_OFFSET    =            1ULL <<   3;


template<typename Dtype>
struct const_enable_if
  : std::enable_if<std::is_same<Dtype, const bool>::value ||
                   std::is_same<Dtype, const char>::value ||
                   std::is_same<Dtype, const int8_t>::value ||
                   std::is_same<Dtype, const int16_t>::value ||
                   std::is_same<Dtype, const int32_t>::value ||
                   std::is_same<Dtype, const int64_t>::value ||
                   std::is_same<Dtype, const uint8_t>::value ||
                   std::is_same<Dtype, const uint16_t>::value ||
                   std::is_same<Dtype, const uint32_t>::value ||
                   std::is_same<Dtype, const uint64_t>::value ||
                   std::is_same<Dtype, const half_fp>::value ||
                   std::is_same<Dtype, const float>::value ||
                   std::is_same<Dtype, const double>::value ||
                   std::is_same<Dtype, const void>::value>
{};

template<typename Dtype>
struct non_const_enable_if
  : std::enable_if<std::is_same<Dtype, bool>::value ||
                   std::is_same<Dtype, char>::value ||
                   std::is_same<Dtype, int8_t>::value ||
                   std::is_same<Dtype, int16_t>::value ||
                   std::is_same<Dtype, int32_t>::value ||
                   std::is_same<Dtype, int64_t>::value ||
                   std::is_same<Dtype, uint8_t>::value ||
                   std::is_same<Dtype, uint16_t>::value ||
                   std::is_same<Dtype, uint32_t>::value ||
                   std::is_same<Dtype, uint64_t>::value ||
                   std::is_same<Dtype, half_fp>::value ||
                   std::is_same<Dtype, float>::value ||
                   std::is_same<Dtype, double>::value ||
                   std::is_same<Dtype, void>::value>
{};

template<typename Dtype>
struct const_is_same {
  constexpr static bool value =
      std::is_same<Dtype, const bool>::value ||
      std::is_same<Dtype, const char>::value ||
      std::is_same<Dtype, const int8_t>::value ||
      std::is_same<Dtype, const int16_t>::value ||
      std::is_same<Dtype, const int32_t>::value ||
      std::is_same<Dtype, const int64_t>::value ||
      std::is_same<Dtype, const uint8_t>::value ||
      std::is_same<Dtype, const uint16_t>::value ||
      std::is_same<Dtype, const uint32_t>::value ||
      std::is_same<Dtype, const uint64_t>::value ||
      std::is_same<Dtype, const half_fp>::value ||
      std::is_same<Dtype, const float>::value ||
      std::is_same<Dtype, const double>::value ||
      std::is_same<Dtype, const void>::value;
};

template<typename Dtype>
struct non_const_is_same {
  constexpr static bool value =
      std::is_same<Dtype, bool>::value ||
      std::is_same<Dtype, char>::value ||
      std::is_same<Dtype, int8_t>::value ||
      std::is_same<Dtype, int16_t>::value ||
      std::is_same<Dtype, int32_t>::value ||
      std::is_same<Dtype, int64_t>::value ||
      std::is_same<Dtype, uint8_t>::value ||
      std::is_same<Dtype, uint16_t>::value ||
      std::is_same<Dtype, uint32_t>::value ||
      std::is_same<Dtype, uint64_t>::value ||
      std::is_same<Dtype, half_fp>::value ||
      std::is_same<Dtype, float>::value ||
      std::is_same<Dtype, double>::value ||
      std::is_same<Dtype, void>::value;
};
}  // namespace caffe

#endif  // CAFFE_BACKEND_BACKEND_HPP_
