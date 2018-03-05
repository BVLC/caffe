#ifndef CAFFE_TRAIT_HELPER_HPP_
#define CAFFE_TRAIT_HELPER_HPP_

#include <type_traits>

#include "caffe/util/half_fp.hpp"

namespace caffe {

template<typename Dtype>
struct float_enable_if
  : std::enable_if<std::is_same<Dtype, half_fp>::value ||
                   std::is_same<Dtype, float>::value ||
                   std::is_same<Dtype, double>::value>
{};

template<typename Dtype>
struct signed_integer_enable_if
  : std::enable_if<std::is_same<Dtype, int8_t>::value ||
                   std::is_same<Dtype, int16_t>::value ||
                   std::is_same<Dtype, int32_t>::value ||
                   std::is_same<Dtype, int64_t>::value>
{};

template<typename Dtype>
struct unsigned_integer_enable_if
  : std::enable_if<std::is_same<Dtype, uint8_t>::value ||
                   std::is_same<Dtype, uint16_t>::value ||
                   std::is_same<Dtype, uint32_t>::value ||
                   std::is_same<Dtype, uint64_t>::value>
{};

template<typename Dtype>
struct integer_enable_if
  : std::enable_if<std::is_same<Dtype, int8_t>::value ||
                   std::is_same<Dtype, int16_t>::value ||
                   std::is_same<Dtype, int32_t>::value ||
                   std::is_same<Dtype, int64_t>::value ||
                   std::is_same<Dtype, uint8_t>::value ||
                   std::is_same<Dtype, uint16_t>::value ||
                   std::is_same<Dtype, uint32_t>::value ||
                   std::is_same<Dtype, uint64_t>::value>
{};

template<typename Dtype>
struct float_is_same {
  constexpr static bool value =
      std::is_same<Dtype, half_fp>::value ||
      std::is_same<Dtype, float>::value ||
      std::is_same<Dtype, double>::value;
};

template<typename Dtype>
struct signed_integer_is_same {
  constexpr static bool value =
      std::is_same<Dtype, int8_t>::value ||
      std::is_same<Dtype, int16_t>::value ||
      std::is_same<Dtype, int32_t>::value ||
      std::is_same<Dtype, int64_t>::value;
};

template<typename Dtype>
struct unsigned_integer_is_same {
  constexpr static bool value =
      std::is_same<Dtype, uint8_t>::value ||
      std::is_same<Dtype, uint16_t>::value ||
      std::is_same<Dtype, uint32_t>::value ||
      std::is_same<Dtype, uint64_t>::value;
};

template<typename Dtype>
struct integer_is_same {
  constexpr static bool value =
      std::is_same<Dtype, int8_t>::value ||
      std::is_same<Dtype, int16_t>::value ||
      std::is_same<Dtype, int32_t>::value ||
      std::is_same<Dtype, int64_t>::value ||
      std::is_same<Dtype, uint8_t>::value ||
      std::is_same<Dtype, uint16_t>::value ||
      std::is_same<Dtype, uint32_t>::value ||
      std::is_same<Dtype, uint64_t>::value;
};

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

template<typename Dtype>
struct proto_type_is_same {
  constexpr static bool value =
      std::is_same<Dtype, uint8_t>::value ||
      std::is_same<Dtype, uint16_t>::value ||
      std::is_same<Dtype, uint32_t>::value ||
      std::is_same<Dtype, uint64_t>::value ||
      std::is_same<Dtype, half_fp>::value ||
      std::is_same<Dtype, float>::value ||
      std::is_same<Dtype, double>::value;
};


template<typename Dtype>
inline bool is_signed_integer_type() {
  return std::is_same<Dtype, int8_t>::value
      || std::is_same<Dtype, int16_t>::value
      || std::is_same<Dtype, int32_t>::value
      || std::is_same<Dtype, int64_t>::value;
}

template<typename Dtype>
inline bool is_unsigned_integer_type() {
  return std::is_same<Dtype, uint8_t>::value
      || std::is_same<Dtype, uint16_t>::value
      || std::is_same<Dtype, uint32_t>::value
      || std::is_same<Dtype, uint64_t>::value;
}

template<typename Dtype>
inline bool is_integer_type() {
  return std::is_same<Dtype, int8_t>::value
      || std::is_same<Dtype, int16_t>::value
      || std::is_same<Dtype, int32_t>::value
      || std::is_same<Dtype, int64_t>::value
      || std::is_same<Dtype, uint8_t>::value
      || std::is_same<Dtype, uint16_t>::value
      || std::is_same<Dtype, uint32_t>::value
      || std::is_same<Dtype, uint64_t>::value;
}

template<typename Dtype>
inline bool is_float_type() {
  return std::is_same<Dtype, half_fp>::value
      || std::is_same<Dtype, float>::value
      || std::is_same<Dtype, double>::value;
}

}  // namespace caffe

#endif  // CAFFE_TRAIT_HELPER_HPP_
