#ifndef CAFFE_UTIL_INLINE_MATH_HPP_
#define CAFFE_UTIL_INLINE_MATH_HPP_

namespace caffe {

inline uint32_t flp2(uint32_t x) {
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >> 16);
  return x - (x >> 1);
}

inline uint64_t flp2(uint64_t x) {
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >> 16);
  x = x | (x >> 32);
  return x - (x >> 1);
}


}  // namespace caffe

#endif  // CAFFE_UTIL_INLINE_MATH_HPP_
