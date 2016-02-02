#ifndef CAFFE_SERIALIZATION_BITFIELD_HPP_
#define CAFFE_SERIALIZATION_BITFIELD_HPP_

#include <vector>
#include "caffe/util/math_functions.hpp"

struct bitfield {
  std::vector<unsigned char> buffer;

  explicit bitfield(size_t bits)
    : buffer((bits + __CHAR_BIT__ - 1) / __CHAR_BIT__) {
  }

  bitfield(const char* data, size_t bytes)
    : buffer(bytes) {
    caffe::caffe_copy<char>(
      bytes, data, reinterpret_cast<char*>(&buffer.front()));
  }

  char* raw() {
    return reinterpret_cast<char*>(&buffer.front());
  }

  size_t bytes() const {
    return buffer.size();
  }

  void shift(uint32_t* bit, uint32_t* byte) {
    if (*bit == __CHAR_BIT__ - 1) {
      ++(*byte);
      bit = 0;
    } else {
      ++(*bit);
    }
  }

  void set_bits(int32_t val, uint32_t mask_size, uint32_t at_bit) {
    uint32_t byte = at_bit / __CHAR_BIT__;
    uint32_t bit = at_bit % __CHAR_BIT__;

    buffer[byte] = buffer[byte] | (((val < 0) ? 1u : 0u) << bit);
    shift(&bit, &byte);

    uint32_t aux = abs(val);
    for (int i = 0; i < mask_size; ++i) {
      buffer[byte] = buffer[byte] | (unsigned char)((aux & 1) << bit);
      aux >>= 1;
      shift(&bit, &byte);
    }
  }

  int32_t get_bits(uint32_t mask_size, uint32_t at_bit) {
    uint32_t byte = at_bit / __CHAR_BIT__;
    uint32_t bit = at_bit % __CHAR_BIT__;

    bool negative = ((buffer[byte] >> bit) & 1) > 0;
    shift(&bit, &byte);

    uint32_t ret = (mask_size == 0) ? 1u : 0u;
    for (int i = 0; i < mask_size; ++i) {
      ret = ret | ((uint32_t(buffer[byte] >> bit) & 1u) << i);
      shift(&bit, &byte);
    }
    return (negative) ? -int32_t(ret) : int32_t(ret);
  }
};

#endif  // CAFFE_SERIALIZATION_BITFIELD_HPP_

