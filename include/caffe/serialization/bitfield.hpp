/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

