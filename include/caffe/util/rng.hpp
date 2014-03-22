// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <boost/random/mersenne_twister.hpp>
#include "caffe/common.hpp"

namespace caffe {

  typedef boost::mt19937 rng_t;
  inline rng_t& caffe_rng() {
    Caffe::RNG &generator = Caffe::rng_stream();
    return *(caffe::rng_t*) generator.generator();
  }

}  // namespace caffe

#endif  // CAFFE_RNG_HPP_
