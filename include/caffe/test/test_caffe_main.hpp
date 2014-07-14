// Copyright 2014 BVLC and contributors.

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdio>

#include "caffe/common.hpp"

using std::cout;
using std::endl;

int main(int argc, char** argv);

namespace caffe {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiDeviceTest() {
    Caffe::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

typedef ::testing::Types<float, double> TestDtypes;

struct FloatCPU {
  typedef float Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

struct DoubleCPU {
  typedef double Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

struct FloatGPU {
  typedef float Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

struct DoubleGPU {
  typedef double Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

typedef ::testing::Types<FloatCPU, DoubleCPU, FloatGPU, DoubleGPU>
    TestDtypesAndDevices;

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
