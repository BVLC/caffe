// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif

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

#ifdef CPU_ONLY

typedef ::testing::Types<FloatCPU, DoubleCPU> TestDtypesAndDevices;

#else

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

#endif

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
