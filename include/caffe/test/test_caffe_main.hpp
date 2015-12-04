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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  #include "caffe_config.h"
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  #include "caffe_config.h"
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  #include "caffe_config.h"
=======
>>>>>>> pod/device/blob.hpp
=======
  #include "caffe_config.h"
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  #include "caffe_config.h"
=======
>>>>>>> pod/caffe-merge
  #include <cmake_test_defines.hpp.gen.cmake>
>>>>>>> origin/BVLC/parallel
=======
  #include "caffe_config.h"
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  #include "caffe_config.h"
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
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

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
};

#ifdef CPU_ONLY

typedef ::testing::Types<CPUDevice<float>,
                         CPUDevice<double> > TestDtypesAndDevices;

#else

template <typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
};

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         GPUDevice<float>, GPUDevice<double> >
                         TestDtypesAndDevices;

#endif

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
