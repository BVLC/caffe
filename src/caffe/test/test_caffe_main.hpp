// Copyright 2013 Yangqing Jia

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdio>

using std::cout;
using std::endl;

namespace caffe {

cudaDeviceProp CAFFE_TEST_CUDA_PROP;

}  // namespace caffe

using caffe::CAFFE_TEST_CUDA_PROP;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::google::InitGoogleLogging(argv[0]);
  // Before starting testing, let's first print out a few cuda defice info.
  int device;
  cudaGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    cudaSetDevice(device);
    cout << "Setting to use device " << device << endl;
  }
  cudaGetDevice(&device);
  cout << "Current device id: " << device << endl;
  cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
  // invoke the test.
  return RUN_ALL_TESTS();
}

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
