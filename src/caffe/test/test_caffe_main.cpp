// Copyright 2014 BVLC and contributors.

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#include "test_caffe_main.hpp"

namespace caffe {
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
}

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
