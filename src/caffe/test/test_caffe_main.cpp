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

// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#include <gmock/gmock.h>
#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
#ifndef CPU_ONLY
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}

#ifndef CPU_ONLY
using caffe::CAFFE_TEST_CUDA_PROP;
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifdef USE_MLSL
  caffe::mn::init(&argc, &argv);
#endif

#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  int device;
  cudaGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    cudaSetDevice(device);
    cout << "Setting to use device " << device << endl;
  } else if (CUDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = CUDA_TEST_DEVICE;
  }
  cudaGetDevice(&device);
  cout << "Current device id: " << device << endl;
  cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
  cout << "Current device name: " << CAFFE_TEST_CUDA_PROP.name << endl;
#endif
  // invoke the test.
  return RUN_ALL_TESTS();
}
