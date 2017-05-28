#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"

#ifndef TEST_DEVICE
#define TEST_DEVICE 0
#endif

namespace caffe {
#ifndef CPU_ONLY
#ifdef USE_CUDA
cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif  // USE_CUDA
#endif
}

#ifdef USE_GREENTEA
template <typename Dtype>
bool caffe::isSupported(void) {
  return true;
}

template <>
bool caffe::isSupported<float>(void) {
  return true;
}

template <>
bool caffe::isSupported<caffe::GPUDevice<float>>(void) {
  return isSupported<float>();
}

template <>
bool caffe::isSupported<double>(void) {
  return caffe::Caffe::GetDefaultDevice()->backend() != caffe::BACKEND_OpenCL ||
         caffe::Caffe::GetDefaultDevice()->CheckCapability("cl_khr_fp64");
}

template <>
bool caffe::isSupported<caffe::GPUDevice<double>>(void) {
  return caffe::isSupported<double>();
}

template <>
bool caffe::isSupported<caffe::CPUDevice<float>>(void) {
  return true;
}

template <>
bool caffe::isSupported<caffe::CPUDevice<double>>(void) {
  return true;
}

#if defined(USE_LEVELDB) && defined(USE_LMDB)
template <>
bool caffe::isSupported<caffe::TypeLevelDB>(void) {
  return true;
}

template <>
bool caffe::isSupported<caffe::TypeLMDB>(void) {
  return true;
}
#endif
#endif

#ifndef CPU_ONLY
#ifdef USE_CUDA
using caffe::CAFFE_TEST_CUDA_PROP;
#endif  // USE_CUDA
#endif

using caffe::Caffe;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
  int device = 0;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
  } else if (TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = TEST_DEVICE;
  }
  cout << "Setting to use device " << device << endl;
  Caffe::SetDevices(std::vector<int>{device});
  Caffe::SetDevice(device);
#endif
  // invoke the test.
  int r =  RUN_ALL_TESTS();
#ifdef USE_GREENTEA
  // Call explicitly for OCL + FFT
  caffe::Caffe::TeardownDevice(device);
#endif
  return r;
}
