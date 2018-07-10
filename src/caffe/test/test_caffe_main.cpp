#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/backend/device.hpp"

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

template<typename Dtype>
bool caffe::isSupported(void) {
  return false;
}

#ifdef USE_HALF
template<>
bool caffe::isSupported<caffe::half_fp>(void) {
  return caffe::Caffe::GetDefaultDevice()->
      CheckCapability(caffe::DEVICE_FP16_SUPPORT);
}
template<>
bool caffe::isSupported<caffe::CPUDevice<caffe::half_fp> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<caffe::half_fp> >(void) {
  return caffe::Caffe::GetDefaultDevice()->
      CheckCapability(caffe::DEVICE_FP16_SUPPORT);
}
#endif  // USE_HALF

#ifdef USE_SINGLE
template<>
bool caffe::isSupported<float>(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::CPUDevice<float> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<float> >(void) {
  return true;
}
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
template<>
bool caffe::isSupported<double>(void) {
  return caffe::Caffe::GetDefaultDevice()->
      CheckCapability(DEVICE_FP64_SUPPORT);
}
template<>
bool caffe::isSupported<caffe::CPUDevice<double> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<double> >(void) {
  return caffe::Caffe::GetDefaultDevice()->
      CheckCapability(DEVICE_FP64_SUPPORT);
}
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
template<>
bool caffe::isSupported<uint8_t>(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::CPUDevice<uint8_t> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<uint8_t> >(void) {
  return true;
}
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
template<>
bool caffe::isSupported<uint16_t>(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::CPUDevice<uint16_t> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<uint16_t> >(void) {
  return true;
}
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
template<>
bool caffe::isSupported<uint32_t>(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::CPUDevice<uint32_t> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<uint32_t> >(void) {
  return true;
}
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
template<>
bool caffe::isSupported<uint64_t>(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::CPUDevice<uint64_t> >(void) {
  return true;
}
template<>
bool caffe::isSupported<caffe::GPUDevice<uint64_t> >(void) {
  return true;
}
#endif  // USE_INT_QUANT_64


#if defined(USE_LEVELDB) && defined(USE_LMDB)
template<>
bool caffe::isSupported<caffe::TypeLevelDB>(void) {
  return true;
}

template<>
bool caffe::isSupported<caffe::TypeLMDB>(void) {
  return true;
}
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
  Caffe::SetDevices(vector<int>{device});
#endif
  // invoke the test.
  int r =  RUN_ALL_TESTS();
#ifdef USE_OPENCL
  // Call explicitly for OCL + FFT
  caffe::Caffe::TeardownDevice(device);
#endif
  return r;
}
