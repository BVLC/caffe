// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/backend/device.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define TEST_DEVICE -1
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define ABS_TEST_DATA_DIR "src/caffe/test/test_data"
#endif

// Macros for test instantiation
#include "caffe/test_macros.hpp"

int main(int argc, char** argv);

namespace caffe {

template<typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiDeviceTest() {
    Caffe::set_mode(TypeParam::device);
  }
  // Caffe tests may create some temporary files, here we will do the cleanup.
  virtual ~MultiDeviceTest() { RemoveCaffeTempDir(); }
};

template<typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

template<typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
};

#ifndef CPU_ONLY
template<typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

template<typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
};
#endif  // CPU_ONLY


#if defined(USE_LEVELDB) && defined(USE_LMDB)
struct TypeLevelDB {
  static DataParameter_DB backend;
};

struct TypeLMDB {
  static DataParameter_DB backend;
};
#endif


template<typename Dtype>
bool isSupported(void);

#ifdef USE_HALF
template<> bool isSupported<half_fp>(void);
template<> bool isSupported<CPUDevice<half_fp> >(void);
template<> bool isSupported<GPUDevice<half_fp> >(void);
#endif  // USE_HALF

#ifdef USE_SINGLE
template<> bool isSupported<float>(void);
template<> bool isSupported<CPUDevice<float> >(void);
template<> bool isSupported<GPUDevice<float> >(void);
#endif  // USE_SINGLE

#ifdef USE_DOUBLE
template<> bool isSupported<double>(void);
template<> bool isSupported<CPUDevice<double> >(void);
template<> bool isSupported<GPUDevice<double> >(void);
#endif  // USE_DOUBLE

#ifdef USE_INT_QUANT_8
template<> bool isSupported<uint8_t>(void);
template<> bool isSupported<CPUDevice<uint8_t> >(void);
template<> bool isSupported<GPUDevice<uint8_t> >(void);
#endif  // USE_INT_QUANT_8

#ifdef USE_INT_QUANT_16
template<> bool isSupported<uint16_t>(void);
template<> bool isSupported<CPUDevice<uint16_t> >(void);
template<> bool isSupported<GPUDevice<uint16_t> >(void);
#endif  // USE_INT_QUANT_16

#ifdef USE_INT_QUANT_32
template<> bool isSupported<uint32_t>(void);
template<> bool isSupported<CPUDevice<uint32_t> >(void);
template<> bool isSupported<GPUDevice<uint32_t> >(void);
#endif  // USE_INT_QUANT_32

#ifdef USE_INT_QUANT_64
template<> bool isSupported<uint64_t>(void);
template<> bool isSupported<CPUDevice<uint64_t> >(void);
template<> bool isSupported<GPUDevice<uint64_t> >(void);
#endif  // USE_INT_QUANT_64

#if defined(USE_LEVELDB)
template<> bool isSupported<TypeLevelDB>(void);
#endif  // USE_LEVELDB

#if defined(USE_LMDB)
template<> bool isSupported<TypeLMDB>(void);
#endif  // USE_LMDB

#ifdef TYPED_TEST
#undef TYPED_TEST
#endif  // TYPED_TEST
#define TYPED_TEST(CaseName, TestName) \
  template<typename gtest_TypeParam_> \
  class GTEST_TEST_CLASS_NAME_(CaseName, TestName) \
      : public CaseName<gtest_TypeParam_> { \
    private: \
    typedef CaseName<gtest_TypeParam_> TestFixture; \
    typedef gtest_TypeParam_ TypeParam; \
    virtual void TestBody(); \
    virtual void TestBody_Impl();\
  }; \
  bool gtest_##CaseName##_##TestName##_registered_ GTEST_ATTRIBUTE_UNUSED_ = \
      ::testing::internal::TypeParameterizedTest< \
          CaseName, \
          ::testing::internal::TemplateSel< \
              GTEST_TEST_CLASS_NAME_(CaseName, TestName)>, \
          GTEST_TYPE_PARAMS_(CaseName)>::Register(\
              "", #CaseName, #TestName, 0); \
  template<typename gtest_TypeParam_> \
  void GTEST_TEST_CLASS_NAME_(CaseName, TestName)<gtest_TypeParam_>::TestBody()\
  {\
     if (isSupported<gtest_TypeParam_>())\
       TestBody_Impl();\
  }\
  template<typename gtest_TypeParam_> \
  void GTEST_TEST_CLASS_NAME_(CaseName, TestName) \
     <gtest_TypeParam_>::TestBody_Impl()

}  // namespace caffe
#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
