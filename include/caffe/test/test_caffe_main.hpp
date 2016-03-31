// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "../device.hpp"
#include "caffe/common.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define TEST_DEVICE -1
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

typedef ::testing::Types<CPUDevice<float> >
                         TestFloatAndDevices;

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

typedef ::testing::Types<CPUDevice<float>,
                         GPUDevice<float> >
                         TestFloatAndDevices;

#endif

#if defined(USE_LEVELDB) && defined(USE_LMDB)
struct TypeLevelDB {
  static DataParameter_DB backend;
};

struct TypeLMDB {
  static DataParameter_DB backend;
};
#endif

#ifdef USE_GREENTEA

template <typename Dtype>
bool isSupported(void);

template <>
bool isSupported<double>(void);

template <>
bool isSupported<GPUDevice<double> >(void);

template <>
bool isSupported<float>(void);

template <>
bool isSupported<GPUDevice<float> >(void);

template <>
bool isSupported<CPUDevice<float> >(void);

template <>
bool isSupported<CPUDevice<double> >(void);

#if defined(USE_LEVELDB) && defined(USE_LMDB)
template <>
bool isSupported<TypeLevelDB>(void);

template <>
bool isSupported<TypeLMDB>(void);
#endif

#ifdef TYPED_TEST
#undef TYPED_TEST
#endif

# define TYPED_TEST(CaseName, TestName) \
  template <typename gtest_TypeParam_> \
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
  template <typename gtest_TypeParam_> \
  void GTEST_TEST_CLASS_NAME_(CaseName, TestName)<gtest_TypeParam_>::TestBody()\
  {\
     if (isSupported<gtest_TypeParam_>())\
       TestBody_Impl();\
  }\
  template <typename gtest_TypeParam_> \
  void GTEST_TEST_CLASS_NAME_(CaseName, TestName) \
     <gtest_TypeParam_>::TestBody_Impl()
#endif


}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
