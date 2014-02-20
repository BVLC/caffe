// Copyright 2014 kloudkl@github

#include <cstring> // for memset
#include <cuda_runtime.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/regularizer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "gtest/gtest.h"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template<typename Dtype>
class RegularizationAsLossTest : public ::testing::Test {
 protected:
  RegularizationAsLossTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 3, 2)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
  }
  virtual ~RegularizationAsLossTest() {
    delete blob_bottom_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(RegularizationAsLossTest, Dtypes);

#define TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(device_mode, regularizer_type, coeff_type, coeff) \
TYPED_TEST(RegularizationAsLossTest, TestGradient##device_mode##_##regularizer_type##_##coeff_type){ \
  Caffe::set_mode(Caffe::device_mode); \
  LayerParameter layer_param; \
  RegularizerParameter* reg_param = layer_param.add_regularizer(); \
  reg_param->set_type(REG_TYPE(regularizer_type)); \
  reg_param->set_coeff(coeff); \
  if (coeff < 0) { \
    /* To suppress Google Test warning of death tests running in multiple threads */ \
    /* http://code.google.com/p/googletest/wiki/AdvancedGuide#Death_Test_Styles */ \
    ::testing::FLAGS_gtest_death_test_style = "threadsafe"; \
    ASSERT_DEATH(RegularizerAsLossLayer<TypeParam> layer(layer_param), \
                 "Regularizer coefficient must be greater than or equal to zero"); \
  } else { \
    RegularizerAsLossLayer<TypeParam> layer(layer_param); \
    layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_); \
    /* The second argment is the threshold. The current value is set large enough to */ \
    /*   ensure that all the following test cases instantiated with this macro pass. */ \
    /* Although not all of them need so large a threshold to pass, */ \
    /*   we have to let even the toughest ones to pass too. */ \
    GradientChecker<TypeParam> checker(1e-2, 5e-2, 1701); \
    for (int loop = 0; loop < 10; ++loop) { \
      checker.CheckGradientSingle(layer, this->blob_bottom_vec_, \
          this->blob_top_vec_, 0, -1, -1); \
    } \
  } \
}

TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, L1, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, L1, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, L1, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, L2, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, L2, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, L2, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, MAX_NORM, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, MAX_NORM, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(CPU, MAX_NORM, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, L1, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, L1, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, L1, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, L2, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, L2, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, L2, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, MAX_NORM, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, MAX_NORM, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_SINGLE_TYPE(GPU, MAX_NORM, ZERO, 0);


#define TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(device_mode, regularizer_type_a, \
    regularizer_type_b, coeff_type_a, coeff_type_b, coeff_a, coeff_b) \
TYPED_TEST(RegularizationAsLossTest, \
    TestGradient##device_mode##_##regularizer_type_a##_##regularizer_type_b##_##coeff_type_a##_##coeff_type_b){ \
  Caffe::set_mode(Caffe::device_mode); \
  LayerParameter layer_param; \
  RegularizerParameter* reg_param; \
  reg_param = layer_param.add_regularizer(); \
  reg_param->set_type(REG_TYPE(regularizer_type_a)); \
  reg_param->set_coeff(coeff_a); \
  reg_param = layer_param.add_regularizer(); \
  reg_param->set_type(REG_TYPE(regularizer_type_b)); \
  reg_param->set_coeff(coeff_b); \
  if (coeff_a < 0 || coeff_b < 0) { \
    /* To suppress Google Test warning of death tests running in multiple threads */ \
    /* http://code.google.com/p/googletest/wiki/AdvancedGuide#Death_Test_Styles */ \
    ::testing::FLAGS_gtest_death_test_style = "threadsafe"; \
    ASSERT_DEATH(RegularizerAsLossLayer<TypeParam> layer(layer_param), \
                 "Regularizer coefficient must be greater than or equal to zero"); \
  } else { \
    RegularizerAsLossLayer<TypeParam> layer(layer_param); \
    layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_); \
    /* The second argment is the threshold. The current value is set large enough to */ \
    /*   ensure that all the following test cases instantiated with this macro pass. */ \
    /* Although not all of them need so large a threshold to pass, */ \
    /*   we have to let even the toughest ones to pass too. */ \
    GradientChecker<TypeParam> checker(1e-2, 5e-2, 1701); \
    for (int loop = 0; loop < 10; ++loop) { \
      checker.CheckGradientSingle(layer, this->blob_bottom_vec_, \
          this->blob_top_vec_, 0, -1, -1); \
    } \
  } \
}

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, NEGA, NEGA, -1, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, NEGA, POSI, -1, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, NEGA, ZERO, -1, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, POSI, NEGA, 1, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, POSI, POSI, 1, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, POSI, ZERO, 1, 0);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, ZERO, NEGA, 0, -1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, ZERO, POSI, 0, 1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2, ZERO, ZERO, 0, 0);


}
