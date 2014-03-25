// Copyright 2014 kloudkl@github

#include <cuda_runtime.h>
#include <cstring>  // for memset
#include <vector>

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

  void TestSubroutine(const bool death_condition,
                      const LayerParameter& layer_param, const Dtype step_size,
                      const Dtype threshold, const unsigned int seed = 1701);

  Blob<Dtype>* const blob_bottom_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(RegularizationAsLossTest, Dtypes);

// The death test only abort the current function
// http://code.google.com/p/googletest/wiki/V1_6_AdvancedGuide
//        #Propagating_Fatal_Failures
// We want to test all the combinations of coefficients.
// If this subroutine is place in the test cases directly,
// the test cases cannot enumerate the combinations after the first failure.
template<typename Dtype>
void RegularizationAsLossTest<Dtype>::TestSubroutine(
    const bool death_condition, const LayerParameter& layer_param,
    const Dtype step_size, const Dtype threshold, const unsigned int seed) {
  if (death_condition) {
    ASSERT_DEATH(
        RegularizerAsLossLayer<Dtype> layer(layer_param),
        "Regularizer coefficient must be greater than or equal to zero");
  } else {
    RegularizerAsLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
    GradientChecker<Dtype> checker(step_size, threshold, seed);
    for (int loop = 0; loop < 10; ++loop) {
      checker.CheckGradientSingle(layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0, -1, -1);
    }
  }
}

// ::testing::FLAGS_gtest_death_test_style = "threadsafe";
// To suppress Google Test warning of death tests running in multiple threads
// http://code.google.com/p/googletest/wiki/AdvancedGuide#Death_Test_Styles
#define TEST_REG_LOSS_LAYER_SINGLE_TYPE(mode, regularizer) \
TYPED_TEST(RegularizationAsLossTest, TestGradient##mode##_##regularizer) { \
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; \
  Caffe::set_mode(Caffe::mode); \
  TypeParam coeff[] = {1, 0, -1}; \
  /* Restart from failure crash is too slow. Do not test negative coeff. */ \
  int num_ceoff = 2; \
  bool condition; \
  for (int i = 0; i < num_ceoff; ++i) { \
    LayerParameter layer_param; \
    RegularizerParameter* reg_param = layer_param.add_regularizer(); \
    reg_param->set_type(REG_TYPE(regularizer)); \
    reg_param->set_coeff(coeff[i]); \
    condition = coeff[i] < 0; \
    this->TestSubroutine(condition, layer_param, 1e-2, 5e-2, 1701); \
  } \
}

TEST_REG_LOSS_LAYER_SINGLE_TYPE(CPU, L1);
TEST_REG_LOSS_LAYER_SINGLE_TYPE(CPU, L2);
TEST_REG_LOSS_LAYER_SINGLE_TYPE(CPU, MAX_NORM);

TEST_REG_LOSS_LAYER_SINGLE_TYPE(GPU, L1);
TEST_REG_LOSS_LAYER_SINGLE_TYPE(GPU, L2);
TEST_REG_LOSS_LAYER_SINGLE_TYPE(GPU, MAX_NORM);

#define TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(mode, regularizer_type_a, \
    regularizer_type_b) \
TYPED_TEST(RegularizationAsLossTest, \
    TestGradient##mode##_##regularizer_type_a##_##regularizer_type_b) { \
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; \
  Caffe::set_mode(Caffe::mode); \
  TypeParam coeff[] = {1, 0, -1}; \
  /* Restart from failure crash is too slow. Do not test negative coeff. */ \
  int num_ceoff = 2; \
  bool condition; \
  for (int i = 0; i < num_ceoff; ++i) { \
    for (int j = 0; j < num_ceoff; ++j) { \
      LayerParameter layer_param; \
      RegularizerParameter* reg_param; \
      reg_param = layer_param.add_regularizer(); \
      reg_param->set_type(REG_TYPE(regularizer_type_a)); \
      reg_param->set_coeff(coeff[i]); \
      reg_param = layer_param.add_regularizer(); \
      reg_param->set_type(REG_TYPE(regularizer_type_b)); \
      reg_param->set_coeff(coeff[j]); \
      condition = coeff[i] < 0 || coeff[j] < 0; \
      this->TestSubroutine(condition, layer_param, 1e-2, 5e-2, 1701); \
    } \
  } \
}

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, L2);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L1, MAX_NORM);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L2, L1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L2, L2);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, L2, MAX_NORM);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, MAX_NORM, L1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, MAX_NORM, L2);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(CPU, MAX_NORM, MAX_NORM);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, L1, L1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, L1, L2);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, L1, MAX_NORM);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, L2, L1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, L2, L2);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, L2, MAX_NORM);

TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, MAX_NORM, L1);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, MAX_NORM, L2);
TEST_REGULARIZER_AS_LOSS_LAYER_TWO_TYPES(GPU, MAX_NORM, MAX_NORM);

}  // namespace caffe
