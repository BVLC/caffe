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

#define TEST_REGULARIZER_AS_LOSS_LAYER(regularizer_type, device_mode, coeff_type, coeff) \
TYPED_TEST(RegularizationAsLossTest, TestGradient##regularizer_type##_##device_mode##_##coeff_type){ \
  Caffe::set_mode(Caffe::device_mode); \
  LayerParameter layer_param; \
  layer_param.set_regularizer(REG_TYPE(regularizer_type)); \
  layer_param.set_regularizer_coeff(coeff); \
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

TEST_REGULARIZER_AS_LOSS_LAYER(L1, CPU, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER(L1, CPU, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER(L1, CPU, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER(L1, GPU, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER(L1, GPU, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER(L1, GPU, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER(L2, CPU, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER(L2, CPU, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER(L2, CPU, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER(L2, GPU, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER(L2, GPU, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER(L2, GPU, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER(MAX_NORM, CPU, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER(MAX_NORM, CPU, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER(MAX_NORM, CPU, ZERO, 0);

TEST_REGULARIZER_AS_LOSS_LAYER(MAX_NORM, GPU, NEGA, -1);
TEST_REGULARIZER_AS_LOSS_LAYER(MAX_NORM, GPU, POSI, 1);
TEST_REGULARIZER_AS_LOSS_LAYER(MAX_NORM, GPU, ZERO, 0);

}
