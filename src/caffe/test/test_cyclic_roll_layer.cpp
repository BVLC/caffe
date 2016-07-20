#include <algorithm>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/cyclic_roll_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CyclicRollLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CyclicRollLayerTest()
    : blob_bottom_(new Blob<Dtype>(4, 1, 3, 3)),
    blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    // fill the values
    Dtype input[] =  {1, 2, 3,
                      4, 5, 6,
                      7, 8, 9,
                      11, 12, 13,
                      14, 15, 16,
                      17, 18, 19,
                      21, 22, 23,
                      24, 25, 26,
                      27, 28, 29,
                      31, 32, 33,
                      34, 35, 36,
                      37, 38, 39,
                      };
    // assign the values
    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
      this->blob_bottom_vec_[0]->mutable_cpu_data()[i] = input[i];
    }
  }
  virtual ~CyclicRollLayerTest() { delete blob_bottom_;
    delete blob_top_;}
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CyclicRollLayerTest, TestDtypesAndDevices);

TYPED_TEST(CyclicRollLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CyclicRollLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(0),
    this->blob_bottom_vec_[0]->shape(0));
  EXPECT_EQ(this->blob_top_vec_[0]->shape(1),
    this->blob_bottom_vec_[0]->shape(1)*4);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(2),
    this->blob_bottom_vec_[0]->shape(2));
  EXPECT_EQ(this->blob_top_vec_[0]->shape(3),
    this->blob_bottom_vec_[0]->shape(3));
}

TYPED_TEST(CyclicRollLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CyclicRollLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // fill the values
  Dtype expected_data[] =  { 1, 2, 3,
                            4, 5, 6,
                            7, 8, 9,
                            13, 16, 19,
                            12, 15, 18,
                            11, 14, 17,
                            29, 28, 27,
                            26, 25, 24,
                            23, 22, 21,
                            37, 34, 31,
                            38, 35, 32,
                            39, 36, 33,
                            11, 12, 13,
                            14, 15, 16,
                            17, 18, 19,
                            23, 26, 29,
                            22, 25, 28,
                            21, 24, 27,
                            39, 38, 37,
                            36, 35, 34,
                            33, 32, 31,
                            7, 4, 1,
                            8, 5, 2,
                            9, 6, 3,
                            21, 22, 23,
                            24, 25, 26,
                            27, 28, 29,
                            33, 36, 39,
                            32, 35, 38,
                            31, 34, 37,
                            9, 8, 7,
                            6, 5, 4,
                            3, 2, 1,
                            17, 14, 11,
                            18, 15, 12,
                            19, 16, 13,
                            31, 32, 33,
                            34, 35, 36,
                            37, 38, 39,
                            3, 6, 9,
                            2, 5, 8,
                            1, 4, 7,
                            19, 18, 17,
                            16, 15, 14,
                            13, 12, 11,
                            27, 24, 21,
                            28, 25, 22,
                            29, 26, 23};
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[i], expected_data[i]);
  }
}

TYPED_TEST(CyclicRollLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CyclicRollLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_);
}

}  // namespace caffe
