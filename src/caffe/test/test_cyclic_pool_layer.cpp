#include <algorithm>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cyclic_pool_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CyclicPoolLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CyclicPoolLayerTest()
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
                      31, 32, 33,
                      24, 25, 26,
                      37, 38, 39,
                      21, 22, 23,
                      34, 35, 36,
                      27, 28, 29,
                      };
    // assign the values
    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
      this->blob_bottom_vec_[0]->mutable_cpu_data()[i] = input[i];
    }
  }
  virtual ~CyclicPoolLayerTest() { delete blob_bottom_; delete blob_top_;}
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CyclicPoolLayerTest, TestDtypesAndDevices);

TYPED_TEST(CyclicPoolLayerTest, TestSetup) {
  this->SetUp();
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CyclicPoolLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(0),
    this->blob_bottom_vec_[0]->shape(0)/4);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(1),
    this->blob_bottom_vec_[0]->shape(1));
  EXPECT_EQ(this->blob_top_vec_[0]->shape(2),
    this->blob_bottom_vec_[0]->shape(2));
  EXPECT_EQ(this->blob_top_vec_[0]->shape(3),
    this->blob_bottom_vec_[0]->shape(3));
}

TYPED_TEST(CyclicPoolLayerTest, TestAVEForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cyclic_pool_param()->
    set_pool(CyclicPoolParameter_PoolMethod_AVE);
  CyclicPoolLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype expected_data[] = { 16, 17, 18,
                            19, 20, 21,
                            22, 23, 24};
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_EQ(this->blob_top_vec_[0]->
      cpu_data()[i], expected_data[i]);
  }
}

TYPED_TEST(CyclicPoolLayerTest, TestMAXForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cyclic_pool_param()->
    set_pool(CyclicPoolParameter_PoolMethod_MAX);
  CyclicPoolLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype expected_data[] = {31, 32, 33,
                           34, 35, 36,
                           37, 38, 39};
  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_EQ(this->blob_top_vec_[0]->
      cpu_data()[i], expected_data[i]);
  }
}

TYPED_TEST(CyclicPoolLayerTest, TestAVEBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cyclic_pool_param()->
    set_pool(CyclicPoolParameter_PoolMethod_AVE);
  CyclicPoolLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = -1;
  }
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down,
    this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_->cpu_diff()[i], -1/4.);
  }
}

TYPED_TEST(CyclicPoolLayerTest, TestMAXBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cyclic_pool_param()->
    set_pool(CyclicPoolParameter_PoolMethod_MAX);
  CyclicPoolLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = -1;
  }
  vector<bool> propagate_down(1, true);
  layer.Backward(this->blob_top_vec_, propagate_down,
    this->blob_bottom_vec_);
  // expected output
  Dtype output[] = {0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                    -1, -1, -1,
                    0, 0, 0,
                    -1, -1, -1,
                    0, 0, 0,
                    -1, -1, -1,
                    0, 0, 0,
                    };
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_EQ(this->blob_bottom_->cpu_diff()[i], output[i]);
  }
}

TYPED_TEST(CyclicPoolLayerTest, TestMAXBackwardExhaustive) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cyclic_pool_param()->
    set_pool(CyclicPoolParameter_PoolMethod_MAX);
  CyclicPoolLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_);
}

TYPED_TEST(CyclicPoolLayerTest, TestAVEBackwardExhaustive) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cyclic_pool_param()->
    set_pool(CyclicPoolParameter_PoolMethod_AVE);
  CyclicPoolLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_);
}

}  // namespace caffe
