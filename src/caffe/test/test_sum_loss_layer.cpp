#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SumLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SumLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SumLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    SumLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = -3.7;
    layer_param.add_loss_weight(kLossWeight);
    SumLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SumLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SumLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(SumLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = -3.7;
  layer_param.add_loss_weight(kLossWeight);
  SumLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_));
}

}  // namespace caffe
