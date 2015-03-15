#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class XCovLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  XCovLossLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(5, 3, 1, 1)),
        blob_bottom_1_(new Blob<Dtype>(5, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    CountFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~XCovLossLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(XCovLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(XCovLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  XCovLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  for (int i = 0; i < this->blob_top_->count(); i++) {
    Dtype val = *(this->blob_top_->cpu_data() + i);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(val, 0.048000015318393707, kErrorMargin);
  }
}

TYPED_TEST(XCovLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  XCovLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
