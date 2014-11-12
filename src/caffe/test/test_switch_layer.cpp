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
class SwitchLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SwitchLayerTest()
      : blob_bottom_0(new Blob<Dtype>(6, 3, 2, 5)),
        blob_bottom_1(new Blob<Dtype>(6, 3, 2, 5)),
        blob_selector(new Blob<Dtype>(6, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0);
    filler.Fill(this->blob_bottom_1);
    // Initialize the selector with 0s and 1s
    Dtype* selector_data = blob_selector->mutable_cpu_data();
    for (int i = 0; i < blob_selector->count(); ++i) {
      selector_data[i] = i % 2;
    }
    blob_bottom_vec.push_back(blob_bottom_0);
    blob_bottom_vec.push_back(blob_bottom_1);
    blob_bottom_vec.push_back(blob_selector);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SwitchLayerTest() {
    delete blob_bottom_0; delete blob_bottom_1;
    delete blob_selector; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0;
  Blob<Dtype>* const blob_bottom_1;
  Blob<Dtype>* const blob_selector;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SwitchLayerTest, TestDtypesAndDevices);

TYPED_TEST(SwitchLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SwitchLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0->width());
}

TYPED_TEST(SwitchLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SwitchLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    int index = this->blob_selector->data_at(n, 0, 0, 0);
    EXPECT_TRUE(index == 0 || index == 1);
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
            this->blob_bottom_vec[index]->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(SwitchLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SwitchLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec,
    this->blob_top_vec_, 0);
}

}  // namespace caffe
