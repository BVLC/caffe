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
class CCCPLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CCCPLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CCCPLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CCCPLayerTest, TestDtypesAndDevices);

TYPED_TEST(CCCPLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CCCPParameter* cccp_param =
      layer_param.mutable_cccp_param();
  cccp_param->set_num_output(4);
  shared_ptr<Layer<Dtype> > layer(
      new CCCPLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  // setting group should not change the shape
  cccp_param->set_num_output(3);
  cccp_param->set_group(3);
  layer.reset(new CCCPLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(CCCPLayerTest, TestSimpleCCCP) {
  // We will simply see if the cccp layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr<ConstantFiller<Dtype> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new ConstantFiller<Dtype>(filler_param));
  filler->Fill(this->blob_bottom_);
  LayerParameter layer_param;
  CCCPParameter* cccp_param =
      layer_param.mutable_cccp_param();
  cccp_param->set_num_output(4);
  cccp_param->mutable_weight_filler()->set_type("constant");
  cccp_param->mutable_weight_filler()->set_value(1);
  cccp_param->mutable_bias_filler()->set_type("constant");
  cccp_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new CCCPLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the cccp, the output should all have output values 3.1
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], 3.1, 1e-4);
  }
}

TYPED_TEST(CCCPLayerTest, TestSimpleCCCPGroup) {
  // We will simply see if the cccp layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          bottom_data[this->blob_bottom_->offset(n, c, h, w)] = c;
        }
      }
    }
  }
  LayerParameter layer_param;
  CCCPParameter* cccp_param =
      layer_param.mutable_cccp_param();
  cccp_param->set_num_output(3);
  cccp_param->set_group(3);
  cccp_param->mutable_weight_filler()->set_type("constant");
  cccp_param->mutable_weight_filler()->set_value(1);
  cccp_param->mutable_bias_filler()->set_type("constant");
  cccp_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new CCCPLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // After the cccp, the output should all have output values 9.1
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          Dtype data = top_data[this->blob_top_->offset(n, c, h, w)];
          EXPECT_NEAR(data, c * 1 + 0.1, 1e-4);
        }
      }
    }
  }
}

TYPED_TEST(CCCPLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CCCPParameter* cccp_param =
      layer_param.mutable_cccp_param();
  cccp_param->set_num_output(2);
  cccp_param->mutable_weight_filler()->set_type("gaussian");
  cccp_param->mutable_bias_filler()->set_type("gaussian");
  CCCPLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(CCCPLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CCCPParameter* cccp_param =
      layer_param.mutable_cccp_param();
  cccp_param->set_num_output(3);
  cccp_param->set_group(3);
  cccp_param->mutable_weight_filler()->set_type("gaussian");
  cccp_param->mutable_bias_filler()->set_type("gaussian");
  CCCPLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
