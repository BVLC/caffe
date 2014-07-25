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
class ConcatLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConcatLayerTest()
      : blob_bottom_0(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_1(new Blob<Dtype>(2, 5, 6, 5)),
        blob_bottom_2(new Blob<Dtype>(5, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<ConstantFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_value(1.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0);
    filler_param.set_value(2.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_1);
    filler_param.set_value(3.);
    filler.reset(new ConstantFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_2);
    blob_bottom_vec_0.push_back(blob_bottom_0);
    blob_bottom_vec_0.push_back(blob_bottom_1);
    blob_bottom_vec_1.push_back(blob_bottom_0);
    blob_bottom_vec_1.push_back(blob_bottom_2);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConcatLayerTest() {
    delete blob_bottom_0; delete blob_bottom_1;
    delete blob_bottom_2; delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0;
  Blob<Dtype>* const blob_bottom_1;
  Blob<Dtype>* const blob_bottom_2;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0, blob_bottom_vec_1;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConcatLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConcatLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_concat_dim(0);
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(),
    this->blob_bottom_0->num() + this->blob_bottom_2->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0->width());
}

TYPED_TEST(ConcatLayerTest, TestSetupChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0->num());
  EXPECT_EQ(this->blob_top_->channels(),
    this->blob_bottom_0->channels()+this->blob_bottom_1->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0->width());
}


TYPED_TEST(ConcatLayerTest, TestNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_0, &(this->blob_top_vec_));
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_0->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c, h, w),
            this->blob_bottom_vec_0[0]->data_at(n, c, h, w));
        }
      }
    }
    for (int c = 0; c < this->blob_bottom_1->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          EXPECT_EQ(this->blob_top_->data_at(n, c+3, h, w),
            this->blob_bottom_vec_0[1]->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(ConcatLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, &(this->blob_bottom_vec_0),
    &(this->blob_top_vec_));
}

}  // namespace caffe
