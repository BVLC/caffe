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
class IgnoreOverlayLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  IgnoreOverlayLayerTest()
      : blob_bottom_0(new Blob<Dtype>(2, 4, 3, 2)),
        blob_bottom_1(new Blob<Dtype>(2, 4, 3, 2)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<GaussianFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler.reset(new GaussianFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0);
    filler->Fill(this->blob_bottom_1);
    for (int i = 0; i < blob_bottom_1->count(); i += 5) {
      blob_bottom_0->mutable_cpu_data()[i] = 255;
    }
    blob_bottom_vec_.push_back(blob_bottom_0);
    blob_bottom_vec_.push_back(blob_bottom_1);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~IgnoreOverlayLayerTest() {
    delete blob_bottom_0; delete blob_bottom_1;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0;
  Blob<Dtype>* const blob_bottom_1;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(IgnoreOverlayLayerTest, TestDtypesAndDevices);

TYPED_TEST(IgnoreOverlayLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_ignore_overlay_param()->set_ignore_label(255);
  IgnoreOverlayLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0->width());
}

TYPED_TEST(IgnoreOverlayLayerTest, TestCPU) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  const int ignore_label = 255;
  layer_param.mutable_ignore_overlay_param()->set_ignore_label(ignore_label);
  IgnoreOverlayLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_0->count(); ++i) {
    const Dtype top_val = this->blob_top_->cpu_data()[i];
    if (this->blob_bottom_0->cpu_data()[i] == ignore_label) {
      EXPECT_EQ(top_val, this->blob_bottom_0->cpu_data()[i]);
    }
    else {
      EXPECT_EQ(top_val, this->blob_bottom_1->cpu_data()[i]);
    }
  }
}


}  // namespace caffe
