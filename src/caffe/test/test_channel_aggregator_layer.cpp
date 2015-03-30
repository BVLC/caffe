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
class ChannelAggregatorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ChannelAggregatorLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 5, 3, 2)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<GaussianFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler.reset(new GaussianFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ChannelAggregatorLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ChannelAggregatorLayerTest, TestDtypesAndDevices);

TYPED_TEST(ChannelAggregatorLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_channel_aggregator_param()->set_label_map_file("src/caffe/test/label_map_file.txt");
  ChannelAggregatorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  //EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

  /*
TYPED_TEST(ChannelAggregatorLayerTest, TestCPU) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  const Dtype bg_bias = 1.5;
  const Dtype fg_bias = 3.0;
  layer_param.mutable_bias_channel_param()->set_bg_bias(bg_bias);
  layer_param.mutable_bias_channel_param()->set_fg_bias(fg_bias);
  ChannelAggregatorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_0->num(); ++n) {
    vector<Dtype> values(this->blob_bottom_0->channels(), 0);
    values[0] = bg_bias;
    for (int j = 0; j < this->blob_bottom_1->channels(); ++j) {
      const int label = *this->blob_bottom_1->cpu_data(n, j);
      CHECK(label > 0 && label < values.size());
      values[label] += fg_bias;
    }
    for (int c = 0; c < this->blob_bottom_0->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_0->height(); ++h) {
	for (int w = 0; w < this->blob_bottom_0->width(); ++w) {
	  EXPECT_NEAR(*this->blob_bottom_0->cpu_data(n, c, h, w) + values[c],
		      *this->blob_top_->cpu_data(n, c, h, w),
		      1e-5);
	}
      }
    }
  }
}
  */

TYPED_TEST(ChannelAggregatorLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_channel_aggregator_param()->set_label_map_file("src/caffe/test/label_map_file.txt");
  ChannelAggregatorLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
