#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/channelwise_affine_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ChannelwiseAffineLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ChannelwiseAffineLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ChannelwiseAffineLayerTest() {
      delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestChannelwiseAffine(ChannelwiseAffineLayer<Dtype> *layer) {
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype* slope_data = layer->blobs()[0]->cpu_data();
    const Dtype* bias_data = layer->blobs()[1]->cpu_data();
    const Dtype kDelta = 2e-5;
    int hw = this->blob_bottom_->height() * this->blob_bottom_->width();
    int channels = this->blob_bottom_->channels();
    bool channel_shared =
        layer->layer_param().channelwise_affine_param().channel_shared();
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
          int c = channel_shared ? 0 : (i / hw) % channels;
          EXPECT_NEAR(top_data[i],
                       bottom_data[i]* slope_data[c] + bias_data[c], kDelta);
        }
  }
};
TYPED_TEST_CASE(ChannelwiseAffineLayerTest, TestDtypesAndDevices);


TYPED_TEST(ChannelwiseAffineLayerTest, TestChannelwiseAffineForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ChannelwiseAffineLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(layer.blobs()[0].get());
    filler.Fill(layer.blobs()[1].get());
    this->TestChannelwiseAffine(&layer);
}

TYPED_TEST(ChannelwiseAffineLayerTest,
           TestChannelwiseAffineForwardChannelShared) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_channelwise_affine_param()->set_channel_shared(true);
    ChannelwiseAffineLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    this->TestChannelwiseAffine(&layer);
}

TYPED_TEST(ChannelwiseAffineLayerTest, TestChannelwiseAffineGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_channelwise_affine_param()->set_channel_shared(false);
    ChannelwiseAffineLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
}

TYPED_TEST(ChannelwiseAffineLayerTest,
           TestChannelwiseAffineGradientChannelShared) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_channelwise_affine_param()->set_channel_shared(true);
    ChannelwiseAffineLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
}

}  // namespace caffe
