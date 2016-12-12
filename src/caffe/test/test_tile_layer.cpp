#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/tile_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TileLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TileLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
  }

  virtual ~TileLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TileLayerTest, TestDtypesAndDevices);

TYPED_TEST(TileLayerTest, TestTrivialSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 1;
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  for (int i = 0; i < this->blob_bottom_->num_axes(); ++i) {
    layer_param.mutable_tile_param()->set_axis(i);
    TileLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_->num_axes(), this->blob_bottom_->num_axes());
    for (int j = 0; j < this->blob_bottom_->num_axes(); ++j) {
      EXPECT_EQ(this->blob_top_->shape(j), this->blob_bottom_->shape(j));
    }
  }
}

TYPED_TEST(TileLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 3;
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  for (int i = 0; i < this->blob_bottom_->num_axes(); ++i) {
    layer_param.mutable_tile_param()->set_axis(i);
    TileLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_->num_axes(), this->blob_bottom_->num_axes());
    for (int j = 0; j < this->blob_bottom_->num_axes(); ++j) {
      const int top_dim =
          ((i == j) ? kNumTiles : 1) * this->blob_bottom_->shape(j);
      EXPECT_EQ(top_dim, this->blob_top_->shape(j));
    }
  }
}

TYPED_TEST(TileLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kTileAxis = 0;
  const int kNumTiles = 3;
  layer_param.mutable_tile_param()->set_axis(kTileAxis);
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  TileLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
       for (int h = 0; h < this->blob_top_->height(); ++h) {
         for (int w = 0; w < this->blob_top_->width(); ++w) {
           const int bottom_n = n % this->blob_bottom_->num();
           EXPECT_EQ(this->blob_bottom_->data_at(bottom_n, c, h, w),
                     this->blob_top_->data_at(n, c, h, w));
         }
       }
    }
  }
}

TYPED_TEST(TileLayerTest, TestForwardChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 3;
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  TileLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
       for (int h = 0; h < this->blob_top_->height(); ++h) {
         for (int w = 0; w < this->blob_top_->width(); ++w) {
           const int bottom_c = c % this->blob_bottom_->channels();
           EXPECT_EQ(this->blob_bottom_->data_at(n, bottom_c, h, w),
                     this->blob_top_->data_at(n, c, h, w));
         }
       }
    }
  }
}

TYPED_TEST(TileLayerTest, TestTrivialGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 1;
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  TileLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(TileLayerTest, TestGradientNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kTileAxis = 0;
  const int kNumTiles = 3;
  layer_param.mutable_tile_param()->set_axis(kTileAxis);
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  TileLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(TileLayerTest, TestGradientChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kTileAxis = 1;
  const int kNumTiles = 3;
  layer_param.mutable_tile_param()->set_axis(kTileAxis);
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  TileLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
