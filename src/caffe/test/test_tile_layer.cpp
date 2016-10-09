/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
