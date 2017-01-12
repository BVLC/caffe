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

#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NormalizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  NormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    // GaussianFiller<Dtype> filler(filler_param);
    filler_param.set_value(1);
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizeLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test norm
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    Dtype norm = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          norm += data * data;
        }
      }
    }
    const Dtype kErrorBound = 1e-5;
    // expect unit norm
    EXPECT_NEAR(1, sqrt(norm), kErrorBound);
  }
}

TYPED_TEST(NormalizeLayerTest, TestForwardScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(10);
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test norm
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    Dtype norm = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          norm += data * data;
        }
      }
    }
    const Dtype kErrorBound = 1e-5;
    // expect unit norm
    EXPECT_NEAR(10, sqrt(norm), kErrorBound);
  }
}

TYPED_TEST(NormalizeLayerTest, TestForwardScaleChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_channel_shared(false);
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(10);
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test norm
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    Dtype norm = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          norm += data * data;
        }
      }
    }
    const Dtype kErrorBound = 1e-5;
    // expect unit norm
    EXPECT_NEAR(10, sqrt(norm), kErrorBound);
  }
}

TYPED_TEST(NormalizeLayerTest, TestForwardEltWise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_across_spatial(false);
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test norm
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < height; ++k) {
      for (int l = 0; l < width; ++l) {
        Dtype norm = 0;
        for (int j = 0; j < channels; ++j) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          norm += data * data;
        }
        const Dtype kErrorBound = 1e-5;
        // expect unit norm
        EXPECT_NEAR(1, sqrt(norm), kErrorBound);
      }
    }
  }
}

TYPED_TEST(NormalizeLayerTest, TestForwardEltWiseScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_across_spatial(false);
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(10);
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test norm
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < height; ++k) {
      for (int l = 0; l < width; ++l) {
        Dtype norm = 0;
        for (int j = 0; j < channels; ++j) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          norm += data * data;
        }
        const Dtype kErrorBound = 1e-5;
        // expect unit norm
        EXPECT_NEAR(10, sqrt(norm), kErrorBound);
      }
    }
  }
}

TYPED_TEST(NormalizeLayerTest, TestForwardEltWiseScaleChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_across_spatial(false);
  norm_param->set_channel_shared(false);
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(10);
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test norm
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < height; ++k) {
      for (int l = 0; l < width; ++l) {
        Dtype norm = 0;
        for (int j = 0; j < channels; ++j) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          norm += data * data;
        }
        const Dtype kErrorBound = 1e-5;
        // expect unit norm
        EXPECT_NEAR(10, sqrt(norm), kErrorBound);
      }
    }
  }
}

TYPED_TEST(NormalizeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(NormalizeLayerTest, TestGradientScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(3);
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NormalizeLayerTest, TestGradientScaleChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_channel_shared(false);
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(3);
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NormalizeLayerTest, TestGradientEltWise) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_across_spatial(false);
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NormalizeLayerTest, TestGradientEltWiseScale) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_across_spatial(false);
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(3);
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 2e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NormalizeLayerTest, TestGradientEltWiseScaleChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeParameter* norm_param = layer_param.mutable_norm_param();
  norm_param->set_across_spatial(false);
  norm_param->set_channel_shared(false);
  norm_param->mutable_scale_filler()->set_type("constant");
  norm_param->mutable_scale_filler()->set_value(3);
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 2e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
