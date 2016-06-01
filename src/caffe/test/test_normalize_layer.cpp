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
