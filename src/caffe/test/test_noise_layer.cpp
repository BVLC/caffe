#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/neuron_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "google/protobuf/text_format.h"

namespace caffe {

template <typename TypeParam>
class NoiseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NoiseLayerTest()
    : blob_bottom_(new Blob<Dtype>(5, 40, 5, 6)),
      blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_inplace_top_vec_.push_back(blob_bottom_);
  }

  virtual ~NoiseLayerTest() { delete blob_bottom_; delete blob_top_; }

  // We can't use the GradientChecker since it checks by numerically computing
  // gradient using finite differences, and the forward prop noise added by
  // NoiseLayer should not have any effect the gradient backprop.
  // The behavior we want is to just pass the diffs through from top to bottom.
  void RunBackwardTest(const std::string& layer_param_string, bool in_place) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(layer_param_string,
                                                        &layer_param));
    NoiseLayer<Dtype> layer(layer_param);
    std::vector<bool> prop_down;
    prop_down.push_back(true);

    // Put some diffs in the top blob.
    Blob<Dtype> top_diffs;
    top_diffs.ReshapeLike(*this->blob_bottom_);
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      top_diffs.mutable_cpu_data()[i] = static_cast<Dtype>(i);
    }
    if (in_place) {
      this->blob_inplace_top_vec_[0]->CopyFrom(top_diffs, true, true);
      layer.Backward(this->blob_inplace_top_vec_, prop_down,
                     this->blob_bottom_vec_);
    } else {
      this->blob_top_vec_[0]->CopyFrom(top_diffs, true, true);
      layer.Backward(this->blob_top_vec_, prop_down,
                     this->blob_bottom_vec_);
    }

    const Dtype TOLERANCE = 0.001;
    // Bottom diffs should be the same as the diffs we created above.
    CHECK_GT(this->blob_bottom_->count(), 0);
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      EXPECT_NEAR(this->blob_bottom_->cpu_diff()[i], top_diffs.cpu_diff()[i],
                  TOLERANCE);
    }
  }

  void RunGaussianBackwardsTest(bool inplace) {
    typedef typename TypeParam::Dtype Dtype;
    const Dtype NOISE_MEAN = static_cast<Dtype>(-0.5);
    const Dtype NOISE_STD_DEV = static_cast<Dtype>(0.7);

    LayerParameter layer_param;
    std::stringstream ss;
    ss << "noise_param { filler_param { type: 'gaussian' mean: "
       << NOISE_MEAN << " std:" << NOISE_STD_DEV << "} }";
    bool in_place = false;
    this->RunBackwardTest(ss.str(), in_place);
  }

  void RunUniformBackwardsTest(bool inplace) {
    typedef typename TypeParam::Dtype Dtype;
    const float NOISE_MIN = static_cast<Dtype>(-0.5);
    const float NOISE_MAX = static_cast<Dtype>(0.2);
    LayerParameter layer_param;
    std::stringstream ss;
    ss << "noise_param { filler_param { type: 'uniform' min: " << NOISE_MIN <<
          " max:" << NOISE_MAX << "} }";
    bool in_place = false;
    this->RunBackwardTest(ss.str(), in_place);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_inplace_top_vec_;
};

TYPED_TEST_CASE(NoiseLayerTest, TestDtypesAndDevices);

TYPED_TEST(NoiseLayerTest, TestForwardGaussian) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype NOISE_MEAN = static_cast<Dtype>(-0.5);
  const Dtype NOISE_STD_DEV = static_cast<Dtype>(0.7);

  LayerParameter layer_param;
  std::stringstream ss;
  ss << "noise_param{ filler_param { type: 'gaussian' mean: " << NOISE_MEAN <<
        " std:" << NOISE_STD_DEV << "} }";
  CHECK(google::protobuf::TextFormat::ParseFromString(ss.str(), &layer_param));
  NoiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check top shape.
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());

  // Compute the mean and variance of the noise that was added to the bottom
  // blob.
  Dtype mean = static_cast<Dtype>(0.0);
  Dtype var = static_cast<Dtype>(0.0);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          int offset = this->blob_bottom_->offset(n, c, h, w);
          Dtype delta =  this->blob_top_->cpu_data()[offset] -
                          this->blob_bottom_->cpu_data()[offset];
          mean += delta;
          var += delta*delta;
        }
      }
    }
  }

  mean /= this->blob_bottom_->count();
  var /= this->blob_bottom_->count();
  var -= mean*mean;
  Dtype std_dev = std::sqrt(var);
  const Dtype kErrorBound = NOISE_STD_DEV/10.0;
  // Computed mean and variance should match what we specified as the noise
  // layer's parameter.
  EXPECT_NEAR(mean, NOISE_MEAN, kErrorBound);
  EXPECT_NEAR(std_dev, NOISE_STD_DEV, kErrorBound);
}

TYPED_TEST(NoiseLayerTest, TestForwardUniform) {
  typedef typename TypeParam::Dtype Dtype;
  const float NOISE_MIN = static_cast<Dtype>(-0.5);
  const float NOISE_MAX = static_cast<Dtype>(0.2);

  LayerParameter layer_param;
  std::stringstream ss;
  ss << "noise_param{ filler_param { type: 'uniform' min: "
     << NOISE_MIN << " max:" << NOISE_MAX << "} }";
  CHECK(google::protobuf::TextFormat::ParseFromString(ss.str(), &layer_param));
  NoiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check top shape.
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());

  int positive_noise_count = 0;
  int negative_noise_count = 0;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          int offset = this->blob_bottom_->offset(n, c, h, w);
          float noise_middle = (NOISE_MIN + NOISE_MAX) / 2.0f;

          // delta is the value that we added to the bottom blob
          // to produce the top blob.
          Dtype delta =  this->blob_top_->cpu_data()[offset] -
                          this->blob_bottom_->cpu_data()[offset];

          if (delta < noise_middle) {
            negative_noise_count++;
          } else if (delta > noise_middle) {
            positive_noise_count++;
          }
          // All of the noise values should be between NOISE_MIN and NOISE_MAX.
          EXPECT_GE(delta, NOISE_MIN);
          EXPECT_LE(delta, NOISE_MAX);
        }
      }
    }
  }

  float total_noise_count = negative_noise_count + positive_noise_count;
  float positive_noise_ratio = positive_noise_count / total_noise_count;
  float negative_noise_ratio = negative_noise_count / total_noise_count;

  // Sanity check that we didn't fall through the loops without testing
  // anything.
  EXPECT_GT(total_noise_count, 10.0f);
  // Approx half the top values should be greater than the bottom val,
  // and approx half less than the bottom val.
  EXPECT_NEAR(negative_noise_ratio, 0.5f, 0.1f);
  EXPECT_NEAR(positive_noise_ratio, 0.5f, 0.1f);
}

TYPED_TEST(NoiseLayerTest, TestForwardGaussianInplace) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype NOISE_MEAN = static_cast<Dtype>(-0.5);
  const Dtype NOISE_STD_DEV = static_cast<Dtype>(0.7);

  // We are noising inplace, so copy the original bottom blob before it gets
  // changed.
  Blob<Dtype> orig_bottom;
  orig_bottom.CopyFrom(*this->blob_bottom_, false, true);

  LayerParameter layer_param;
  std::stringstream ss;
  ss << "noise_param{ filler_param { type: 'gaussian' mean: " << NOISE_MEAN <<
        " std:" << NOISE_STD_DEV << "} }";
  CHECK(google::protobuf::TextFormat::ParseFromString(ss.str(), &layer_param));
  NoiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_inplace_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_inplace_top_vec_);

  // Compute the mean and variance of the noise that was added to the bottom
  // blob.
  Dtype mean = static_cast<Dtype>(0.0);
  Dtype var = static_cast<Dtype>(0.0);
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          int offset = this->blob_bottom_->offset(n, c, h, w);
          // inplace, so bottom blob is also the top blob.
          // delta is the value that we added to the bottom blob
          // to produce the top blob.
          Dtype delta =  this->blob_bottom_->cpu_data()[offset] -
                          orig_bottom.cpu_data()[offset];
          mean += delta;
          var += delta*delta;
        }
      }
    }
  }

  mean /= this->blob_bottom_->count();
  var /= this->blob_bottom_->count();
  var -= mean*mean;
  Dtype std_dev = std::sqrt(var);
  const Dtype kErrorBound = NOISE_STD_DEV/10.0;
  // Computed mean and variance should match what we specified as the noise
  // layer's parameter.
  EXPECT_NEAR(mean, NOISE_MEAN, kErrorBound);
  EXPECT_NEAR(std_dev, NOISE_STD_DEV, kErrorBound);
}


TYPED_TEST(NoiseLayerTest, TestForwardUniformInplace) {
  typedef typename TypeParam::Dtype Dtype;
  const float NOISE_MIN = static_cast<Dtype>(-0.5);
  const float NOISE_MAX = static_cast<Dtype>(0.2);

  // We are noising inplace, so copy the original bottom blob before it gets
  // changed.
  Blob<Dtype> orig_bottom;
  orig_bottom.CopyFrom(*this->blob_bottom_, false, true);

  LayerParameter layer_param;
  std::stringstream ss;
  ss << "noise_param{ filler_param { type: 'uniform' min: "
     << NOISE_MIN << " max:" << NOISE_MAX << "} }";
  CHECK(google::protobuf::TextFormat::ParseFromString(ss.str(), &layer_param));
  NoiseLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_inplace_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_inplace_top_vec_);

  int positive_noise_count = 0;
  int negative_noise_count = 0;
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    for (int c = 0; c < this->blob_bottom_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_->width(); ++w) {
          int offset = this->blob_bottom_->offset(n, c, h, w);
          float noise_middle = (NOISE_MIN + NOISE_MAX) / 2.0f;

          // inplace, so bottom blob is also the top blob.
          // delta is the value that we added to the bottom blob
          // to produce the top blob.
          Dtype delta =  this->blob_bottom_->cpu_data()[offset] -
                          orig_bottom.cpu_data()[offset];

          if (delta < noise_middle) {
            negative_noise_count++;
          } else if (delta > noise_middle) {
            positive_noise_count++;
          }
          // All of the noise values should be between NOISE_MIN and NOISE_MAX.
          EXPECT_GE(delta, NOISE_MIN);
          EXPECT_LE(delta, NOISE_MAX);
        }
      }
    }
  }

  float total_noise_count = negative_noise_count + positive_noise_count;
  float positive_noise_ratio = positive_noise_count / total_noise_count;
  float negative_noise_ratio = negative_noise_count / total_noise_count;

  // Sanity check that we didn't fall through the loops without testing
  // anything.
  EXPECT_GT(total_noise_count, 10.0f);
  // Approx half the top values should be greater than the bottom val,
  // and approx half less than the bottom val.
  EXPECT_NEAR(negative_noise_ratio, 0.5f, 0.1f);
  EXPECT_NEAR(positive_noise_ratio, 0.5f, 0.1f);
}

TYPED_TEST(NoiseLayerTest, TestGradientGaussian) {
  bool in_place = false;
  this->RunGaussianBackwardsTest(in_place);
}

TYPED_TEST(NoiseLayerTest, TestGradientUniform) {
  bool in_place = false;
  this->RunUniformBackwardsTest(in_place);
}

TYPED_TEST(NoiseLayerTest, TestGradientGaussianInPlace) {
  bool in_place = true;
  this->RunGaussianBackwardsTest(in_place);
}

TYPED_TEST(NoiseLayerTest, TestGradientUniformInPlace) {
  bool in_place = true;
  this->RunUniformBackwardsTest(in_place);
}


}  // namespace caffe
