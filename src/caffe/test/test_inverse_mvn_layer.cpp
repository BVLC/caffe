#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace {
  const int INPUT_NUM = 2;
  const int INPUT_CHANNELS = 3;
  const int INPUT_HEIGHT = 4;
  const int INPUT_WIDTH = 5;
}

namespace caffe {

template <typename TypeParam>
class InverseMVNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  void AddMvnTopBlob(Blob<Dtype>* blob, const std::string& name) {
    mvn_blob_top_vec_.push_back(blob);
    blob_finder_.AddBlob(name, blob);
  }

  static Blob<Dtype>* RandomBottomBlob() {
    // Give each channel a different mean and std deviation.
    std::vector<Dtype> means;
    means.push_back(0.5);
    means.push_back(-1.0);
    means.push_back(2.0);
    std::vector<Dtype> std_devs;
    std_devs.push_back(5.0);
    std_devs.push_back(0.5);
    std_devs.push_back(2.0);

    Blob<Dtype>* blob = new Blob<Dtype>(INPUT_NUM, INPUT_CHANNELS,
                                         INPUT_HEIGHT, INPUT_WIDTH);
    Dtype* data = blob->mutable_cpu_data();
    for (int channel = 0; channel < 3; ++channel) {
      using boost::normal_distribution;
      using boost::variate_generator;
      normal_distribution<Dtype> random_distribution(means[channel],
                                                     std_devs[channel]);
      variate_generator<caffe::rng_t*, normal_distribution<Dtype> >
          generator(caffe_rng(), random_distribution);

      for (int n = 0; n < blob->num(); ++n) {
        for (int h = 0; h < blob->height(); ++h) {
          for (int w = 0; w < blob->width(); ++w) {
            *(blob->offset(n, channel, h, w) + data) = generator();
          }
        }
      }
    }
    return blob;
  }

  InverseMVNLayerTest()
      : mvn_bottom_blob_(RandomBottomBlob()),
        mvn_mean_blob_(new Blob<Dtype>()),
        mvn_variance_blob_(new Blob<Dtype>()),
        mvn_result_blob_(new Blob<Dtype>()),
        inverse_mvn_blob_top_(new Blob<Dtype>()) {
    mvn_bottom_blob_vec_.push_back(mvn_bottom_blob_);

    AddMvnTopBlob(this->mvn_mean_blob_, "mean_a");
    AddMvnTopBlob(this->mvn_variance_blob_, "variance_a");
    AddMvnTopBlob(this->mvn_result_blob_, "normalized");

    // The blob that contains the means computed by the mvn layer.
    inverse_mvn_bottom_blob_vec_.push_back(mvn_mean_blob_);
    // The blob that contains the scales computed by the mvn layer.
    inverse_mvn_bottom_blob_vec_.push_back(mvn_variance_blob_);
    // The blob that has the scaled, mean-subtracted output of the mvn layer.
    inverse_mvn_bottom_blob_vec_.push_back(mvn_result_blob_);
    // The inverse mvn layer's output blob.
    inverse_mvn_blob_top_vec_.push_back(inverse_mvn_blob_top_);
    blob_finder_.AddBlob("unnormalized", inverse_mvn_blob_top_);
  }
  virtual ~InverseMVNLayerTest() {
    delete mvn_mean_blob_;
    delete mvn_variance_blob_;
    delete mvn_result_blob_;
    delete inverse_mvn_blob_top_;
  }
  Blob<Dtype>* const mvn_bottom_blob_;
  Blob<Dtype>* const mvn_mean_blob_;
  Blob<Dtype>* const mvn_variance_blob_;
  Blob<Dtype>* const mvn_result_blob_;
  Blob<Dtype>* const inverse_mvn_blob_top_;

  vector<Blob<Dtype>*> mvn_bottom_blob_vec_;
  vector<Blob<Dtype>*> mvn_blob_top_vec_;
  vector<Blob<Dtype>*> inverse_mvn_bottom_blob_vec_;
  vector<Blob<Dtype>*> inverse_mvn_blob_top_vec_;

  BlobFinder<Dtype> blob_finder_;
};

TYPED_TEST_CASE(InverseMVNLayerTest, TestDtypesAndDevices);

TYPED_TEST(InverseMVNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   } "
      " top: \"normalized\" top: \"variance_a\" top: \"mean_a\" ",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   } "
      " bottom: \"normalized\" bottom: \"variance_a\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  shared_ptr<InverseMVNLayer<Dtype> >
      inverse_mvn_layer(new InverseMVNLayer<Dtype>( inverse_mvn_layer_param ));
  inverse_mvn_layer->SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                           this->blob_finder_);

  EXPECT_EQ(this->mvn_blob_top_vec_.size(), 3);
  EXPECT_EQ(this->inverse_mvn_blob_top_vec_.size(), 1);

  Blob<Dtype>* normalized_blob =
      this->blob_finder_.PointerFromName("normalized");
  EXPECT_EQ(normalized_blob->num(), INPUT_NUM);
  EXPECT_EQ(normalized_blob->height(), INPUT_HEIGHT);
  EXPECT_EQ(normalized_blob->width(), INPUT_WIDTH);
  EXPECT_EQ(normalized_blob->channels(), INPUT_CHANNELS);

  Blob<Dtype>* mean_blob = this->blob_finder_.PointerFromName("mean_a");
  EXPECT_EQ(mean_blob->num(), INPUT_NUM);
  EXPECT_EQ(mean_blob->height(), 1);
  EXPECT_EQ(mean_blob->width(), 1);
  EXPECT_EQ(mean_blob->channels(), INPUT_CHANNELS);

  Blob<Dtype>* variance_blob = this->blob_finder_.PointerFromName("variance_a");
  EXPECT_EQ(variance_blob->num(), INPUT_NUM);
  EXPECT_EQ(variance_blob->height(), 1);
  EXPECT_EQ(variance_blob->width(), 1);
  EXPECT_EQ(variance_blob->channels(), INPUT_CHANNELS);

  Blob<Dtype>* unnormalized_blob =
      this->blob_finder_.PointerFromName("unnormalized");
  EXPECT_EQ(unnormalized_blob->num(), INPUT_NUM);
  EXPECT_EQ(unnormalized_blob->height(), INPUT_HEIGHT);
  EXPECT_EQ(unnormalized_blob->width(), INPUT_WIDTH);
  EXPECT_EQ(unnormalized_blob->channels(), INPUT_CHANNELS);
}

TYPED_TEST(InverseMVNLayerTest, TestSetUp_AcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true across_channels: true  } "
      " top: \"normalized\" top: \"variance_a\" top: \"mean_a\" ",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true  across_channels: true } "
      " bottom: \"normalized\" bottom: \"variance_a\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  shared_ptr<InverseMVNLayer<Dtype> >
      inverse_mvn_layer(new InverseMVNLayer<Dtype>( inverse_mvn_layer_param ));
  inverse_mvn_layer->SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                           this->blob_finder_);

  EXPECT_EQ(this->mvn_blob_top_vec_.size(), 3);
  EXPECT_EQ(this->inverse_mvn_blob_top_vec_.size(), 1);

  Blob<Dtype>* normalized_blob =
      this->blob_finder_.PointerFromName("normalized");
  EXPECT_EQ(normalized_blob->num(), INPUT_NUM);
  EXPECT_EQ(normalized_blob->height(), INPUT_HEIGHT);
  EXPECT_EQ(normalized_blob->width(), INPUT_WIDTH);
  EXPECT_EQ(normalized_blob->channels(), INPUT_CHANNELS);

  Blob<Dtype>* mean_blob = this->blob_finder_.PointerFromName("mean_a");
  EXPECT_EQ(mean_blob->num(), INPUT_NUM);
  EXPECT_EQ(mean_blob->height(), 1);
  EXPECT_EQ(mean_blob->width(), 1);
  EXPECT_EQ(mean_blob->channels(), 1);

  Blob<Dtype>* variance_blob = this->blob_finder_.PointerFromName("variance_a");
  EXPECT_EQ(variance_blob->num(), INPUT_NUM);
  EXPECT_EQ(variance_blob->height(), 1);
  EXPECT_EQ(variance_blob->width(), 1);
  EXPECT_EQ(variance_blob->channels(), 1);

  Blob<Dtype>* unnormalized_blob =
      this->blob_finder_.PointerFromName("unnormalized");
  EXPECT_EQ(unnormalized_blob->num(), INPUT_NUM);
  EXPECT_EQ(unnormalized_blob->height(), INPUT_HEIGHT);
  EXPECT_EQ(unnormalized_blob->width(), INPUT_WIDTH);
  EXPECT_EQ(unnormalized_blob->channels(), INPUT_CHANNELS);
}

// We rely on MVNLayer working correctly (as it is independently tested).
// Then we test that the mvn layer composed with the inverse mvn layer
// yields the identity function (i.e. returns the data originally fed
// into the MVNLayer.
TYPED_TEST(InverseMVNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   } "
      " top: \"normalized\" top: \"variance_a\" top: \"mean_a\" ",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   } "
      " bottom: \"normalized\" bottom: \"variance_a\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype> inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                          this->blob_finder_);

  // Run the blob forward through the MVN layer.
  mvn_layer.Forward(this->mvn_bottom_blob_vec_,
                this->mvn_blob_top_vec_);

  // Run the output of the MVN layer forward through the Inverse MVN layer.
  inverse_mvn_layer.Forward(this->inverse_mvn_bottom_blob_vec_,
                this->inverse_mvn_blob_top_vec_);

  int num = this->mvn_bottom_blob_->num();
  int channels = this->mvn_bottom_blob_->channels();
  int height = this->mvn_bottom_blob_->height();
  int width = this->mvn_bottom_blob_->width();

  // Since across_channels==false, there should be a mean and a variance
  // for each channel of the input image.
  Blob<Dtype>* mean_blob = this->blob_finder_.PointerFromName("mean_a");
  EXPECT_EQ(mean_blob->num(), INPUT_NUM);
  EXPECT_EQ(mean_blob->height(), 1);
  EXPECT_EQ(mean_blob->width(), 1);
  EXPECT_EQ(mean_blob->channels(), INPUT_CHANNELS);

  Blob<Dtype>* variance_blob = this->blob_finder_.PointerFromName("variance_a");
  EXPECT_EQ(variance_blob->num(), INPUT_NUM);
  EXPECT_EQ(variance_blob->height(), 1);
  EXPECT_EQ(variance_blob->width(), 1);
  EXPECT_EQ(variance_blob->channels(), INPUT_CHANNELS);

  // Expect that the dimensions of the blob coming out of the InverseMVN layer
  // are the same as what went in to the MVN layer.
  EXPECT_EQ(num, this->inverse_mvn_blob_top_->num());
  EXPECT_EQ(channels, this->inverse_mvn_blob_top_->channels());
  EXPECT_EQ(height, this->inverse_mvn_blob_top_->height());
  EXPECT_EQ(width, this->inverse_mvn_blob_top_->width());

  EXPECT_EQ(num, INPUT_NUM);
  EXPECT_EQ(channels, INPUT_CHANNELS);
  EXPECT_EQ(height, INPUT_HEIGHT);
  EXPECT_EQ(width, INPUT_WIDTH);

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          // Since the InverseMVNLayer is the inverse operation of the
          // MVNLayer, expect its effect to exactly cancel out, giving us
          // a top blob that is equal to the bottom blob of the MVNLayer.
          Dtype expected_data = this->mvn_bottom_blob_->data_at(i, j, k, l);
          Dtype actual_data = this->inverse_mvn_blob_top_->data_at(i, j, k, l);
          const Dtype kErrorBound = 0.0001;
          EXPECT_NEAR(expected_data, actual_data, kErrorBound);
        }
      }
    }
  }
}

// We rely on MVNLayer working correctly (as it is independently tested).
// Then we test that the mvn layer composed with the inverse mvn layer
// yields the identity function (i.e. returns the data originally fed
// into the MVNLayer.
TYPED_TEST(InverseMVNLayerTest, TestForward_AcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true  across_channels: true   } "
      " top: \"normalized\" top: \"variance_a\" top: \"mean_a\" ",
          &mvn_layer_param));

  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   across_channels: true } "
      " bottom: \"normalized\" bottom: \"variance_a\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype> inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                          this->blob_finder_);

  // Run the blob forward through the MVN layer.
  mvn_layer.Forward(this->mvn_bottom_blob_vec_,
                this->mvn_blob_top_vec_);

  // Run the output of the MVN layer forward through the Inverse MVN layer.
  inverse_mvn_layer.Forward(this->inverse_mvn_bottom_blob_vec_,
                this->inverse_mvn_blob_top_vec_);

  int num = this->mvn_bottom_blob_->num();
  int channels = this->mvn_bottom_blob_->channels();
  int height = this->mvn_bottom_blob_->height();
  int width = this->mvn_bottom_blob_->width();

  // Since across_channels==true, there should be one mean and one variance
  // per input image.
  Blob<Dtype>* mean_blob = this->blob_finder_.PointerFromName("mean_a");
  EXPECT_EQ(mean_blob->num(), INPUT_NUM);
  EXPECT_EQ(mean_blob->height(), 1);
  EXPECT_EQ(mean_blob->width(), 1);
  EXPECT_EQ(mean_blob->channels(), 1);

  Blob<Dtype>* variance_blob = this->blob_finder_.PointerFromName("variance_a");
  EXPECT_EQ(variance_blob->num(), INPUT_NUM);
  EXPECT_EQ(variance_blob->height(), 1);
  EXPECT_EQ(variance_blob->width(), 1);
  EXPECT_EQ(variance_blob->channels(), 1);

  // Expect that the dimensions of the blob coming out of the InverseMVN layer
  // are the same as what went in to the MVN layer.
  EXPECT_EQ(num, this->inverse_mvn_blob_top_->num());
  EXPECT_EQ(channels, this->inverse_mvn_blob_top_->channels());
  EXPECT_EQ(height, this->inverse_mvn_blob_top_->height());
  EXPECT_EQ(width, this->inverse_mvn_blob_top_->width());

  EXPECT_EQ(num, INPUT_NUM);
  EXPECT_EQ(channels, INPUT_CHANNELS);
  EXPECT_EQ(height, INPUT_HEIGHT);
  EXPECT_EQ(width, INPUT_WIDTH);

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          // Since the InverseMVNLayer is the inverse operation of the
          // MVNLayer, expect its effect to exactly cancel out, giving us
          // a top blob that is equal to the bottom blob of the MVNLayer.
          Dtype expected_data = this->mvn_bottom_blob_->data_at(i, j, k, l);
          Dtype actual_data = this->inverse_mvn_blob_top_->data_at(i, j, k, l);
          const Dtype kErrorBound = 0.0001;
          ASSERT_NEAR(expected_data, actual_data, kErrorBound);
        }
      }
    }
  }
}

// We rely on MVNLayer working correctly (as it is independently tested).
// Then we test that the mvn layer composed with the inverse mvn layer
// yields the identity function (i.e. returns the data originally fed
// into the MVNLayer.
TYPED_TEST(InverseMVNLayerTest, TestForward_MeanOnly) {
  typedef typename TypeParam::Dtype Dtype;

  // The test setup code added the variance blob to the top vector. But
  // it shouldn't be there in this case. So erase it.
  this->mvn_blob_top_vec_.erase(std::remove(this->mvn_blob_top_vec_.begin(),
                                              this->mvn_blob_top_vec_.end(),
                                              this->mvn_variance_blob_),
                                 this->mvn_blob_top_vec_.end());

  // The setup code added the variance blob to the inverse_mvn layer's
  // bottom vector. But it shouldn't be there in this case. So erase it.
  this->inverse_mvn_bottom_blob_vec_.erase(std::remove(
                                   this->inverse_mvn_bottom_blob_vec_.begin(),
                                   this->inverse_mvn_bottom_blob_vec_.end(),
                                   this->mvn_variance_blob_),
                                 this->inverse_mvn_bottom_blob_vec_.end());

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" normalize_variance: false } "
      " top: \"normalized\" top: \"mean_a\" ",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" normalize_variance: false }"
      " bottom: \"normalized\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype> inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                          this->blob_finder_);

  // Run the blob forward through the MVN layer.
  mvn_layer.Forward(this->mvn_bottom_blob_vec_,
                this->mvn_blob_top_vec_);

  // Run the output of the MVN layer forward through the Inverse MVN layer.
  inverse_mvn_layer.Forward(this->inverse_mvn_bottom_blob_vec_,
                this->inverse_mvn_blob_top_vec_);

  int num = this->mvn_bottom_blob_->num();
  int channels = this->mvn_bottom_blob_->channels();
  int height = this->mvn_bottom_blob_->height();
  int width = this->mvn_bottom_blob_->width();

  // Expect that the dimensions of the blob coming out of the InverseMVN layer
  // are the same as what went in to the MVN layer.
  EXPECT_EQ(num, this->inverse_mvn_blob_top_->num());
  EXPECT_EQ(channels, this->inverse_mvn_blob_top_->channels());
  EXPECT_EQ(height, this->inverse_mvn_blob_top_->height());
  EXPECT_EQ(width, this->inverse_mvn_blob_top_->width());

  EXPECT_EQ(num, INPUT_NUM);
  EXPECT_EQ(channels, INPUT_CHANNELS);
  EXPECT_EQ(height, INPUT_HEIGHT);
  EXPECT_EQ(width, INPUT_WIDTH);

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          // Since the InverseMVNLayer is the inverse operation of the
          // MVNLayer, expect its effect to exactly cancel out, giving us
          // a top blob that is equal to the bottom blob of the MVNLayer.
          Dtype expected_data = this->mvn_bottom_blob_->data_at(i, j, k, l);
          Dtype actual_data = this->inverse_mvn_blob_top_->data_at(i, j, k, l);
          const Dtype kErrorBound = 0.0001;
          EXPECT_NEAR(expected_data, actual_data, kErrorBound);
        }
      }
    }
  }
}

TYPED_TEST(InverseMVNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true  across_channels: true   } "
      " top: \"normalized\" top: \"variance_a\" top: \"mean_a\" ",
          &mvn_layer_param));

  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   across_channels: true } "
      " bottom: \"normalized\" bottom: \"variance_a\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype> inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                          this->blob_finder_);

  // Run the blob forward through the MVN layer.
  mvn_layer.Forward(this->mvn_bottom_blob_vec_,
                this->mvn_blob_top_vec_);

  // Run the output of the MVN layer forward through the Inverse MVN layer.
  inverse_mvn_layer.Forward(this->inverse_mvn_bottom_blob_vec_,
                this->inverse_mvn_blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.SetBlobFinder(this->blob_finder_);

  int blob_index_to_check = -1;
  for (int index = 0; index < this->inverse_mvn_bottom_blob_vec_.size();
       ++index) {
    if (this->inverse_mvn_bottom_blob_vec_[index] ==
         this->blob_finder_.PointerFromName("normalized")) {
      blob_index_to_check = index;
    }
  }

  checker.CheckGradientExhaustive(&inverse_mvn_layer,
      this->inverse_mvn_bottom_blob_vec_,
      this->inverse_mvn_blob_top_vec_, blob_index_to_check);
}

TYPED_TEST(InverseMVNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true   } "
      " top: \"normalized\" top: \"variance_a\" top: \"mean_a\" ",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" variance_blob: \"variance_a\" "
      " normalize_variance: true  } "
      " bottom: \"normalized\" bottom: \"variance_a\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype> inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                          this->blob_finder_);

  EXPECT_EQ(this->mvn_blob_top_vec_.size(), 3);
  EXPECT_EQ(this->inverse_mvn_blob_top_vec_.size(), 1);

  mvn_layer.Forward(this->mvn_bottom_blob_vec_,
                this->mvn_blob_top_vec_);

  inverse_mvn_layer.Forward(this->inverse_mvn_bottom_blob_vec_,
                            this->inverse_mvn_blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.SetBlobFinder(this->blob_finder_);

  int blob_index_to_check = -1;
  for (int index = 0; index < this->inverse_mvn_bottom_blob_vec_.size();
       ++index) {
    if (this->inverse_mvn_bottom_blob_vec_[index] ==
         this->blob_finder_.PointerFromName("normalized")) {
      blob_index_to_check = index;
    }
  }

  checker.CheckGradientExhaustive(&inverse_mvn_layer,
      this->inverse_mvn_bottom_blob_vec_,
      this->inverse_mvn_blob_top_vec_, blob_index_to_check);
}

TYPED_TEST(InverseMVNLayerTest, TestGradient_MeanOnly) {
  typedef typename TypeParam::Dtype Dtype;

  // The test setup code added the variance blob to the top vector. But
  // it shouldn't be there in this case. So erase it.
  this->mvn_blob_top_vec_.erase(std::remove(this->mvn_blob_top_vec_.begin(),
                                              this->mvn_blob_top_vec_.end(),
                                              this->mvn_variance_blob_),
                                 this->mvn_blob_top_vec_.end());

  // The setup code added the variance blob to the inverse_mvn layer's
  // bottom vector. But it shouldn't be there in this case. So erase it.
  this->inverse_mvn_bottom_blob_vec_.erase(std::remove(
                                   this->inverse_mvn_bottom_blob_vec_.begin(),
                                   this->inverse_mvn_bottom_blob_vec_.end(),
                                   this->mvn_variance_blob_),
                                 this->inverse_mvn_bottom_blob_vec_.end());

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" normalize_variance: false } "
      " top: \"normalized\" top: \"mean_a\" ",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  mvn_layer.SetUp(this->mvn_bottom_blob_vec_, this->mvn_blob_top_vec_,
                  this->blob_finder_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" normalize_variance: false }"
      " bottom: \"normalized\" bottom: \"mean_a\" "
      " top: \"unnormalized\"", &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype> inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_bottom_blob_vec_,
                           this->inverse_mvn_blob_top_vec_,
                          this->blob_finder_);

  EXPECT_EQ(this->mvn_blob_top_vec_.size(), 2);
  EXPECT_EQ(this->inverse_mvn_blob_top_vec_.size(), 1);

  mvn_layer.Forward(this->mvn_bottom_blob_vec_,
                this->mvn_blob_top_vec_);

  inverse_mvn_layer.Forward(this->inverse_mvn_bottom_blob_vec_,
                            this->inverse_mvn_blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.SetBlobFinder(this->blob_finder_);

  int blob_index_to_check = -1;
  for (int index = 0; index < this->inverse_mvn_bottom_blob_vec_.size();
       ++index) {
    if (this->inverse_mvn_bottom_blob_vec_[index] ==
         this->blob_finder_.PointerFromName("normalized")) {
      blob_index_to_check = index;
    }
  }

  checker.CheckGradientExhaustive(&inverse_mvn_layer,
      this->inverse_mvn_bottom_blob_vec_,
      this->inverse_mvn_blob_top_vec_, blob_index_to_check);
}

}  // namespace caffe
