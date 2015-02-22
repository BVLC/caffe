#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/blob_finder.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MVNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  void AddTopBlob(Blob<Dtype>* blob, const std::string& name) {
    blob_top_vec_.push_back(blob);
    blob_finder_.AddBlob(name, blob);
  }

  static Blob<Dtype>* RandomBottomBlob() {
    // Give each channel a different mean and std deviation.
    vector<Dtype> means;
    means.push_back(0.5);
    means.push_back(-1.0);
    means.push_back(2.0);
    vector<Dtype> std_devs;
    std_devs.push_back(5.0);
    std_devs.push_back(0.5);
    std_devs.push_back(2.0);

    Blob<Dtype>* blob = new Blob<Dtype>(2, 3, 4, 5);
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

  MVNLayerTest()
      : blob_bottom_(RandomBottomBlob()),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_);
    AddTopBlob(blob_top_, "top0");
  }
  virtual ~MVNLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  BlobFinder<Dtype> blob_finder_;
};

TYPED_TEST_CASE(MVNLayerTest, TestDtypesAndDevices);

TYPED_TEST(MVNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MVNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_, this->blob_finder_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
      sum /= height * width;
      var /= height * width;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  EXPECT_EQ(this->blob_top_, this->blob_finder_.PointerFromName("top0"));
}

// Test the case where the MVNParameter specifies that the mean and variance
// blobs are to appear in the layer's top blobs.
TYPED_TEST(MVNLayerTest, TestForward_MeanAndVarianceInTopBlobs) {
  typedef typename TypeParam::Dtype Dtype;

  this->AddTopBlob(new Blob<Dtype>(), "mean");
  this->AddTopBlob(new Blob<Dtype>(), "variance");

  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean\" variance_blob: \"variance\""
      " normalize_variance: true  across_channels: false  } "
      " top: \"normalized\" top: \"variance\" top: \"mean\" ", &layer_param));
  MVNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_, this->blob_finder_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  Blob<Dtype> expected_input_means(num, channels, 1, 1);
  Blob<Dtype> expected_input_variances(num, channels, 1, 1);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Dtype input_mean = 0.0;
      Dtype input_variance = 0.0;
      Dtype sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;

          Dtype input_data = this->blob_bottom_->data_at(i, j, k, l);
          input_mean += input_data;
          input_variance += input_data*input_data;
        }
      }
      sum /= height * width;
      var /= height * width;

      Dtype n = height*width;
      input_mean /= n;
      input_variance /= n;
      input_variance -= input_mean*input_mean;
      // actually standard deviation, not variance.
      input_variance = sqrt(input_variance);

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
      *(expected_input_means.mutable_cpu_data() +
          expected_input_means.offset(i, j, 0, 0)) = input_mean;
      *(expected_input_variances.mutable_cpu_data() +
          expected_input_variances.offset(i, j, 0, 0)) = input_variance;
    }
  }

  // The variances and means should match what we computed in the loop above.
  Blob<Dtype>* mean_blob = this->blob_finder_.PointerFromName("mean");
  Blob<Dtype>* variance_blob = this->blob_finder_.PointerFromName("variance");
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      const Dtype kErrorBound = 0.0001;
      EXPECT_NEAR(expected_input_means.data_at(i, j, 0, 0),
                   mean_blob->data_at(i, j, 0, 0),
                   kErrorBound);
      EXPECT_NEAR(expected_input_variances.data_at(i, j, 0, 0),
                   variance_blob->data_at(i, j, 0, 0),
                   kErrorBound);
    }
  }
}

// Test the case where the MVNParameter specifies that the mean
// blob is to appear in the layer's top blobs.
TYPED_TEST(MVNLayerTest, TestForward_MeanInTopBlobs) {
  typedef typename TypeParam::Dtype Dtype;

  this->AddTopBlob(new Blob<Dtype>(), "mean");

  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean\"  }"
      " top: \"normalized\" top: \"mean\" ", &layer_param));
  MVNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_, this->blob_finder_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  Blob<Dtype> expected_input_means(num, channels, 1, 1);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Dtype input_mean = 0.0;
      Dtype sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;

          Dtype input_data = this->blob_bottom_->data_at(i, j, k, l);
          input_mean += input_data;
        }
      }
      sum /= height * width;
      var /= height * width;

      Dtype n = height*width;
      input_mean /= n;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
      *(expected_input_means.mutable_cpu_data() +
          expected_input_means.offset(i, j, 0, 0)) = input_mean;
    }
  }

  // The means should match what we computed in the loop above.
  Blob<Dtype>* mean_blob = this->blob_finder_.PointerFromName("mean");
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      const Dtype kErrorBound = 0.0001;
      EXPECT_NEAR(expected_input_means.data_at(i, j, 0, 0),
                   mean_blob->data_at(i, j, 0, 0),
                   kErrorBound);
    }
  }
}

TYPED_TEST(MVNLayerTest, TestForwardMeanOnly) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.ParseFromString("mvn_param{normalize_variance: false}");
  MVNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_, this->blob_finder_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;
        }
      }
      sum /= height * width;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
    }
  }
}

TYPED_TEST(MVNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
          " mvn_param { across_channels: true } ", &layer_param) );
  MVNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_,
              this->blob_finder_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Test mean
  int num = this->blob_bottom_->num();
  int channels = this->blob_bottom_->channels();
  int height = this->blob_bottom_->height();
  int width = this->blob_bottom_->width();

  for (int i = 0; i < num; ++i) {
    Dtype sum = 0, var = 0;
    Dtype bsum = 0;
    Dtype bvar = 0;
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          Dtype data = this->blob_top_->data_at(i, j, k, l);
          sum += data;
          var += data * data;

          data = this->blob_bottom_->data_at(i, j, k, l);
          bsum += data;
          bvar += data * data;
        }
      }
    }
    sum /= height * width * channels;
    var /= height * width * channels;

    bsum /= height * width * channels;
    bvar /= height * width * channels;

    const Dtype kErrorBound = 0.001;
    // expect zero mean
    EXPECT_NEAR(0, sum, kErrorBound);
    // expect unit variance
    EXPECT_NEAR(1, var, kErrorBound);
  }
}

TYPED_TEST(MVNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MVNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.SetBlobFinder(this->blob_finder_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MVNLayerTest, TestGradientMeanOnly) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.ParseFromString("mvn_param{normalize_variance: false}");
  MVNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.SetBlobFinder(this->blob_finder_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(MVNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.ParseFromString("mvn_param{across_channels: true}");
  MVNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.SetBlobFinder(this->blob_finder_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
