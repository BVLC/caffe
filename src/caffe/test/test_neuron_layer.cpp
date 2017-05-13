#include <algorithm>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/absval_layer.hpp"
#include "caffe/layers/bnll_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/elu_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/log_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/threshold_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NeuronLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NeuronLayerTest()
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
  virtual ~NeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestDropoutForward(const float dropout_ratio) {
    LayerParameter layer_param;
    // Fill in the given dropout_ratio, unless it's 0.5, in which case we don't
    // set it explicitly to test that 0.5 is the default.
    if (dropout_ratio != 0.5) {
      layer_param.mutable_dropout_param()->set_dropout_ratio(dropout_ratio);
    }
    DropoutLayer<Dtype> layer(layer_param);
    layer_param.set_phase(TRAIN);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    float scale = 1. / (1. - layer_param.dropout_param().dropout_ratio());
    const int count = this->blob_bottom_->count();
    // Initialize num_kept to count the number of inputs NOT dropped out.
    int num_kept = 0;
    for (int i = 0; i < count; ++i) {
      if (top_data[i] != 0) {
        ++num_kept;
        EXPECT_EQ(top_data[i], bottom_data[i] * scale);
      }
    }
    const Dtype std_error = sqrt(dropout_ratio * (1 - dropout_ratio) / count);
    // Fail if the number dropped was more than 1.96 * std_error away from the
    // expected number -- requires 95% confidence that the dropout layer is not
    // obeying the given dropout_ratio for test failure.
    const Dtype empirical_dropout_ratio = 1 - num_kept / Dtype(count);
    EXPECT_NEAR(empirical_dropout_ratio, dropout_ratio, 1.96 * std_error);
  }

  void TestExpForward(const float base, const float scale, const float shift) {
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base);
    layer_param.mutable_exp_param()->set_scale(scale);
    layer_param.mutable_exp_param()->set_shift(shift);
    ExpLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    const Dtype kDelta = 2e-4;
    const Dtype* bottom_data = blob_bottom_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      const Dtype bottom_val = bottom_data[i];
      const Dtype top_val = top_data[i];
      if (base == -1) {
        EXPECT_NEAR(top_val, exp(shift + scale * bottom_val), kDelta);
      } else {
        EXPECT_NEAR(top_val, pow(base, shift + scale * bottom_val), kDelta);
      }
    }
  }

  void TestExpGradient(const float base, const float scale, const float shift) {
    LayerParameter layer_param;
    layer_param.mutable_exp_param()->set_base(base);
    layer_param.mutable_exp_param()->set_scale(scale);
    layer_param.mutable_exp_param()->set_shift(shift);
    ExpLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientEltwise(&layer, blob_bottom_vec_, blob_top_vec_);
  }

  void TestPReLU(PReLULayer<Dtype> *layer) {
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype* slope_data = layer->blobs()[0]->cpu_data();
    int hw = this->blob_bottom_->height() * this->blob_bottom_->width();
    int channels = this->blob_bottom_->channels();
    bool channel_shared = layer->layer_param().prelu_param().channel_shared();
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      int c = channel_shared ? 0 : (i / hw) % channels;
      EXPECT_EQ(top_data[i],
          std::max(bottom_data[i], (Dtype)(0))
          + slope_data[c] * std::min(bottom_data[i], (Dtype)(0)));
    }
  }

  void LogBottomInit() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
    caffe_exp(this->blob_bottom_->count(), bottom_data, bottom_data);
  }

  void TestLogForward(const float base, const float scale, const float shift) {
    LogBottomInit();
    LayerParameter layer_param;
    layer_param.mutable_log_param()->set_base(base);
    layer_param.mutable_log_param()->set_scale(scale);
    layer_param.mutable_log_param()->set_shift(shift);
    LogLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    const Dtype kDelta = 2e-4;
    const Dtype* bottom_data = blob_bottom_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      const Dtype bottom_val = bottom_data[i];
      const Dtype top_val = top_data[i];
      if (base == -1) {
        EXPECT_NEAR(top_val, log(shift + scale * bottom_val), kDelta);
      } else {
        EXPECT_NEAR(top_val, log(shift + scale * bottom_val) / log(base),
                    kDelta);
      }
    }
  }

  void TestLogGradient(const float base, const float scale, const float shift) {
    LogBottomInit();
    LayerParameter layer_param;
    layer_param.mutable_log_param()->set_base(base);
    layer_param.mutable_log_param()->set_scale(scale);
    layer_param.mutable_log_param()->set_shift(shift);
    LogLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientEltwise(&layer, blob_bottom_vec_, blob_top_vec_);
  }
};

TYPED_TEST_CASE(NeuronLayerTest, TestDtypesAndDevices);

TYPED_TEST(NeuronLayerTest, TestAbsVal) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AbsValLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data    = this->blob_top_->cpu_data();
  const int count = this->blob_bottom_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(top_data[i], fabs(bottom_data[i]));
  }
}

TYPED_TEST(NeuronLayerTest, TestAbsGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AbsValLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestReLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}

TYPED_TEST(NeuronLayerTest, TestReLUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ReLULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestReLUWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  ReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
    } else {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestReLUGradientWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  ReLULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestELU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "elu_param { alpha: 0.5 }", &layer_param));
  ELULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype kDelta = 2e-4;
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (bottom_data[i] > 0) {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
    } else {
      EXPECT_NEAR(top_data[i], 0.5 * (exp(bottom_data[i]) - 1), kDelta);
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestELUasReLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "elu_param { alpha: 0 }", &layer_param));
  ELULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}

TYPED_TEST(NeuronLayerTest, TestELUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ELULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestELUasReLUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "elu_param { alpha: 0 }", &layer_param));
  ELULayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestSigmoid) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SigmoidLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_FLOAT_EQ(top_data[i], 1. / (1 + exp(-bottom_data[i])));
    // check that we squashed the value between 0 and 1
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
  }
}

TYPED_TEST(NeuronLayerTest, TestSigmoidGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SigmoidLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestTanH) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TanHLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int k = 0; k < this->blob_bottom_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom_->width(); ++l) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) - 1) /
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) + 1));
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) - 1) /
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) + 1));
        }
      }
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestTanHGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TanHLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestExpLayer) {
  typedef typename TypeParam::Dtype Dtype;
  // Test default base of "-1" -- should actually set base := e.
  const Dtype kBase = -1;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradient) {
  typedef typename TypeParam::Dtype Dtype;
  // Test default base of "-1" -- should actually set base := e.
  const Dtype kBase = -1;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerWithShift) {
  typedef typename TypeParam::Dtype Dtype;
  // Test default base of "-1" -- should actually set base := e,
  // with a non-zero shift
  const Dtype kBase = -1;
  const Dtype kScale = 1;
  const Dtype kShift = 1;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientWithShift) {
  typedef typename TypeParam::Dtype Dtype;
  // Test default base of "-1" -- should actually set base := e,
  // with a non-zero shift
  const Dtype kBase = -1;
  const Dtype kScale = 1;
  const Dtype kShift = 1;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2Shift1) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 1;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2Shift1) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 1;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 0;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 0;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpLayerBase2Shift1Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 1;
  this->TestExpForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestExpGradientBase2Shift1Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 1;
  this->TestExpGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayer) {
  typedef typename TypeParam::Dtype Dtype;
  // Test default base of "-1" -- should actually set base := e.
  const Dtype kBase = -1;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradient) {
  typedef typename TypeParam::Dtype Dtype;
  // Test default base of "-1" -- should actually set base := e.
  const Dtype kBase = -1;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 0;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2Shift1) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 1;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2Shift1) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 1;
  const Dtype kShift = 1;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 0;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 0;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogLayerBase2Shift1Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 1;
  this->TestLogForward(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestLogGradientBase2Shift1Scale3) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kBase = 2;
  const Dtype kScale = 3;
  const Dtype kShift = 1;
  this->TestLogGradient(kBase, kScale, kShift);
}

TYPED_TEST(NeuronLayerTest, TestDropoutHalf) {
  const float kDropoutRatio = 0.5;
  this->TestDropoutForward(kDropoutRatio);
}

TYPED_TEST(NeuronLayerTest, TestDropoutThreeQuarters) {
  const float kDropoutRatio = 0.75;
  this->TestDropoutForward(kDropoutRatio);
}

TYPED_TEST(NeuronLayerTest, TestDropoutTestPhase) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  DropoutLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] != 0) {
      EXPECT_EQ(top_data[i], bottom_data[i]);
    }
  }
}

TYPED_TEST(NeuronLayerTest, TestDropoutGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  DropoutLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestDropoutGradientTest) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  DropoutLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestBNLL) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BNLLLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_GE(top_data[i], bottom_data[i]);
  }
}

TYPED_TEST(NeuronLayerTest, TestBNLLGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BNLLLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestPReLUParam) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* slopes = layer.blobs()[0]->cpu_data();
  int count = layer.blobs()[0]->count();
  for (int i = 0; i < count; ++i, ++slopes) {
    EXPECT_EQ(*slopes, 0.25);
  }
}

TYPED_TEST(NeuronLayerTest, TestPReLUForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(layer.blobs()[0].get());
  this->TestPReLU(&layer);
}

TYPED_TEST(NeuronLayerTest, TestPReLUForwardChannelShared) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_channel_shared(true);
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->TestPReLU(&layer);
}

TYPED_TEST(NeuronLayerTest, TestPReLUGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(layer.blobs()[0].get());
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestPReLUGradientChannelShared) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_prelu_param()->set_channel_shared(true);
  PReLULayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestPReLUConsistencyReLU) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter prelu_layer_param;
  LayerParameter relu_layer_param;
  relu_layer_param.mutable_relu_param()->set_negative_slope(0.25);
  PReLULayer<Dtype> prelu(prelu_layer_param);
  ReLULayer<Dtype> relu(relu_layer_param);
  // Set up blobs
  vector<Blob<Dtype>*> blob_bottom_vec_2;
  vector<Blob<Dtype>*> blob_top_vec_2;
  shared_ptr<Blob<Dtype> > blob_bottom_2(new Blob<Dtype>());
  shared_ptr<Blob<Dtype> > blob_top_2(new Blob<Dtype>());
  blob_bottom_vec_2.push_back(blob_bottom_2.get());
  blob_top_vec_2.push_back(blob_top_2.get());
  blob_bottom_2->CopyFrom(*this->blob_bottom_, false, true);
  // SetUp layers
  prelu.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  relu.SetUp(blob_bottom_vec_2, blob_top_vec_2);
  // Check forward
  prelu.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  relu.Forward(this->blob_bottom_vec_, blob_top_vec_2);
  for (int s = 0; s < blob_top_2->count(); ++s) {
    EXPECT_EQ(this->blob_top_->cpu_data()[s], blob_top_2->cpu_data()[s]);
  }
  // Check backward
  shared_ptr<Blob<Dtype> > tmp_blob(new Blob<Dtype>());
  tmp_blob->ReshapeLike(*blob_top_2.get());
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(tmp_blob.get());
  caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
      this->blob_top_->mutable_cpu_diff());
  caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
      blob_top_2->mutable_cpu_diff());
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  prelu.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  relu.Backward(blob_top_vec_2, propagate_down, blob_bottom_vec_2);
  for (int s = 0; s < blob_bottom_2->count(); ++s) {
    EXPECT_EQ(this->blob_bottom_->cpu_diff()[s], blob_bottom_2->cpu_diff()[s]);
  }
}

TYPED_TEST(NeuronLayerTest, TestPReLUInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  // Set layer parameters
  LayerParameter ip_layer_param;
  LayerParameter prelu_layer_param;
  InnerProductParameter *ip_param =
      ip_layer_param.mutable_inner_product_param();
  ip_param->mutable_weight_filler()->set_type("gaussian");
  ip_param->set_num_output(3);
  InnerProductLayer<Dtype> ip(ip_layer_param);
  PReLULayer<Dtype> prelu(prelu_layer_param);
  InnerProductLayer<Dtype> ip2(ip_layer_param);
  PReLULayer<Dtype> prelu2(prelu_layer_param);
  // Set up blobs
  vector<Blob<Dtype>*> blob_bottom_vec_2;
  vector<Blob<Dtype>*> blob_middle_vec_2;
  vector<Blob<Dtype>*> blob_top_vec_2;
  shared_ptr<Blob<Dtype> > blob_bottom_2(new Blob<Dtype>());
  shared_ptr<Blob<Dtype> > blob_middle_2(new Blob<Dtype>());
  shared_ptr<Blob<Dtype> > blob_top_2(new Blob<Dtype>());
  blob_bottom_vec_2.push_back(blob_bottom_2.get());
  blob_middle_vec_2.push_back(blob_middle_2.get());
  blob_top_vec_2.push_back(blob_top_2.get());
  blob_bottom_2->CopyFrom(*this->blob_bottom_, false, true);
  // SetUp layers
  ip.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  prelu.SetUp(this->blob_top_vec_, this->blob_top_vec_);
  ip2.SetUp(blob_bottom_vec_2, blob_middle_vec_2);
  prelu2.SetUp(blob_middle_vec_2, blob_top_vec_2);
  caffe_copy(ip2.blobs()[0]->count(), ip.blobs()[0]->cpu_data(),
      ip2.blobs()[0]->mutable_cpu_data());
  // Forward in-place
  ip.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  prelu.Forward(this->blob_top_vec_, this->blob_top_vec_);
  // Forward non-in-place
  ip2.Forward(blob_bottom_vec_2, blob_middle_vec_2);
  prelu2.Forward(blob_middle_vec_2, blob_top_vec_2);
  // Check numbers
  for (int s = 0; s < blob_top_2->count(); ++s) {
    EXPECT_EQ(this->blob_top_->cpu_data()[s], blob_top_2->cpu_data()[s]);
  }
  // Fill top diff with random numbers
  shared_ptr<Blob<Dtype> > tmp_blob(new Blob<Dtype>());
  tmp_blob->ReshapeLike(*blob_top_2.get());
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(tmp_blob.get());
  caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
      this->blob_top_->mutable_cpu_diff());
  caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
      blob_top_2->mutable_cpu_diff());
  // Backward in-place
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  prelu.Backward(this->blob_top_vec_, propagate_down, this->blob_top_vec_);
  ip.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  // Backward non-in-place
  prelu2.Backward(blob_top_vec_2, propagate_down, blob_middle_vec_2);
  ip2.Backward(blob_middle_vec_2, propagate_down, blob_bottom_vec_2);
  // Check numbers
  for (int s = 0; s < blob_bottom_2->count(); ++s) {
    EXPECT_FLOAT_EQ(this->blob_bottom_->cpu_diff()[s],
        blob_bottom_2->cpu_diff()[s]);
  }
  for (int s = 0; s < ip.blobs()[0]->count(); ++s) {
    EXPECT_FLOAT_EQ(ip.blobs()[0]->cpu_diff()[s],
        ip2.blobs()[0]->cpu_diff()[s]);
  }
  for (int s = 0; s < ip.blobs()[1]->count(); ++s) {
    EXPECT_FLOAT_EQ(ip.blobs()[1]->cpu_diff()[s],
        ip2.blobs()[1]->cpu_diff()[s]);
  }
  for (int s = 0; s < prelu.blobs()[0]->count(); ++s) {
    EXPECT_FLOAT_EQ(prelu.blobs()[0]->cpu_diff()[s],
        prelu2.blobs()[0]->cpu_diff()[s]);
  }
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNNeuronLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNNeuronLayerTest()
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
  virtual ~CuDNNNeuronLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNNeuronLayerTest, TestDtypes);

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUCuDNN) {
  LayerParameter layer_param;
  CuDNNReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(top_data[i], 0.);
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == bottom_data[i]);
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUGradientCuDNN) {
  LayerParameter layer_param;
  CuDNNReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUWithNegativeSlopeCuDNN) {
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  CuDNNReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    if (top_data[i] >= 0) {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i]);
    } else {
      EXPECT_FLOAT_EQ(top_data[i], bottom_data[i] * 0.01);
    }
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUGradientWithNegativeSlopeCuDNN) {
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "relu_param { negative_slope: 0.01 }", &layer_param));
  CuDNNReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestSigmoidCuDNN) {
  LayerParameter layer_param;
  CuDNNSigmoidLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_FLOAT_EQ(top_data[i], 1. / (1 + exp(-bottom_data[i])));
    // check that we squashed the value between 0 and 1
    EXPECT_GE(top_data[i], 0.);
    EXPECT_LE(top_data[i], 1.);
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestSigmoidGradientCuDNN) {
  LayerParameter layer_param;
  CuDNNSigmoidLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestTanHCuDNN) {
  LayerParameter layer_param;
  CuDNNTanHLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int k = 0; k < this->blob_bottom_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom_->width(); ++l) {
          EXPECT_GE(this->blob_top_->data_at(i, j, k, l) + 1e-4,
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) - 1) /
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) + 1));
          EXPECT_LE(this->blob_top_->data_at(i, j, k, l) - 1e-4,
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) - 1) /
             (exp(2*this->blob_bottom_->data_at(i, j, k, l)) + 1));
        }
      }
    }
  }
}

TYPED_TEST(CuDNNNeuronLayerTest, TestTanHGradientCuDNN) {
  LayerParameter layer_param;
  CuDNNTanHLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif

}  // namespace caffe
