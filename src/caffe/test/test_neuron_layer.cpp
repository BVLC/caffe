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
    Caffe::set_phase(Caffe::TRAIN);
    DropoutLayer<Dtype> layer(layer_param);
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
    
    void TestTopKForward(const unsigned int k = 10) {
    LayerParameter layer_param;
    layer_param.mutable_topk_param()->set_k(k);
    Caffe::set_phase(Caffe::TRAIN);
    TopKLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const int num = this->blob_bottom_->num();
    const int single_count = this->blob_bottom_->count() / this->blob_bottom_->num();

    for (int n = 0; n < num; ++n) {

        std::vector<Dtype> values;
        values.reserve(single_count);
        for (int c=0; c < single_count; c++) {
         values.push_back(bottom_data[c]);
        }

        //Getting top k values in brute-force way
        std::vector<unsigned int> idxs;
        for (int i = 0; i < single_count; ++i) {
           Dtype max_el = -99999999999;
           unsigned int max_idx;
            for (int j=0; j < values.size(); ++j) {
              if (values[j] > max_el) {
                  max_el = values[j];
                  max_idx = j;
                }
              }
          idxs.push_back(max_idx);
          values[max_idx] = Dtype(-9999999999);
          }

        for (int i = 0; i < k; ++i) {
            EXPECT_EQ(top_data[idxs[i]], bottom_data[idxs[i]]);
          }

        for (int i = k; i < single_count; ++i) {
            EXPECT_EQ(top_data[idxs[i]], Dtype(0));
          }

        bottom_data += this->blob_bottom_->offset(1);
        top_data += this->blob_top_->offset(1);
      }
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
  layer_param.ParseFromString("relu_param{negative_slope:0.01}");
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

TYPED_TEST(NeuronLayerTest, TestReLUGradientWithNegativeSlope) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.ParseFromString("relu_param{negative_slope:0.01}");
  ReLULayer<Dtype> layer(layer_param);
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

TYPED_TEST(NeuronLayerTest, TestDropoutHalf) {
  const float kDropoutRatio = 0.5;
  this->TestDropoutForward(kDropoutRatio);
}

TYPED_TEST(NeuronLayerTest, TestDropoutThreeQuarters) {
  const float kDropoutRatio = 0.75;
  this->TestDropoutForward(kDropoutRatio);
}


TYPED_TEST(NeuronLayerTest, TestTopKTen) {
  const unsigned int k = 10;
  this->TestTopKForward(k);
}

TYPED_TEST(NeuronLayerTest, TestTopKFifty) {
  const unsigned int k = 50;
  this->TestTopKForward(k);
}

TYPED_TEST(NeuronLayerTest, TestDropoutTestPhase) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Caffe::set_phase(Caffe::TEST);
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
  Caffe::set_phase(Caffe::TRAIN);
  DropoutLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(NeuronLayerTest, TestDropoutGradientTest) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Caffe::set_phase(Caffe::TEST);
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

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNNeuronLayerTest : public ::testing::Test {
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
  Caffe::set_mode(Caffe::GPU);
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
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  CuDNNReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUWithNegativeSlopeCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.ParseFromString("relu_param{negative_slope:0.01}");
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

TYPED_TEST(CuDNNNeuronLayerTest, TestReLUGradientWithNegativeSlopeCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.ParseFromString("relu_param{negative_slope:0.01}");
  CuDNNReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestSigmoidCuDNN) {
  Caffe::set_mode(Caffe::GPU);
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
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  CuDNNSigmoidLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3, 1701, 0., 0.01);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNNeuronLayerTest, TestTanHCuDNN) {
  Caffe::set_mode(Caffe::GPU);
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
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  CuDNNTanHLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif

}  // namespace caffe
