#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/reduction_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReductionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReductionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ReductionLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward(ReductionParameter_ReductionOp op,
                   float coeff = 1, int axis = 0) {
    LayerParameter layer_param;
    ReductionParameter* reduction_param = layer_param.mutable_reduction_param();
    reduction_param->set_operation(op);
    if (coeff != 1.0) { reduction_param->set_coeff(coeff); }
    if (axis != 0) { reduction_param->set_axis(axis); }
    shared_ptr<ReductionLayer<Dtype> > layer(
        new ReductionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* in_data = this->blob_bottom_->cpu_data();
    const int num = this->blob_bottom_->count(0, axis);
    const int dim = this->blob_bottom_->count(axis);
    for (int n = 0; n < num; ++n) {
      Dtype expected_result = 0;
      for (int d = 0; d < dim; ++d) {
        switch (op) {
          case ReductionParameter_ReductionOp_SUM:
            expected_result += *in_data;
            break;
          case ReductionParameter_ReductionOp_MEAN:
            expected_result += *in_data / dim;
            break;
          case ReductionParameter_ReductionOp_ASUM:
            expected_result += fabs(*in_data);
            break;
          case ReductionParameter_ReductionOp_SUMSQ:
            expected_result += (*in_data) * (*in_data);
            break;
          default:
            LOG(FATAL) << "Unknown reduction op: "
                << ReductionParameter_ReductionOp_Name(op);
        }
        ++in_data;
      }
      expected_result *= coeff;
      const Dtype computed_result = this->blob_top_->cpu_data()[n];
      EXPECT_FLOAT_EQ(expected_result, computed_result)
          << "Incorrect result computed with op "
          << ReductionParameter_ReductionOp_Name(op) << ", coeff " << coeff;
    }
  }

  void TestGradient(ReductionParameter_ReductionOp op,
                    float coeff = 1, int axis = 0) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ReductionParameter* reduction_param = layer_param.mutable_reduction_param();
    reduction_param->set_operation(op);
    reduction_param->set_coeff(coeff);
    reduction_param->set_axis(axis);
    ReductionLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 2e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReductionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReductionLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<ReductionLayer<Dtype> > layer(
      new ReductionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 0);
}

TYPED_TEST(ReductionLayerTest, TestSetUpWithAxis1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reduction_param()->set_axis(1);
  shared_ptr<ReductionLayer<Dtype> > layer(
      new ReductionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 1);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
}

TYPED_TEST(ReductionLayerTest, TestSetUpWithAxis2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_reduction_param()->set_axis(2);
  shared_ptr<ReductionLayer<Dtype> > layer(
      new ReductionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 3);
}

TYPED_TEST(ReductionLayerTest, TestSum) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumCoeff) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumCoeffAxis1) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestForward(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestSumGradient) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumCoeffGradient) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumCoeffAxis1Gradient) {
  const ReductionParameter_ReductionOp kOp = ReductionParameter_ReductionOp_SUM;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestGradient(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestMean) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestMeanCoeff) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestMeanCoeffAxis1) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestForward(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestMeanGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestMeanCoeffGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestMeanCoeffGradientAxis1) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_MEAN;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestGradient(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestAbsSum) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumCoeff) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumCoeffAxis1) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestForward(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumCoeffGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestAbsSumCoeffAxis1Gradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_ASUM;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestGradient(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquares) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUMSQ;
  this->TestForward(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresCoeff) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUMSQ;
  const float kCoeff = 2.3;
  this->TestForward(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresCoeffAxis1) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUMSQ;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestForward(kOp, kCoeff, kAxis);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUMSQ;
  this->TestGradient(kOp);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresCoeffGradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUMSQ;
  const float kCoeff = 2.3;
  this->TestGradient(kOp, kCoeff);
}

TYPED_TEST(ReductionLayerTest, TestSumOfSquaresCoeffAxis1Gradient) {
  const ReductionParameter_ReductionOp kOp =
      ReductionParameter_ReductionOp_SUMSQ;
  const float kCoeff = 2.3;
  const int kAxis = 1;
  this->TestGradient(kOp, kCoeff, kAxis);
}

}  // namespace caffe
