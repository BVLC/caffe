#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lstm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LSTMLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LSTMLayerTest() : num_output_(7) {
    blob_bottom_vec_.push_back(&blob_bottom_);
    blob_bottom_vec_.push_back(&blob_bottom_cont_);
    blob_top_vec_.push_back(&blob_top_);
    unit_blob_bottom_vec_.push_back(&unit_blob_bottom_c_prev_);
    unit_blob_bottom_vec_.push_back(&unit_blob_bottom_x_);
    unit_blob_bottom_vec_.push_back(&unit_blob_bottom_cont_);
    unit_blob_top_vec_.push_back(&unit_blob_top_c_);
    unit_blob_top_vec_.push_back(&unit_blob_top_h_);

    ReshapeBlobs(1, 3);

    layer_param_.mutable_recurrent_param()->set_num_output(num_output_);
    FillerParameter* weight_filler =
        layer_param_.mutable_recurrent_param()->mutable_weight_filler();
    weight_filler->set_type("gaussian");
    weight_filler->set_std(0.2);
    FillerParameter* bias_filler =
        layer_param_.mutable_recurrent_param()->mutable_bias_filler();
    bias_filler->set_type("gaussian");
    bias_filler->set_std(0.1);

    layer_param_.set_phase(TEST);
  }

  void ReshapeBlobs(int num_timesteps, int num_instances) {
    blob_bottom_.Reshape(num_timesteps, num_instances, 3, 2);
    blob_bottom_static_.Reshape(num_instances, 2, 3, 4);
    vector<int> shape(2);
    shape[0] = num_timesteps;
    shape[1] = num_instances;
    blob_bottom_cont_.Reshape(shape);
    shape.push_back(num_output_);

    shape[0] = 1; shape[1] = num_instances; shape[2] = 4 * num_output_;
    unit_blob_bottom_x_.Reshape(shape);
    shape[0] = 1; shape[1] = num_instances; shape[2] = num_output_;
    unit_blob_bottom_c_prev_.Reshape(shape);
    shape.resize(2);
    shape[0] = 1; shape[1] = num_instances;
    unit_blob_bottom_cont_.Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_);
    filler.Fill(&unit_blob_bottom_c_prev_);
    filler.Fill(&unit_blob_bottom_x_);
  }

  int num_output_;
  LayerParameter layer_param_;
  Blob<Dtype> blob_bottom_;
  Blob<Dtype> blob_bottom_cont_;
  Blob<Dtype> blob_bottom_static_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  Blob<Dtype> unit_blob_bottom_cont_;
  Blob<Dtype> unit_blob_bottom_c_prev_;
  Blob<Dtype> unit_blob_bottom_x_;
  Blob<Dtype> unit_blob_top_c_;
  Blob<Dtype> unit_blob_top_h_;
  vector<Blob<Dtype>*> unit_blob_bottom_vec_;
  vector<Blob<Dtype>*> unit_blob_top_vec_;
};

TYPED_TEST_CASE(LSTMLayerTest, TestDtypesAndDevices);

TYPED_TEST(LSTMLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->blob_bottom_.shape();
  expected_top_shape.resize(3);
  expected_top_shape[2] = this->num_output_;
  EXPECT_TRUE(this->blob_top_.shape() == expected_top_shape);
}

TYPED_TEST(LSTMLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  const int kNumTimesteps = 3;
  const int num = this->blob_bottom_.shape(1);
  this->ReshapeBlobs(kNumTimesteps, num);

  // Fill the cont blob with <0, 1, 1, ..., 1>,
  // indicating a sequence that begins at the first timestep
  // then continues for the rest of the sequence.
  for (int t = 0; t < kNumTimesteps; ++t) {
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_cont_.mutable_cpu_data()[t * num + n] = t > 0;
    }
  }

  // Process the full sequence in a single batch.
  FillerParameter filler_param;
  filler_param.set_mean(0);
  filler_param.set_std(1);
  GaussianFiller<Dtype> sequence_filler(filler_param);
  Caffe::set_random_seed(1);
  sequence_filler.Fill(&this->blob_bottom_);
  shared_ptr<LSTMLayer<Dtype> > layer(new LSTMLayer<Dtype>(this->layer_param_));
  Caffe::set_random_seed(1701);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Calling forward for full sequence LSTM";
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Copy the inputs and outputs to reuse/check them later.
  Blob<Dtype> bottom_copy(this->blob_bottom_.shape());
  bottom_copy.CopyFrom(this->blob_bottom_);
  Blob<Dtype> top_copy(this->blob_top_.shape());
  top_copy.CopyFrom(this->blob_top_);

  // Process the batch one timestep at a time;
  // check that we get the same result.
  this->ReshapeBlobs(1, num);
  layer.reset(new LSTMLayer<Dtype>(this->layer_param_));
  Caffe::set_random_seed(1701);
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const int bottom_count = this->blob_bottom_.count();
  const int top_count = this->blob_top_.count();
  const Dtype kEpsilon = 1e-5;
  for (int t = 0; t < kNumTimesteps; ++t) {
    caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
               this->blob_bottom_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_cont_.mutable_cpu_data()[n] = t > 0;
    }
    LOG(INFO) << "Calling forward for LSTM timestep " << t;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < top_count; ++i) {
      ASSERT_LT(t * top_count + i, top_copy.count());
      EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                  top_copy.cpu_data()[t * top_count + i], kEpsilon)
         << "t = " << t << "; i = " << i;
    }
  }

  // Process the batch one timestep at a time with all cont blobs set to 0.
  // Check that we get a different result, except in the first timestep.
  Caffe::set_random_seed(1701);
  layer.reset(new LSTMLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int t = 0; t < kNumTimesteps; ++t) {
    caffe_copy(bottom_count, bottom_copy.cpu_data() + t * bottom_count,
               this->blob_bottom_.mutable_cpu_data());
    for (int n = 0; n < num; ++n) {
      this->blob_bottom_cont_.mutable_cpu_data()[n] = 0;
    }
    LOG(INFO) << "Calling forward for LSTM timestep " << t;
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < top_count; ++i) {
      if (t == 0) {
        EXPECT_NEAR(this->blob_top_.cpu_data()[i],
                    top_copy.cpu_data()[t * top_count + i], kEpsilon)
           << "t = " << t << "; i = " << i;
      } else {
        EXPECT_NE(this->blob_top_.cpu_data()[i],
                  top_copy.cpu_data()[t * top_count + i])
           << "t = " << t << "; i = " << i;
      }
    }
  }
}

TYPED_TEST(LSTMLayerTest, TestLSTMUnitSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMUnitLayer<Dtype> layer(layer_param);
  layer.SetUp(this->unit_blob_bottom_vec_, this->unit_blob_top_vec_);
  const int num_axes = this->unit_blob_bottom_c_prev_.num_axes();
  ASSERT_EQ(num_axes, this->unit_blob_top_c_.num_axes());
  ASSERT_EQ(num_axes, this->unit_blob_top_h_.num_axes());
  for (int i = 0; i < num_axes; ++i) {
    EXPECT_EQ(this->unit_blob_bottom_c_prev_.shape(i),
              this->unit_blob_top_c_.shape(i));
    EXPECT_EQ(this->unit_blob_bottom_c_prev_.shape(i),
              this->unit_blob_top_h_.shape(i));
  }
}

TYPED_TEST(LSTMLayerTest, TestLSTMUnitGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMUnitLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  Dtype* cont_data = this->blob_bottom_cont_.mutable_cpu_data();
  cont_data[0] = 0;
  cont_data[1] = 0;
  cont_data[2] = 0;
  checker.CheckGradientExhaustive(&layer, this->unit_blob_bottom_vec_,
      this->unit_blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->unit_blob_bottom_vec_,
      this->unit_blob_top_vec_, 1);
}

TYPED_TEST(LSTMLayerTest, TestLSTMUnitGradientNonZeroCont) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMUnitLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  Dtype* cont_data = this->blob_bottom_cont_.mutable_cpu_data();
  cont_data[0] = 1;
  cont_data[1] = 0;
  cont_data[2] = 1;
  checker.CheckGradientExhaustive(&layer, this->unit_blob_bottom_vec_,
      this->unit_blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->unit_blob_bottom_vec_,
      this->unit_blob_top_vec_, 1);
}

TYPED_TEST(LSTMLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroCont) {
  typedef typename TypeParam::Dtype Dtype;
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_cont_.count(); ++i) {
    this->blob_bottom_cont_.mutable_cpu_data()[i] = i > 2;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroContBufferSize2) {
  typedef typename TypeParam::Dtype Dtype;
  this->ReshapeBlobs(2, 2);
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&this->blob_bottom_);
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_cont_.count(); ++i) {
    this->blob_bottom_cont_.mutable_cpu_data()[i] = i > 2;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(LSTMLayerTest, TestGradientNonZeroContBufferSize2WithStaticInput) {
  typedef typename TypeParam::Dtype Dtype;
  this->ReshapeBlobs(2, 2);
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&this->blob_bottom_);
  filler.Fill(&this->blob_bottom_static_);
  this->blob_bottom_vec_.push_back(&this->blob_bottom_static_);
  LSTMLayer<Dtype> layer(this->layer_param_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  for (int i = 0; i < this->blob_bottom_cont_.count(); ++i) {
    this->blob_bottom_cont_.mutable_cpu_data()[i] = i > 2;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
}


}  // namespace caffe
