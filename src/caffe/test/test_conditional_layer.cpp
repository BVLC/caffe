#include <cstring>
#include <limits>
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
class ConditionalLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConditionalLayerTest()
      : blob_bottom_IF_(new Blob<Dtype>(4, 12, 2, 3)),
        blob_bottom_THEN_(new Blob<Dtype>(4, 14, 5, 7)),
        blob_bottom_LABELS_(new Blob<Dtype>(4, 1, 1, 1)),
        blob_top_LABELS_OR_INDICES_(new Blob<Dtype>()),
        blob_top_CONDITIONAL_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1890);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_IF_);
    // initialize bottom_IF with different MAX_VALUES
    Dtype max_value = std::numeric_limits<Dtype>::max();
    int num_elements = blob_bottom_IF_->count()/blob_bottom_IF_->num();
    Dtype* bottom_data_IF = blob_bottom_IF_->mutable_cpu_data();
    // fill first item with right index
    int offset_data = 0;
    int index_max = 20;
    *(bottom_data_IF + offset_data + index_max) = max_value;
    // fill second item with all max_values
    offset_data = num_elements;
    for (size_t n = 0; n < num_elements; n++)
      *(bottom_data_IF + offset_data + n) = max_value;
    // fill third item with wrong index
    offset_data = num_elements*2;
    index_max = 19;
    *(bottom_data_IF + offset_data + index_max) = max_value;
    // fill forth item with all zero values
    offset_data = num_elements*3;
    for (size_t n = 0; n < num_elements; n++)
      *(bottom_data_IF + offset_data + n) = 0;


    filler.Fill(blob_bottom_THEN_);
    filler.Fill(blob_bottom_LABELS_);
    blob_bottom_vec_.push_back(blob_bottom_IF_);
    blob_bottom_vec_.push_back(blob_bottom_THEN_);
    blob_bottom_vec_.push_back(blob_bottom_LABELS_);
    blob_top_vec_.push_back(blob_top_LABELS_OR_INDICES_);
    blob_top_vec_.push_back(blob_top_CONDITIONAL_);
  }
  virtual ~ConditionalLayerTest() {
    delete blob_bottom_IF_;
    delete blob_bottom_THEN_;
    delete blob_bottom_LABELS_;
    delete blob_top_LABELS_OR_INDICES_;
    delete blob_top_CONDITIONAL_;
  }
  int conditional_index_;
  int output_type_;

  Blob<Dtype>* const blob_bottom_IF_;
  Blob<Dtype>* const blob_bottom_THEN_;
  Blob<Dtype>* const blob_bottom_LABELS_;
  // blobs for the top of ConditionalLayer
  Blob<Dtype>* const blob_top_LABELS_OR_INDICES_;
  Blob<Dtype>* const blob_top_CONDITIONAL_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConditionalLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConditionalLayerTest, TestReshapeLabels) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ConditionalParameter* c_param = layer_param.mutable_conditional_param();
  c_param->set_conditional_index(20);
  c_param->set_output_type(ConditionalParameter_OUTPUT_TYPE_FILTERED_LABELS);
  ConditionalLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  //  In the test the items with max_value == conditional_index are 3
  //    (0, 1 and 3)
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->num(), 3);
  EXPECT_EQ(this->blob_top_CONDITIONAL_->num(), 3);
  EXPECT_LT(this->blob_top_LABELS_OR_INDICES_->num(),
      this->blob_bottom_LABELS_->num());
  EXPECT_LT(this->blob_top_CONDITIONAL_->num(),
      this->blob_bottom_THEN_->num());
  EXPECT_EQ(this->blob_bottom_LABELS_->channels(),
      this->blob_top_LABELS_OR_INDICES_->channels());
  EXPECT_EQ(this->blob_bottom_LABELS_->width(),
      this->blob_top_LABELS_OR_INDICES_->width());
  EXPECT_EQ(this->blob_bottom_LABELS_->height(),
      this->blob_top_LABELS_OR_INDICES_->height());
  EXPECT_EQ(this->blob_bottom_THEN_->channels(),
      this->blob_top_CONDITIONAL_->channels());
  EXPECT_EQ(this->blob_bottom_THEN_->width(),
      this->blob_top_CONDITIONAL_->width());
  EXPECT_EQ(this->blob_bottom_THEN_->height(),
      this->blob_top_CONDITIONAL_->height());
}

TYPED_TEST(ConditionalLayerTest, TestReshapeIndices) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ConditionalParameter* c_param = layer_param.mutable_conditional_param();
  c_param->set_conditional_index(20);
  c_param->set_output_type(ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES);
  ConditionalLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  //  In the test the items with max_value == conditional_index are 3
  //    (0, 1 and 3)
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->num(), 3);
  EXPECT_EQ(this->blob_top_CONDITIONAL_->num(), 3);
  EXPECT_LT(this->blob_top_LABELS_OR_INDICES_->num(),
      this->blob_bottom_LABELS_->num());
  EXPECT_LT(this->blob_top_CONDITIONAL_->num(),
      this->blob_bottom_THEN_->num());
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->channels(), 1);
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->width(), 1);
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->height(), 1);
  EXPECT_EQ(this->blob_bottom_THEN_->channels(),
      this->blob_top_CONDITIONAL_->channels());
  EXPECT_EQ(this->blob_bottom_THEN_->width(),
      this->blob_top_CONDITIONAL_->width());
  EXPECT_EQ(this->blob_bottom_THEN_->height(),
      this->blob_top_CONDITIONAL_->height());
}

TYPED_TEST(ConditionalLayerTest, TestForwardLabels) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ConditionalParameter* c_param = layer_param.mutable_conditional_param();
  c_param->set_conditional_index(20);
  c_param->set_output_type(ConditionalParameter_OUTPUT_TYPE_FILTERED_LABELS);
  ConditionalLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->data_at(0, 0, 0, 0),
      this->blob_bottom_LABELS_->data_at(0, 0, 0, 0));
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->data_at(1, 0, 0, 0),
      this->blob_bottom_LABELS_->data_at(1, 0, 0, 0));
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->data_at(2, 0, 0, 0),
      this->blob_bottom_LABELS_->data_at(3, 0, 0, 0));

  int num_elements = this->blob_top_CONDITIONAL_->count() /
      this->blob_top_CONDITIONAL_->num();
  const Dtype* top_CONDITIONAL = this->blob_top_CONDITIONAL_->cpu_data();
  const Dtype* bottom_THEN = this->blob_bottom_THEN_->cpu_data();
  for (size_t n = 0; n < num_elements; n++)
    EXPECT_EQ(*(top_CONDITIONAL+n), *(bottom_THEN+n));

  top_CONDITIONAL += num_elements;
  bottom_THEN += num_elements;
  for (size_t n = 0; n < num_elements; n++)
    EXPECT_EQ(*(top_CONDITIONAL+n), *(bottom_THEN+n));

  top_CONDITIONAL += num_elements;
  bottom_THEN += num_elements*2;
  for (size_t n = 0; n < num_elements; n++)
    EXPECT_EQ(*(top_CONDITIONAL+n), *(bottom_THEN+n));
}

TYPED_TEST(ConditionalLayerTest, TestForwardIndices) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  ConditionalParameter* c_param = layer_param.mutable_conditional_param();
  c_param->set_conditional_index(20);
  c_param->set_output_type(ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES);
  ConditionalLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->data_at(0, 0, 0, 0), 0);
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->data_at(1, 0, 0, 0), 1);
  EXPECT_EQ(this->blob_top_LABELS_OR_INDICES_->data_at(2, 0, 0, 0), 3);

  int num_elements = this->blob_top_CONDITIONAL_->count() /
      this->blob_top_CONDITIONAL_->num();
  const Dtype* top_CONDITIONAL = this->blob_top_CONDITIONAL_->cpu_data();
  const Dtype* bottom_THEN = this->blob_bottom_THEN_->cpu_data();
  for (size_t n = 0; n < num_elements; n++)
    EXPECT_EQ(*(top_CONDITIONAL+n), *(bottom_THEN+n));

  top_CONDITIONAL += num_elements;
  bottom_THEN += num_elements;
  for (size_t n = 0; n < num_elements; n++)
    EXPECT_EQ(*(top_CONDITIONAL+n), *(bottom_THEN+n));

  top_CONDITIONAL += num_elements;
  bottom_THEN += num_elements*2;
  for (size_t n = 0; n < num_elements; n++)
    EXPECT_EQ(*(top_CONDITIONAL+n), *(bottom_THEN+n));
}

TYPED_TEST(ConditionalLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    ConditionalParameter* c_param = layer_param.mutable_conditional_param();
    c_param->set_conditional_index(20);
    c_param->set_output_type(ConditionalParameter_OUTPUT_TYPE_FILTERED_INDICES);
    ConditionalLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}



}  // namespace caffe
