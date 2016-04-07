#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "boost/lexical_cast.hpp"

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class RBMInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RBMInnerProductLayerTest()
      : blob_bottom_input_(new Blob<Dtype>(2, 3, 4, 5)),
        pre_activation_h1_(new Blob<Dtype>()),
        post_activation_h1_(new Blob<Dtype>()),
        sample_h1_(new Blob<Dtype>()),
        blob_top_error_1_(new Blob<Dtype>()),
        blob_top_error_2_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
  }

  virtual ~RBMInnerProductLayerTest() {
    delete blob_bottom_input_;
    delete pre_activation_h1_;
    delete post_activation_h1_;
    delete sample_h1_;
    delete blob_top_error_1_;
    delete blob_top_error_2_;
  }

  virtual void InitLayerFromProtoString(const string& proto) {
    LayerParameter layer_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
    layer_.reset(new RBMInnerProductLayer<Dtype>(layer_param));
    target_layer_.reset(new RBMInnerProductLayer<Dtype>(layer_param));
  }

  virtual void Fill() {
    // fill both the layer and the target_layer with (different) random values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    for (int i = 0; i < layer_->blobs().size(); ++i) {
      filler.Fill(layer_->blobs()[i].get());
    }
    for (int i = 0; i < target_layer_->blobs().size(); ++i) {
      filler.Fill(target_layer_->blobs()[i].get());
    }
    filler.Fill(blob_bottom_input_);
  }

  virtual string getLayerText(const string& extra_text = "",
                              bool forward_is_update = true,
                              bool visable_bias_term = true,
                              int num_output = 10,
                              int sample_steps_in_update = 2) {
    string proto =
      "name: 'rbm_inner_product_layer' "
      "type: 'RBMInnerProduct' "
      "rbm_inner_product_param { "
      "  connection_layer_param { "
      "    name: 'connection_inner_product' "
      "    type: 'InnerProduct' "
      "    inner_product_param { "
      "      num_output: ";
    proto += boost::lexical_cast<string>(num_output);
    proto +=
      "      bias_term: true "
      "      weight_filler: { "
      "        type: 'gaussian' "
      "        mean: 0.0 "
      "        std:  0.1 "
      "      } "
      "      bias_filler: { "
      "        type: 'gaussian' "
      "        mean: 0.0 "
      "        std:  0.1 "
      "      } "
      "    } "
      "  } "
      "  hidden_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  hidden_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } ";
    proto += "  sample_steps_in_update: ";
    proto += boost::lexical_cast<string>(sample_steps_in_update);
    if (forward_is_update) {
      proto += "  forward_is_update: true ";
    } else {
      proto += "  forward_is_update: false ";
    }
    if (visable_bias_term) {
      proto +=
        "  visible_bias_term: true "
        "  visible_bias_filler { "
        "    type: 'gaussian' "
        "    mean: 0.0 "
        "    std:  0.1 "
        "  } ";
    } else {
      proto += "  visible_bias_term: false ";
    }
    proto += extra_text;
    proto += "} ";
    return proto;
  }

  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const pre_activation_h1_;
  Blob<Dtype>* const post_activation_h1_;
  Blob<Dtype>* const sample_h1_;
  Blob<Dtype>* const blob_top_error_1_;
  Blob<Dtype>* const blob_top_error_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<RBMInnerProductLayer<Dtype> > layer_;
  shared_ptr<RBMInnerProductLayer<Dtype> > target_layer_;
};

TYPED_TEST_CASE(RBMInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(RBMInnerProductLayerTest, TestSetUpNoVisibleActivation) {
  string proto = this->getLayerText();
  this->InitLayerFromProtoString(proto);

  this->blob_top_vec_.clear();
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);

  // add something to the top
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->pre_activation_h1_->num(), 2);
  EXPECT_EQ(this->pre_activation_h1_->channels(), 10);
  EXPECT_EQ(this->pre_activation_h1_->height(), 1);
  EXPECT_EQ(this->pre_activation_h1_->width(), 1);

  // add more to the top
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->post_activation_h1_->num(), 2);
  EXPECT_EQ(this->post_activation_h1_->channels(), 10);
  EXPECT_EQ(this->post_activation_h1_->height(), 1);
  EXPECT_EQ(this->post_activation_h1_->width(), 1);

  // add even more to the top
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->sample_h1_->num(), 2);
  EXPECT_EQ(this->sample_h1_->channels(), 10);
  EXPECT_EQ(this->sample_h1_->height(), 1);
  EXPECT_EQ(this->sample_h1_->width(), 1);

  // test that error tops are resized correctly
  string extra_text =
      "  loss_measure: RECONSTRUCTION "
      "  loss_measure: FREE_ENERGY ";
  proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->blob_top_vec_.push_back(this->blob_top_error_2_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_error_1_->num(), 2);
  EXPECT_EQ(this->blob_top_error_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_1_->height(), 4);
  EXPECT_EQ(this->blob_top_error_1_->width(), 5);

  EXPECT_EQ(this->blob_top_error_2_->num(), 2);
  EXPECT_EQ(this->blob_top_error_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_error_2_->height(), 1);
  EXPECT_EQ(this->blob_top_error_2_->width(), 1);
}

TYPED_TEST(RBMInnerProductLayerTest, TestSetUpWithVisibleActivation) {
    string extra_text =
      "  visible_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  visible_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } ";

  string proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);

  this->blob_top_vec_.clear();
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);

  // add something to the top
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->pre_activation_h1_->num(), 2);
  EXPECT_EQ(this->pre_activation_h1_->channels(), 10);
  EXPECT_EQ(this->pre_activation_h1_->height(), 1);
  EXPECT_EQ(this->pre_activation_h1_->width(), 1);

  // add more to the top
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->post_activation_h1_->num(), 2);
  EXPECT_EQ(this->post_activation_h1_->channels(), 10);
  EXPECT_EQ(this->post_activation_h1_->height(), 1);
  EXPECT_EQ(this->post_activation_h1_->width(), 1);

  // add even more to the top
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->sample_h1_->num(), 2);
  EXPECT_EQ(this->sample_h1_->channels(), 10);
  EXPECT_EQ(this->sample_h1_->height(), 1);
  EXPECT_EQ(this->sample_h1_->width(), 1);

  // test that error tops are resized correctly
  extra_text +=
      "  loss_measure: RECONSTRUCTION "
      "  loss_measure: FREE_ENERGY ";
  proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->blob_top_vec_.push_back(this->blob_top_error_2_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_error_1_->num(), 2);
  EXPECT_EQ(this->blob_top_error_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_1_->height(), 4);
  EXPECT_EQ(this->blob_top_error_1_->width(), 5);

  EXPECT_EQ(this->blob_top_error_2_->num(), 2);
  EXPECT_EQ(this->blob_top_error_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_error_2_->height(), 1);
  EXPECT_EQ(this->blob_top_error_2_->width(), 1);
}

}  // namespace caffe
