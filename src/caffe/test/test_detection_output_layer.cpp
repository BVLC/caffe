#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/detection_output_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static const float eps = 1e-6;

template <typename TypeParam>
class DetectionOutputLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DetectionOutputLayerTest()
      : num_(2),
        num_priors_(4),
        num_classes_(2),
        share_location_(true),
        num_loc_classes_(share_location_ ? 1 : num_classes_),
        background_label_id_(0),
        nms_threshold_(0.1),
        top_k_(2),
        blob_bottom_loc_(
            new Blob<Dtype>(num_, num_priors_ * num_loc_classes_ * 4, 1, 1)),
        blob_bottom_conf_(
            new Blob<Dtype>(num_, num_priors_ * num_classes_, 1, 1)),
        blob_bottom_prior_(new Blob<Dtype>(num_, 2, num_priors_ * 4, 1)),
        blob_top_(new Blob<Dtype>()) {
    // Fill prior data first.
    Dtype*  prior_data = blob_bottom_prior_->mutable_cpu_data();
    const float step = 0.5;
    const float box_size = 0.3;
    int idx = 0;
    for (int h = 0; h < 2; ++h) {
      float center_y = (h + 0.5) * step;
      for (int w = 0; w < 2; ++w) {
        float center_x = (w + 0.5) * step;
        prior_data[idx++] = (center_x - box_size / 2);
        prior_data[idx++] = (center_y - box_size / 2);
        prior_data[idx++] = (center_x + box_size / 2);
        prior_data[idx++] = (center_y + box_size / 2);
      }
    }
    for (int i = 0; i < idx; ++i) {
      prior_data[idx + i] = 0.1;
    }

    // Fill confidences.
    Dtype* conf_data = blob_bottom_conf_->mutable_cpu_data();
    idx = 0;
    for (int i = 0; i < this->num_; ++i) {
      for (int j = 0; j < this->num_priors_; ++j) {
        for (int c = 0; c < this->num_classes_; ++c) {
          if (i % 2 == c % 2) {
            conf_data[idx++] = j * 0.2;
          } else {
            conf_data[idx++] = 1 - j * 0.2;
          }
        }
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_prior_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DetectionOutputLayerTest() {
    delete blob_bottom_loc_;
    delete blob_bottom_conf_;
    delete blob_bottom_prior_;
    delete blob_top_;
  }

  void FillLocData(const bool share_location = true) {
    // Fill location offsets.
    int num_loc_classes = share_location ? 1 : this->num_classes_;
    blob_bottom_loc_->Reshape(
        this->num_, this->num_priors_ * num_loc_classes * 4, 1, 1);
    Dtype* loc_data = blob_bottom_loc_->mutable_cpu_data();
    int idx = 0;
    for (int i = 0; i < this->num_; ++i) {
      for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
          for (int c = 0; c < num_loc_classes; ++c) {
            loc_data[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
            loc_data[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
            loc_data[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
            loc_data[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
          }
        }
      }
    }
  }

  void CheckEqual(const Blob<Dtype>& blob, const int num, const string values) {
    CHECK_LT(num, blob.height());

    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    EXPECT_EQ(items.size(), 7);

    // Check data.
    const Dtype* blob_data = blob.cpu_data();
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(static_cast<int>(blob_data[num * blob.width() + i]),
                atoi(items[i].c_str()));
    }
    for (int i = 2; i < 7; ++i) {
      EXPECT_NEAR(blob_data[num * blob.width() + i],
                  atof(items[i].c_str()), eps);
    }
  }

  int num_;
  int num_priors_;
  int num_classes_;
  bool share_location_;
  int num_loc_classes_;
  int background_label_id_;
  float nms_threshold_;
  int top_k_;

  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_prior_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DetectionOutputLayerTest, TestDtypesAndDevices);

TYPED_TEST(DetectionOutputLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  DetectionOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 7);
}

TYPED_TEST(DetectionOutputLayerTest, TestForwardShareLocation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  DetectionOutputLayer<Dtype> layer(layer_param);

  this->FillLocData(true);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 7);

  this->CheckEqual(*(this->blob_top_), 0, "0 1 1.0 0.15 0.15 0.45 0.45");
  this->CheckEqual(*(this->blob_top_), 1, "0 1 0.8 0.55 0.15 0.85 0.45");
  this->CheckEqual(*(this->blob_top_), 2, "0 1 0.6 0.15 0.55 0.45 0.85");
  this->CheckEqual(*(this->blob_top_), 3, "0 1 0.4 0.55 0.55 0.85 0.85");
  this->CheckEqual(*(this->blob_top_), 4, "1 1 0.6 0.45 0.45 0.75 0.75");
  this->CheckEqual(*(this->blob_top_), 5, "1 1 0.0 0.25 0.25 0.55 0.55");
}

TYPED_TEST(DetectionOutputLayerTest, TestForwardShareLocationTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  detection_output_param->mutable_nms_param()->set_top_k(this->top_k_);
  DetectionOutputLayer<Dtype> layer(layer_param);

  this->FillLocData(true);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 7);

  this->CheckEqual(*(this->blob_top_), 0, "0 1 1.0 0.15 0.15 0.45 0.45");
  this->CheckEqual(*(this->blob_top_), 1, "0 1 0.8 0.55 0.15 0.85 0.45");
  this->CheckEqual(*(this->blob_top_), 2, "1 1 0.6 0.45 0.45 0.75 0.75");
}

TYPED_TEST(DetectionOutputLayerTest, TestForwardNoShareLocation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(false);
  detection_output_param->set_background_label_id(-1);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  DetectionOutputLayer<Dtype> layer(layer_param);

  this->FillLocData(false);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 11);
  EXPECT_EQ(this->blob_top_->width(), 7);

  this->CheckEqual(*(this->blob_top_), 0, "0 0 0.6 0.55 0.55 0.85 0.85");
  this->CheckEqual(*(this->blob_top_), 1, "0 0 0.4 0.15 0.55 0.45 0.85");
  this->CheckEqual(*(this->blob_top_), 2, "0 0 0.2 0.55 0.15 0.85 0.45");
  this->CheckEqual(*(this->blob_top_), 3, "0 0 0.0 0.15 0.15 0.45 0.45");
  this->CheckEqual(*(this->blob_top_), 4, "0 1 1.0 0.20 0.20 0.50 0.50");
  this->CheckEqual(*(this->blob_top_), 5, "0 1 0.8 0.50 0.20 0.80 0.50");
  this->CheckEqual(*(this->blob_top_), 6, "0 1 0.6 0.20 0.50 0.50 0.80");
  this->CheckEqual(*(this->blob_top_), 7, "0 1 0.4 0.50 0.50 0.80 0.80");
  this->CheckEqual(*(this->blob_top_), 8, "1 0 1.0 0.25 0.25 0.55 0.55");
  this->CheckEqual(*(this->blob_top_), 9, "1 0 0.4 0.45 0.45 0.75 0.75");
  this->CheckEqual(*(this->blob_top_), 10, "1 1 0.6 0.40 0.40 0.70 0.70");
}

TYPED_TEST(DetectionOutputLayerTest, TestForwardNoShareLocationTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(false);
  detection_output_param->set_background_label_id(-1);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  detection_output_param->mutable_nms_param()->set_top_k(this->top_k_);
  DetectionOutputLayer<Dtype> layer(layer_param);

  this->FillLocData(false);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 7);

  this->CheckEqual(*(this->blob_top_), 0, "0 0 0.6 0.55 0.55 0.85 0.85");
  this->CheckEqual(*(this->blob_top_), 1, "0 0 0.4 0.15 0.55 0.45 0.85");
  this->CheckEqual(*(this->blob_top_), 2, "0 1 1.0 0.20 0.20 0.50 0.50");
  this->CheckEqual(*(this->blob_top_), 3, "0 1 0.8 0.50 0.20 0.80 0.50");
  this->CheckEqual(*(this->blob_top_), 4, "1 0 1.0 0.25 0.25 0.55 0.55");
  this->CheckEqual(*(this->blob_top_), 5, "1 1 0.6 0.40 0.40 0.70 0.70");
}

TYPED_TEST(DetectionOutputLayerTest, TestForwardNoShareLocationNeg0) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(false);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  DetectionOutputLayer<Dtype> layer(layer_param);

  this->FillLocData(false);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 7);

  this->CheckEqual(*(this->blob_top_), 0, "0 1 1.0 0.20 0.20 0.50 0.50");
  this->CheckEqual(*(this->blob_top_), 1, "0 1 0.8 0.50 0.20 0.80 0.50");
  this->CheckEqual(*(this->blob_top_), 2, "0 1 0.6 0.20 0.50 0.50 0.80");
  this->CheckEqual(*(this->blob_top_), 3, "0 1 0.4 0.50 0.50 0.80 0.80");
  this->CheckEqual(*(this->blob_top_), 4, "1 1 0.6 0.40 0.40 0.70 0.70");
}

TYPED_TEST(DetectionOutputLayerTest, TestForwardNoShareLocationNeg0TopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(false);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  detection_output_param->mutable_nms_param()->set_top_k(this->top_k_);
  DetectionOutputLayer<Dtype> layer(layer_param);

  this->FillLocData(false);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 7);

  this->CheckEqual(*(this->blob_top_), 0, "0 1 1.0 0.20 0.20 0.50 0.50");
  this->CheckEqual(*(this->blob_top_), 1, "0 1 0.8 0.50 0.20 0.80 0.50");
  this->CheckEqual(*(this->blob_top_), 2, "1 1 0.6 0.40 0.40 0.70 0.70");
}

}  // namespace caffe
