#include <cfloat>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class AccuracyLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  AccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_per_class_(new Blob<Dtype>()),
        top_k_(3) {
    vector<int> shape(2);
    shape[0] = 100;
    shape[1] = 10;
    blob_bottom_data_->Reshape(shape);
    shape.resize(1);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_);
    blob_top_per_class_vec_.push_back(blob_top_per_class_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = (*prefetch_rng)() % 10;
    }
  }

  virtual ~AccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
    delete blob_top_per_class_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_per_class_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_per_class_vec_;
  int top_k_;
};

TYPED_TEST_CASE(AccuracyLayerTest, TestDtypes);

TYPED_TEST(AccuracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  layer_param.mutable_accuracy_param()->
    set_type(AccuracyParameter_AccuracyType_PRE);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestSetupTopK) {
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param =
      layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(5);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestSetupOutputPerClass) {
  LayerParameter layer_param;
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_per_class_->num(), 10);
  EXPECT_EQ(this->blob_top_per_class_->channels(), 1);
  EXPECT_EQ(this->blob_top_per_class_->height(), 1);
  EXPECT_EQ(this->blob_top_per_class_->width(), 1);
}

TYPED_TEST(AccuracyLayerTest, TestRECAndPREForwardCPU) {
  LayerParameter layer_param;
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestJACForwardCPU) {
  LayerParameter layer_param;
  layer_param.mutable_accuracy_param()->
    set_type(AccuracyParameter_AccuracyType_JAC);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int count = 0;
  vector<int> true_pos(10);
  vector<int> data_count(10);
  vector<int> label_count(10);
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++true_pos[max_id];
    }
    ++count;
    ++data_count[max_id];
    ++label_count[this->blob_bottom_label_->data_at(i, 0, 0, 0)];
  }
  TypeParam jaccard = 0;
  for (int i = 0; i < 10; i++) {
    TypeParam uni = label_count[i] + data_count[i] - true_pos[i];
    jaccard += uni > 0 ?
      static_cast<TypeParam>(label_count[i]) * true_pos[i] / uni : 0;
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0), jaccard / count, 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestRECAndPREForwardCPUWithSpatialAxes) {
  this->blob_bottom_data_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  LayerParameter layer_param;
  layer_param.mutable_accuracy_param()->set_axis(1);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  const int num_labels = this->blob_bottom_label_->count();
  int max_id;
  int num_correct_labels = 0;
  vector<int> label_offset(3);
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for (int h = 0; h < this->blob_bottom_data_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom_data_->width(); ++w) {
        max_value = -FLT_MAX;
        max_id = 0;
        for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
          const TypeParam pred_value =
              this->blob_bottom_data_->data_at(n, c, h, w);
          if (pred_value > max_value) {
            max_value = pred_value;
            max_id = c;
          }
        }
        label_offset[0] = n; label_offset[1] = h; label_offset[2] = w;
        const int correct_label =
            static_cast<int>(this->blob_bottom_label_->data_at(label_offset));
        if (max_id == correct_label) {
          ++num_correct_labels;
        }
      }
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / TypeParam(num_labels), 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestJACForwardCPUWithSpatialAxes) {
  this->blob_bottom_data_->Reshape(2, 10, 4, 5);
  vector<int> label_shape(3);
  label_shape[0] = 2; label_shape[1] = 4; label_shape[2] = 5;
  this->blob_bottom_label_->Reshape(label_shape);
  this->FillBottoms();
  LayerParameter layer_param;
  layer_param.mutable_accuracy_param()->
    set_type(AccuracyParameter_AccuracyType_JAC);
  layer_param.mutable_accuracy_param()->set_axis(1);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  const int num_labels = this->blob_bottom_label_->count();
  int max_id;
  vector<int> true_pos(this->blob_bottom_data_->channels());
  vector<int> data_count(this->blob_bottom_data_->channels());
  vector<int> label_count(this->blob_bottom_data_->channels());
  vector<int> label_offset(3);
  for (int n = 0; n < this->blob_bottom_data_->num(); ++n) {
    for (int h = 0; h < this->blob_bottom_data_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom_data_->width(); ++w) {
        max_value = -FLT_MAX;
        max_id = 0;
        for (int c = 0; c < this->blob_bottom_data_->channels(); ++c) {
          const TypeParam pred_value =
              this->blob_bottom_data_->data_at(n, c, h, w);
          if (pred_value > max_value) {
            max_value = pred_value;
            max_id = c;
          }
        }
        label_offset[0] = n; label_offset[1] = h; label_offset[2] = w;
        const int correct_label =
            static_cast<int>(this->blob_bottom_label_->data_at(label_offset));
        if (max_id == correct_label) {
          ++true_pos[max_id];
        }
        ++data_count[max_id];
        ++label_count[correct_label];
      }
    }
  }
  TypeParam jaccard = 0;
  for (int i = 0; i < 10; i++) {
    TypeParam uni = label_count[i] + data_count[i] - true_pos[i];
    jaccard += uni > 0 ?
      static_cast<TypeParam>(label_count[i]) * true_pos[i] / uni : 0;
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              jaccard / TypeParam(num_labels), 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestRECAndPREForwardCPUIgnoreLabel) {
  LayerParameter layer_param;
  const TypeParam kIgnoreLabelValue = -1;
  layer_param.mutable_accuracy_param()->set_ignore_label(kIgnoreLabelValue);
  AccuracyLayer<TypeParam> layer(layer_param);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[5] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[32] = kIgnoreLabelValue;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (kIgnoreLabelValue == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      continue;
    }
    ++count;
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
    }
  }
  EXPECT_EQ(count, 97);  // We set 3 out of 100 labels to kIgnoreLabelValue.
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / TypeParam(count), 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestRECAndPREForwardCPUTopK) {
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param = layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(this->top_k_);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  TypeParam current_value;
  int current_rank;
  int num_correct_labels = 0;
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 10; ++j) {
      current_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
      current_rank = 0;
      for (int k = 0; k < 10; ++k) {
        if (this->blob_bottom_data_->data_at(i, k, 0, 0) > current_value) {
          ++current_rank;
        }
      }
      if (current_rank < this->top_k_ &&
          j == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
        ++num_correct_labels;
      }
    }
  }

  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
}

TYPED_TEST(AccuracyLayerTest, TestRECForwardCPUPerClass) {
  LayerParameter layer_param;
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  const int num_class = this->blob_top_per_class_->num();
  vector<int> true_pos(num_class, 0);
  vector<int> label_count(num_class, 0);
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    ++label_count[this->blob_bottom_label_->data_at(i, 0, 0, 0)];
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
      ++true_pos[max_id];
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
  for (int i = 0; i < num_class; ++i) {
    TypeParam recall_per_class = (label_count[i] > 0 ?
       static_cast<TypeParam>(true_pos[i]) / label_count[i] : 0);
    EXPECT_NEAR(this->blob_top_per_class_->data_at(i, 0, 0, 0),
                recall_per_class, 1e-4);
  }
}

TYPED_TEST(AccuracyLayerTest, TestPREForwardCPUPerClass) {
  LayerParameter layer_param;
  layer_param.mutable_accuracy_param()->
    set_type(AccuracyParameter_AccuracyType_PRE);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  TypeParam max_value;
  int max_id;
  int num_correct_labels = 0;
  const int num_class = this->blob_top_per_class_->num();
  vector<int> true_pos(num_class, 0);
  vector<int> data_count(num_class, 0);
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    ++data_count[max_id];
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++num_correct_labels;
      ++true_pos[max_id];
    }
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              num_correct_labels / 100.0, 1e-4);
  for (int i = 0; i < num_class; ++i) {
    TypeParam precision_per_class = (data_count[i] > 0 ?
      static_cast<TypeParam>(true_pos[i]) / data_count[i] : 0);
    EXPECT_NEAR(this->blob_top_per_class_->data_at(i, 0, 0, 0),
                precision_per_class, 1e-4);
  }
}

TYPED_TEST(AccuracyLayerTest, TestJACForwardCPUPerClassWithIgnoreLabel) {
  LayerParameter layer_param;
  const TypeParam kIgnoreLabelValue = -1;
  layer_param.mutable_accuracy_param()->set_ignore_label(kIgnoreLabelValue);
  layer_param.mutable_accuracy_param()->
    set_type(AccuracyParameter_AccuracyType_JAC);
  AccuracyLayer<TypeParam> layer(layer_param);
  // Manually set some labels to the ignore label value (-1).
  this->blob_bottom_label_->mutable_cpu_data()[2] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[5] = kIgnoreLabelValue;
  this->blob_bottom_label_->mutable_cpu_data()[32] = kIgnoreLabelValue;
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_per_class_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_per_class_vec_);

  TypeParam max_value;
  int max_id;
  const int num_class = this->blob_top_per_class_->num();
  int count = 0;
  vector<int> true_pos(num_class, 0);
  vector<int> data_count(num_class, 0);
  vector<int> label_count(num_class, 0);
  for (int i = 0; i < 100; ++i) {
    if (kIgnoreLabelValue == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      continue;
    }
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < 10; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
      ++true_pos[max_id];
    }
    ++count;
    ++data_count[max_id];
    ++label_count[this->blob_bottom_label_->data_at(i, 0, 0, 0)];
  }
  EXPECT_EQ(count, 97);
  TypeParam jaccard = 0;
  for (int i = 0; i < num_class; ++i) {
    int uni = label_count[i] + data_count[i] - true_pos[i];
    jaccard += uni > 0 ?
      static_cast<TypeParam>(label_count[i]) * true_pos[i] / uni : 0;
    TypeParam jaccard_per_class = (true_pos[i] > 0 ?
      static_cast<TypeParam>(true_pos[i]) / uni : 0);
    EXPECT_NEAR(this->blob_top_per_class_->data_at(i, 0, 0, 0),
                jaccard_per_class, 1e-4);
  }
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
              jaccard / TypeParam(count), 1e-4);
}

}  // namespace caffe
