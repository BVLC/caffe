#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/confusion_matrix_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ConfusionMatrixLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  ConfusionMatrixLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
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

  virtual ~ConfusionMatrixLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConfusionMatrixLayerTest, TestDtypes);

TYPED_TEST(ConfusionMatrixLayerTest, TestSetup) {
  LayerParameter layer_param;
  ConfusionMatrixLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  int label_axis = 1;
  int num_classes = this->blob_bottom_vec_[0]->shape(label_axis);
  EXPECT_EQ(num_classes, 10);
  EXPECT_EQ(this->blob_top_->shape().size(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), num_classes);
  EXPECT_EQ(this->blob_top_->shape(1), num_classes);
}


TYPED_TEST(ConfusionMatrixLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  ConfusionMatrixLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int num_classes = 10;
  vector<int> shape(2, num_classes);
  vector<float> confusion_matrix(num_classes*num_classes, 0.);
  vector<int> num_tests_per_class(num_classes, 0);

  TypeParam max_value;
  int max_id;
  for (int i = 0; i < 100; ++i) {
    max_value = -FLT_MAX;
    max_id = 0;
    for (int j = 0; j < num_classes; ++j) {
      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
        max_id = j;
      }
    }
    int label = this->blob_bottom_label_->data_at(i, 0, 0, 0);
    num_tests_per_class[label]++;
    confusion_matrix[label*num_classes + max_id] += 1.0;
  }
  for (int j = 0; j < num_classes; ++j)
    if (num_tests_per_class[j] > 0) {
      for (int k = 0; k < num_classes; ++k)
        confusion_matrix[j*num_classes + k] =
           confusion_matrix[j*num_classes + k] /
           static_cast<float>(num_tests_per_class[j]);
  }
  // test output
  for (int j = 0; j < num_classes; ++j)
    for (int k = 0; k < num_classes; ++k) {
      EXPECT_NEAR(this->blob_top_vec_[0]->cpu_data()[j*num_classes + k],
        confusion_matrix[j*num_classes + k], 1e-4);
    }
}
}  // namespace caffe
