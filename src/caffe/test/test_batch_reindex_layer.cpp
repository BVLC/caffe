#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_reindex_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class BatchReindexLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BatchReindexLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_permute_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    vector<int> permsz;
    permsz.push_back(6);
    blob_bottom_permute_->Reshape(permsz);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    int perm[] = { 4, 0, 4, 0, 1, 2 };
    for (int i = 0; i < blob_bottom_permute_->count(); ++i) {
      blob_bottom_permute_->mutable_cpu_data()[i] = perm[i];
    }

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_permute_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BatchReindexLayerTest() {
    delete blob_bottom_permute_;
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_permute_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    LayerParameter layer_param;

    vector<int> sz;
    sz.push_back(5);
    sz.push_back(4);
    sz.push_back(3);
    sz.push_back(2);
    blob_bottom_->Reshape(sz);
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      blob_bottom_->mutable_cpu_data()[i] = i;
    }

    vector<int> permsz;
    permsz.push_back(6);
    blob_bottom_permute_->Reshape(permsz);
    int perm[] = { 4, 0, 4, 0, 1, 2 };
    for (int i = 0; i < blob_bottom_permute_->count(); ++i) {
      blob_bottom_permute_->mutable_cpu_data()[i] = perm[i];
    }
    BatchReindexLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), blob_bottom_permute_->num());
    EXPECT_EQ(blob_top_->channels(), blob_bottom_->channels());
    EXPECT_EQ(blob_top_->height(), blob_bottom_->height());
    EXPECT_EQ(blob_top_->width(), blob_bottom_->width());

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    int channels = blob_top_->channels();
    int height = blob_top_->height();
    int width = blob_top_->width();
    for (int i = 0; i < blob_top_->count(); ++i) {
      int n = i / (channels * width * height);
      int inner_idx = (i % (channels * width * height));
      EXPECT_EQ(
          blob_top_->cpu_data()[i],
          blob_bottom_->cpu_data()[perm[n] * channels * width * height
              + inner_idx]);
    }
  }
};

TYPED_TEST_CASE(BatchReindexLayerTest, TestDtypesAndDevices);

TYPED_TEST(BatchReindexLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(BatchReindexLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BatchReindexLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  }

}  // namespace caffe
