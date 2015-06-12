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

template<typename TypeParam>
class MergeCropLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MergeCropLayerTest()
      : blob_bottom_a_(new Blob<Dtype>()),
        blob_bottom_b_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    blob_bottom_a_->Reshape(2, 3, 4, 2, Caffe::GetDefaultDeviceContext());
    blob_bottom_b_->Reshape(2, 3, 6, 5, Caffe::GetDefaultDeviceContext());
    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MergeCropLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }

  void TestForward() {

    int a_h = blob_bottom_a_->height();
    int a_w = blob_bottom_a_->width();
    int a_c = blob_bottom_a_->channels();

    for (int n = 0; n < blob_bottom_a_->num(); ++n) {
      for (int c = 0; c < a_c; ++c) {
        for (int i = 0; i < a_h * a_w; ++i) {
          blob_bottom_a_->mutable_cpu_data()[i + c * a_h * a_w
              + n * a_h * a_w * a_c] = i + 100 * a_c;
        }
      }
    }

    int b_h = blob_bottom_b_->height();
    int b_w = blob_bottom_b_->width();
    int b_c = blob_bottom_b_->channels();

    for (int n = 0; n < blob_bottom_b_->num(); ++n) {
      for (int c = 0; c < b_c; ++c) {
        for (int i = 0; i < b_h * b_w; ++i) {
          blob_bottom_b_->mutable_cpu_data()[i + c * b_h * b_w
              + n * b_h * b_w * b_c] = -(i + 100 * b_c);
        }
      }
    }

    LayerParameter layer_param;
    MergeCropLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_a_->num());
    EXPECT_EQ(
        this->blob_top_->channels(),
        this->blob_bottom_a_->channels() + this->blob_bottom_b_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_a_->height());
    EXPECT_EQ(this->blob_top_->width(), 2);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    for (int i = 0; i < 5; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
    }
  }

  void TestBackward() {

  }

  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MergeCropLayerTest, TestDtypesAndDevices);

TYPED_TEST(MergeCropLayerTest, TestSetup){
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
MergeCropLayer<Dtype> layer(layer_param);
layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_a_->num());
EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_a_->channels() + this->blob_bottom_b_->channels());
EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_a_->height());
EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(MergeCropLayerTest, TestForward){
this->TestForward();
}

TYPED_TEST(MergeCropLayerTest, TestBackward){
this->TestBackward();
}

}
  // namespace caffe
