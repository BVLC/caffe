#ifndef CPU_ONLY  // CPU-GPU test
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename TypeParam>
class MergeCropLayerTest : public GPUDeviceTest<TypeParam> {
 protected:
  MergeCropLayerTest()
      : blob_bottom_a_(new Blob<TypeParam>()), blob_bottom_b_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()) {
  }

  virtual void SetUp() {
    blob_bottom_a_->Reshape(2, 3, 4, 2);
    blob_bottom_b_->Reshape(2, 3, 6, 4);
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
        for (int h = 0; h < a_h; ++h) {
          for (int w = 0; w < a_w; ++w) {
            blob_bottom_a_->mutable_cpu_data()[w + h * a_w + c * a_h * a_w
                + n * a_h * a_w * a_c] = (w + h * 10 + c * 100 + n * 1000
                + 10000);
          }
        }
      }
    }

    int b_h = blob_bottom_b_->height();
    int b_w = blob_bottom_b_->width();
    int b_c = blob_bottom_b_->channels();

    for (int n = 0; n < blob_bottom_b_->num(); ++n) {
      for (int c = 0; c < b_c; ++c) {
        for (int h = 0; h < b_h; ++h) {
          for (int w = 0; w < b_w; ++w) {
            blob_bottom_b_->mutable_cpu_data()[w + h * b_w + c * b_h * b_w
                + n * b_h * b_w * b_c] = -(w + h * 10 + c * 100 + n * 1000
                + 10000);
          }
        }
      }
    }

    LayerParameter layer_param;
    MergeCropLayer<TypeParam> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_a_->num());
    EXPECT_EQ(
        this->blob_top_->channels(),
        this->blob_bottom_a_->channels() + this->blob_bottom_b_->channels());
    EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_a_->height());
    EXPECT_EQ(this->blob_top_->width(), 2);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    // Test copy from A
    int offset = 0;
    for (int n = 0; n < blob_bottom_a_->num(); ++n) {
      for (int c = 0; c < a_c; ++c) {
        for (int h = 0; h < a_h; ++h) {
          for (int w = 0; w < a_w; ++w) {
            EXPECT_EQ(
                (w + h * 10 + c * 100 + n * 1000 + 10000),
                blob_top_->cpu_data()[offset + w + h * a_w + c * a_h * a_w]);
          }
        }
      }
      offset += a_h * a_w * (a_c + b_c);
    }

    // Test copy from B
    offset = a_h * a_w * a_c;
    for (int n = 0; n < blob_bottom_a_->num(); ++n) {
      for (int c = 0; c < b_c; ++c) {
        for (int h = 0; h < b_h; ++h) {
          for (int w = 0; w < b_w; ++w) {
            if (h >= (b_h - a_h) / 2 && h < a_h && w >= (b_w - a_w) / 2
                && w < a_w) {
              EXPECT_EQ(
                  -(w + h * 10 + c * 100 + n * 1000 + 10000),
                  blob_top_->mutable_cpu_data()[offset + (w - (b_h - a_h) / 2)
                      + (h - (b_h - a_h) / 2) * a_w + c * a_h * a_w]);
            }
          }
        }
      }
      offset += a_h * a_w * (a_c + b_c);
    }
  }

  void TestBackward() {
    int a_h = blob_bottom_a_->height();
    int a_w = blob_bottom_a_->width();
    int a_c = blob_bottom_a_->channels();

    for (int n = 0; n < blob_bottom_a_->num(); ++n) {
      for (int c = 0; c < a_c; ++c) {
        for (int h = 0; h < a_h; ++h) {
          for (int w = 0; w < a_w; ++w) {
            blob_bottom_a_->mutable_cpu_data()[w + h * a_w + c * a_h * a_w
                + n * a_h * a_w * a_c] = (w + h * 10 + c * 100 + n * 1000
                + 10000);
          }
        }
      }
    }

    int b_h = blob_bottom_b_->height();
    int b_w = blob_bottom_b_->width();
    int b_c = blob_bottom_b_->channels();

    for (int n = 0; n < blob_bottom_b_->num(); ++n) {
      for (int c = 0; c < b_c; ++c) {
        for (int h = 0; h < b_h; ++h) {
          for (int w = 0; w < b_w; ++w) {
            blob_bottom_b_->mutable_cpu_data()[w + h * b_w + c * b_h * b_w
                + n * b_h * b_w * b_c] = -(w + h * 10 + c * 100 + n * 1000
                + 10000);
          }
        }
      }
    }

    LayerParameter layer_param;
    MergeCropLayer<TypeParam> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    caffe_cpu_copy<TypeParam>(blob_top_->count(), blob_top_->cpu_data(),
                          blob_top_->mutable_cpu_diff());

    vector<bool> propagate_down(blob_bottom_vec_.size(), true);
    layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);

    // Test copy to A
    for (int n = 0; n < blob_bottom_a_->num(); ++n) {
      for (int c = 0; c < a_c; ++c) {
        for (int h = 0; h < a_h; ++h) {
          for (int w = 0; w < a_w; ++w) {
            EXPECT_EQ(
                (w + h * 10 + c * 100 + n * 1000 + 10000),
                blob_bottom_a_->cpu_diff()[w + h * a_w + c * a_h * a_w
                    + n * a_h * a_w * a_c]);
          }
        }
      }
    }
  }

  Blob<TypeParam>* const blob_bottom_a_;
  Blob<TypeParam>* const blob_bottom_b_;
  Blob<TypeParam>* const blob_top_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_;
};

TYPED_TEST_CASE(MergeCropLayerTest, TestDtypes);

TYPED_TEST(MergeCropLayerTest, TestSetup) {
  LayerParameter layer_param;
  MergeCropLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_a_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_a_->channels()
            + this->blob_bottom_b_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_a_->height());
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(MergeCropLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(MergeCropLayerTest, TestBackward) {
  this->TestBackward();
}

}  // namespace caffe
#endif  // !CPU_ONLY
