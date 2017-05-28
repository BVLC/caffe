#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mergecrop_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef CPU_ONLY  // CPU-GPU test

namespace caffe {

template<typename TypeParam>
class MergeCropLayerTest : public GPUDeviceTest<TypeParam> {
 protected:
  MergeCropLayerTest()
      : blob_bottom_a_(new Blob<TypeParam>()),
        blob_bottom_b_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()) {
  }

  virtual void SetUp() {
    vector<int_tp> shape_a;
    shape_a.push_back(1);
    shape_a.push_back(3);
    shape_a.push_back(3);
    shape_a.push_back(2);
    shape_a.push_back(6);

    vector<int_tp> shape_b;
    shape_b.push_back(1);
    shape_b.push_back(3);
    shape_b.push_back(5);
    shape_b.push_back(4);
    shape_b.push_back(8);

    blob_bottom_a_->Reshape(shape_a);
    blob_bottom_b_->Reshape(shape_b);
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

  void TestForward(MergeCropParameter_MergeOp op) {
    vector<int_tp> shape_a = blob_bottom_a_->shape();
    vector<int_tp> shape_b = blob_bottom_b_->shape();

    for (int_tp i = 0; i < blob_bottom_a_->count(); ++i) {
      int val = i;
      int out = 0;
      int dec = 1;
      for (int_tp d = shape_a.size() - 1; d  >= 0; --d) {
        out += (val % shape_a[d]) * dec;
        val /= shape_a[d];
        dec *= 10;
      }
      blob_bottom_a_->mutable_cpu_data()[i] = out;
      // std::cout << i << " - " << out << std::endl;
    }

    for (int_tp i = 0; i < blob_bottom_b_->count(); ++i) {
      int val = i;
      int out = 0;
      int dec = 1;
      for (int_tp d = shape_b.size() - 1; d  >= 0; --d) {
        out += (val % shape_b[d]) * dec;
        val /= shape_b[d];
        dec *= 10;
      }
      blob_bottom_b_->mutable_cpu_data()[i] = out;
      // std::cout << i << " - " << out << std::endl;
    }


    LayerParameter layer_param;
    MergeCropParameter *merge_param = layer_param.mutable_mergecrop_param();
    merge_param->set_operation(op);
    MergeCropLayer<TypeParam> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_a_->shape(0));
    if (op == MergeCropParameter_MergeOp_STACK) {
      EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_a_->shape(1)
              + this->blob_bottom_b_->shape(1));
    } else {
      EXPECT_EQ(this->blob_bottom_a_->shape(1), this->blob_bottom_b_->shape(1));
      EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_a_->shape(1));
    }

    for (int i = 2; i < this->blob_top_->shape().size(); ++i) {
      EXPECT_EQ(this->blob_top_->shape(i), this->blob_bottom_a_->shape(i));
    }
    vector<int_tp> shape_top = blob_top_->shape();

    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    if (op == MergeCropParameter_MergeOp_STACK) {
      // Test copy from A & B
      for (int_tp i = 0; i < blob_top_->count(); ++i) {
        int val = i < blob_bottom_a_->count() ? i : i - blob_bottom_a_->count();
        int out = 0;
        int dec = 1;
        for (int_tp d = shape_top.size() - 1; d  >= 0; --d) {
          if (i < blob_bottom_a_->count()) {
            out += (val % shape_a[d]) * dec;
            val /= shape_a[d];
            dec *= 10;
          } else {
            out += ((val % shape_a[d]) + (shape_b[d] - shape_a[d]) / 2) * dec;
            val /= shape_a[d];
            dec *= 10;
          }
        }
        EXPECT_EQ(out, blob_top_->mutable_cpu_data()[i]);
        // std::cout << i << " - " << out << std::endl;
      }
    } else {
      // Test copy from A & B
      for (int_tp i = 0; i < blob_top_->count(); ++i) {
        int val = i < blob_bottom_a_->count() ? i : i - blob_bottom_a_->count();
        int out = 0;
        int dec = 1;
        for (int_tp d = shape_top.size() - 1; d  >= 0; --d) {
          out += (val % shape_a[d]) * dec;
          out += ((val % shape_a[d]) + (shape_b[d] - shape_a[d]) / 2) * dec;
          val /= shape_a[d];
          dec *= 10;
        }
        EXPECT_EQ(out, blob_top_->mutable_cpu_data()[i]);
        // std::cout << i << " - " << out << std::endl;
      }
    }
  }

  void TestBackward(MergeCropParameter_MergeOp op) {
    vector<int_tp> shape_a = blob_bottom_a_->shape();
    vector<int_tp> shape_b = blob_bottom_b_->shape();
    vector<int_tp> shape_top = blob_top_->shape();

    for (int_tp i = 0; i < blob_bottom_a_->count(); ++i) {
      int val = i;
      int out = 0;
      int dec = 1;
      for (int_tp d = shape_a.size() - 1; d  >= 0; --d) {
        out += (val % shape_a[d]) * dec;
        val /= shape_a[d];
        dec *= 10;
      }
      blob_bottom_a_->mutable_cpu_data()[i] = out;
      // std::cout << i << " - " << out << std::endl;
    }

    for (int_tp i = 0; i < blob_bottom_b_->count(); ++i) {
      int val = i;
      int out = 0;
      int dec = 1;
      for (int_tp d = shape_b.size() - 1; d  >= 0; --d) {
        out += (val % shape_b[d]) * dec;
        val /= shape_b[d];
        dec *= 10;
      }
      blob_bottom_b_->mutable_cpu_data()[i] = out;
      // std::cout << i << " - " << out << std::endl;
    }

    LayerParameter layer_param;
    MergeCropParameter *merge_param = layer_param.mutable_mergecrop_param();
    merge_param->set_operation(op);
    MergeCropLayer<TypeParam> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    caffe_cpu_copy<TypeParam>(blob_top_->count(), blob_top_->cpu_data(),
                          blob_top_->mutable_cpu_diff());

    vector<bool> propagate_down(blob_bottom_vec_.size(), true);
    layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);

    // Test copy to A
    for (int_tp i = 0; i < blob_bottom_a_->count(); ++i) {
      int val = i;
      int out = 0;
      int dec = 1;
      for (int_tp d = shape_a.size() - 1; d  >= 0; --d) {
        out += (val % shape_a[d]) * dec;
        val /= shape_a[d];
        dec *= 10;
      }
      EXPECT_EQ(out, blob_bottom_a_->mutable_cpu_data()[i]);
      // std::cout << i << " - " << out << std::endl;
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

  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_a_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_a_->shape(1)
            + this->blob_bottom_b_->shape(1));

  for (int i = 2; i < this->blob_top_->shape().size(); ++i) {
    EXPECT_EQ(this->blob_top_->shape(i), this->blob_bottom_a_->shape(i));
  }
}

TYPED_TEST(MergeCropLayerTest, TestStackForward) {
  this->TestForward(MergeCropParameter_MergeOp_STACK);
}

TYPED_TEST(MergeCropLayerTest, TestStackBackward) {
  this->TestBackward(MergeCropParameter_MergeOp_STACK);
}

TYPED_TEST(MergeCropLayerTest, TestAddForward) {
  this->TestForward(MergeCropParameter_MergeOp_ADD);
}

TYPED_TEST(MergeCropLayerTest, TestAddBackward) {
  this->TestBackward(MergeCropParameter_MergeOp_ADD);
}

}  // namespace caffe
#endif  // !CPU_ONLY
