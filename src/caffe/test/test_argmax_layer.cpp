#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ArgMaxLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  ArgMaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 10, 20, 20)),
        blob_top_(new Blob<Dtype>()),
        top_k_(5) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ArgMaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t top_k_;
};

TYPED_TEST_CASE(ArgMaxLayerTest, TestDtypes);

TYPED_TEST(ArgMaxLayerTest, TestSetup) {
  LayerParameter layer_param;
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
}

TYPED_TEST(ArgMaxLayerTest, TestSetupMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 2);
}

TYPED_TEST(ArgMaxLayerTest, TestSetupAxis) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(0);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_->shape(2));
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(ArgMaxLayerTest, TestSetupAxisNegativeIndexing) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-2);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(ArgMaxLayerTest, TestSetupAxisMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), argmax_param->top_k());
  EXPECT_EQ(this->blob_top_->shape(3), this->blob_bottom_->shape(3));
}

TYPED_TEST(ArgMaxLayerTest, TestCPU) {
  LayerParameter layer_param;
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i];
    max_val = bottom_data[i * dim + max_ind];
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(this->blob_top_->data_at(i, 0, 0, 0), 0);
    EXPECT_LE(this->blob_top_->data_at(i, 0, 0, 0), dim);
    for (int j = 0; j < this->top_k_; ++j) {
      max_ind = this->blob_top_->data_at(i, 0, j, 0);
      max_val = bottom_data[i * dim + max_ind];
      int count = 0;
      for (int k = 0; k < dim; ++k) {
        if (bottom_data[i * dim + k] > max_val) {
          ++count;
        }
      }
      EXPECT_EQ(j, count);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxValTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(this->blob_top_->data_at(i, 0, 0, 0), 0);
    EXPECT_LE(this->blob_top_->data_at(i, 0, 0, 0), dim);
    for (int j = 0; j < this->top_k_; ++j) {
      max_ind = this->blob_top_->data_at(i, 0, j, 0);
      max_val = this->blob_top_->data_at(i, 1, j, 0);
      EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
      int count = 0;
      for (int k = 0; k < dim; ++k) {
        if (bottom_data[i * dim + k] > max_val) {
          ++count;
        }
      }
      EXPECT_EQ(j, count);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUAxis) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(0);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  int max_ind;
  TypeParam max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[1]; ++i) {
    for (int j = 0; j < shape[2]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        max_ind = this->blob_top_->data_at(0, i, j, k);
        max_val = this->blob_bottom_->data_at(max_ind, i, j, k);
        EXPECT_GE(max_ind, 0);
        EXPECT_LE(max_ind, shape[0]);
        for (int l = 0; l < shape[0]; ++l) {
          EXPECT_LE(this->blob_bottom_->data_at(l, i, j, k), max_val);
        }
      }
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUAxisTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(2);
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  int max_ind;
  TypeParam max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[3]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_ind = this->blob_top_->data_at(i, j, m, k);
          max_val = this->blob_bottom_->data_at(i, j, max_ind, k);
          EXPECT_GE(max_ind, 0);
          EXPECT_LE(max_ind, shape[2]);
          int count = 0;
          for (int l = 0; l < shape[2]; ++l) {
            if (this->blob_bottom_->data_at(i, j, l, k) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUAxisMaxValTopK) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_axis(-1);
  argmax_param->set_top_k(this->top_k_);
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  TypeParam max_val;
  std::vector<int> shape = this->blob_bottom_->shape();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        for (int m = 0; m < this->top_k_; ++m) {
          max_val = this->blob_top_->data_at(i, j, k, m);
          int count = 0;
          for (int l = 0; l < shape[3]; ++l) {
            if (this->blob_bottom_->data_at(i, j, k, l) > max_val) {
              ++count;
            }
          }
          EXPECT_EQ(m, count);
        }
      }
    }
  }
}

}  // namespace caffe
