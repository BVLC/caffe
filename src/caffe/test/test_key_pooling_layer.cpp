#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/key_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename TypeParam>
class KeyPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  KeyPoolingLayerTest()
      : num_(5),
        blob_bottom_(new Blob<Dtype>()),
      	blob_bottom_keys_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_keys_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    // Fill the values.
    // Simply use the linear index of the array element to make things
    // easier to reason about.
    blob_bottom_->Reshape(num_, 1, 7, 1);
    // This should result in the following matrix:
    // [ 0  1  2  3  4  5  6]
    //   7  8  9 10 11 12 13]
    //  14 15 16 17 18 19 20]
    //  21 22 23 24 25 26 27]
    //  28 29 30 31 32 33 34]
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      this->blob_bottom_->mutable_cpu_data()[i] = i;
    }
    blob_bottom_keys_->Reshape(num_, 1, 1, 1);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_keys_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top_keys_);
  }
  virtual ~KeyPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_keys_;
    delete blob_top_;
    delete blob_top_keys_;
  }

  int num_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_keys_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_keys_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(KeyPoolingLayerTest, TestDtypesAndDevices);


TYPED_TEST(KeyPoolingLayerTest, TestAllKeysDifferentIsStdPooling) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);
  KeyPoolingLayer<Dtype> layer(layer_param);
  PoolingLayer<Dtype> ref_layer(layer_param);

  for (int i = 0; i < this->num_; ++i) {
    this->blob_bottom_keys_->mutable_cpu_data()[i] = i;
  }


  Blob<Dtype> ref_top;
  vector<Blob<Dtype>*> ref_top_vec;
  vector<Blob<Dtype>*> ref_bottom_vec;
  ref_top_vec.push_back(&ref_top);
  ref_bottom_vec.push_back(this->blob_bottom_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer.SetUp(ref_bottom_vec, ref_top_vec);

  EXPECT_EQ(this->blob_top_->shape(), ref_top.shape());
  EXPECT_EQ(this->blob_top_keys_->count(), this->num_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer.Forward(ref_bottom_vec, ref_top_vec);

  EXPECT_EQ(this->blob_top_keys_->count(), this->num_);
  for (int i = 0; i < this->num_; ++i) {
    EXPECT_EQ(this->blob_top_keys_->cpu_data()[i], i);
  }

  for (int i = 0; i < ref_top.count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i], ref_top.cpu_data()[i]);
  }
}


TYPED_TEST(KeyPoolingLayerTest, TestKeySameReduces) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);
  KeyPoolingLayer<Dtype> layer(layer_param);
  PoolingLayer<Dtype> ref_layer(layer_param);

  for (int i = 0; i < this->num_; ++i) {
    this->blob_bottom_keys_->mutable_cpu_data()[i] = 0;
  }


  Blob<Dtype> ref_top;
  vector<Blob<Dtype>*> ref_top_vec;
  vector<Blob<Dtype>*> ref_bottom_vec;
  ref_top_vec.push_back(&ref_top);
  ref_bottom_vec.push_back(this->blob_bottom_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer.SetUp(ref_bottom_vec, ref_top_vec);

  std::vector<int> expected_shape(ref_top.shape());
  expected_shape[0] = 1;

  EXPECT_EQ(this->blob_top_->shape(), expected_shape);
  EXPECT_EQ(this->blob_top_keys_->count(), 1);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer.Forward(ref_bottom_vec, ref_top_vec);

  EXPECT_EQ(this->blob_top_keys_->count(), 1);
  EXPECT_EQ(this->blob_top_->cpu_data()[0], this->blob_bottom_->count()-1);
  // for (int i = 0; i < this->blob_top_->count(); ++i) {
  //   EXPECT_EQ(this->blob_top_->cpu_data()[i], ref_top.cpu_data()[i]);
  // }
}


TYPED_TEST(KeyPoolingLayerTest, TestDifferentKeyGroups) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(1);
  pooling_param->set_kernel_w(1);

  KeyPoolingLayer<Dtype> layer(layer_param);
  PoolingLayer<Dtype> ref_layer(layer_param);

  for (int i = 0; i < this->num_; ++i) {
    if (i < this->num_/2) {
      this->blob_bottom_keys_->mutable_cpu_data()[i] = 0;
    } else {
      this->blob_bottom_keys_->mutable_cpu_data()[i] = 1;
    }
  }


  Blob<Dtype> ref_top;
  vector<Blob<Dtype>*> ref_top_vec;
  vector<Blob<Dtype>*> ref_bottom_vec;
  ref_top_vec.push_back(&ref_top);
  ref_bottom_vec.push_back(this->blob_bottom_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer.SetUp(ref_bottom_vec, ref_top_vec);

  std::vector<int> expected_shape(ref_top.shape());
  expected_shape[0] = 2;

  EXPECT_EQ(this->blob_top_->shape(), expected_shape);
  EXPECT_EQ(this->blob_top_keys_->count(), 2);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer.Forward(ref_bottom_vec, ref_top_vec);

  EXPECT_EQ(this->blob_top_keys_->count(), 2);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i], ref_top.cpu_data()[i]);
  }
}

}