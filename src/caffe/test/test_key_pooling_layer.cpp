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
        blob_top_mask_(new Blob<Dtype>()),
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
    blob_top_vec_.push_back(blob_top_mask_);
    blob_top_vec_.push_back(blob_top_keys_);
  }
  virtual ~KeyPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_keys_;
    delete blob_top_;
    delete blob_top_mask_;
    delete blob_top_keys_;
  }

  int num_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_keys_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  Blob<Dtype>* const blob_top_keys_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void CreateTopData() {
    vector<int> shape;
    shape.push_back(3);
    shape.push_back(3);
    shape.push_back(2);
    shape.push_back(3);

    // define top, meaning, partial derivatives
    blob_top_->Reshape(shape);
    // consider the following 3 partial derivatives
    // [1 2 1]  ||  [3 2 6]  ||  [ 2 1 5]
    // [1 0 1]  ||  [3 1 4]  ||  [-1 2 1]
    // where the channel index is added to each pixel
    // in order to produce different arrays for different channels
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_top_);

    // define top_mask, needed for MAX pooling tests
    blob_top_mask_->Reshape(shape);
    // channel 0,1
    // [0 0 0]  ||  [3 3 1]  ||  [5 5 5]
    // [0 0 0]  ||  [3 2 1]  ||  [4 4 4]
    // channel 2
    // [0 0 0]  ||  [1 2 1]  ||  [5 4 5]
    // [0 0 0]  ||  [3 1 3]  ||  [5 4 4]
    Dtype *top_mask = blob_top_mask_->mutable_cpu_data();
    // Collection 0
    for (int i = 0; i < blob_top_mask_->count(1); ++i) {
      top_mask[blob_top_mask_->offset(0) + i] = i;
    }
    for (int i = 0; i < shape[1]; ++i) {
      // Collection 1
      top_mask[blob_top_mask_->offset(1,i,0,0)] = blob_bottom_->offset(1,i,0,0);
      top_mask[blob_top_mask_->offset(1,i,0,1)] = blob_bottom_->offset(2,i,0,1);
      top_mask[blob_top_mask_->offset(1,i,0,2)] = blob_bottom_->offset(3,i,0,2);
      top_mask[blob_top_mask_->offset(1,i,1,0)] = blob_bottom_->offset(1,i,1,2);
      top_mask[blob_top_mask_->offset(1,i,1,1)] = blob_bottom_->offset(2,i,1,1);
      top_mask[blob_top_mask_->offset(1,i,1,2)] = blob_bottom_->offset(3,i,1,0);
      // Collection 2
      top_mask[blob_top_mask_->offset(2,i,0,0)] = blob_bottom_->offset(4,i,0,0);
      top_mask[blob_top_mask_->offset(2,i,0,1)] = blob_bottom_->offset(5,i,0,1);
      top_mask[blob_top_mask_->offset(2,i,0,2)] = blob_bottom_->offset(4,i,0,2);
      top_mask[blob_top_mask_->offset(2,i,1,0)] = blob_bottom_->offset(5,i,1,2);
      top_mask[blob_top_mask_->offset(2,i,1,1)] = blob_bottom_->offset(4,i,1,1);
      top_mask[blob_top_mask_->offset(2,i,1,2)] = blob_bottom_->offset(5,i,1,0);
    }
  }
  void TestBackwardMax() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(1);
    KeyPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);

    vector<bool> propagate_down(1, true);
    layer.Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);
    CheckBackwardMax();
  }

  void CheckBackwardMax() {
    // Expected output for partial derivatives in cpu_diff, channel 0
    // [1 2 1]  ||  [0 0 6] [0 0 0] [3 2 0]  ||  [0 0 0] [2 1 5]
    // [1 0 1]  ||  [0 0 4] [0 1 0] [3 0 0]  || [-1 2 1] [0 0 0]
    // channel 1
    // [2 3 2]  ||  [0 0 7] [0 0 0] [4 3 0]  ||  [0 0 0] [3 2 6]
    // [2 1 2]  ||  [0 0 5] [0 2 0] [4 0 0]  ||  [0 3 2] [0 0 0]
    // channel 2
    // [3 4 3]  ||  [5 0 8] [0 4 0] [0 0 0]  ||  [0 3 0] [4 0 7]
    // [3 2 3]  ||  [0 3 0] [0 0 0] [5 0 6]  ||  [0 4 3] [1 0 0]
    ASSERT_EQ(blob_bottom_keys_->shape(0), 6);
    vector<int> shape;
    shape.push_back(6);
    shape.push_back(3);
    shape.push_back(2);
    shape.push_back(3);
    ASSERT_EQ(blob_bottom_->shape(), shape);

    Blob<Dtype> store_diff;
    store_diff.CopyFrom(*blob_bottom_, true, true);

    Dtype *result = store_diff.mutable_cpu_data();
    for (int i = 0; i < blob_top_mask_->count(); ++i) {
      const int bottom_i = blob_top_mask_->cpu_data()[i];
      const Dtype expected = blob_top_->cpu_data()[i];
      EXPECT_EQ(result[bottom_i], expected);
      result[bottom_i] = Dtype(0);
    }

    for (int i = 0; i < store_diff.count(); ++i) {
      EXPECT_EQ(result[i], Dtype(0));
    }
  }

};

TYPED_TEST_CASE(KeyPoolingLayerTest, TestDtypesAndDevices);


TYPED_TEST(KeyPoolingLayerTest, TestAllKeysDifferentIsStdPooling) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(1);

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
  pooling_param->set_kernel_size(1);

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
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i], i + 28);
  }
}


TYPED_TEST(KeyPoolingLayerTest, TestDifferentKeyGroups) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);

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

  EXPECT_EQ(this->blob_top_keys_->count(), 2);

  EXPECT_EQ(this->blob_top_->cpu_data()[0], 13);
  EXPECT_EQ(this->blob_top_->cpu_data()[1], 34);

}


TYPED_TEST(KeyPoolingLayerTest, TestForwardMax) {
  typedef typename TypeParam::Dtype Dtype;
  const int num_collection = 3;
  const int num_images = 6;
  const int channels = 3;
  const int height = 2;
  const int width = 3;

  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(1);

  KeyPoolingLayer<Dtype> layer(layer_param);

  this->blob_bottom_keys_->Reshape(num_images, 1, 1, 1);
  Dtype *key_data = this->blob_bottom_keys_->mutable_cpu_data();
  key_data[0] = 0;
  key_data[1] = 1; key_data[2] = 1; key_data[3] = 1;
  key_data[4] = 2; key_data[5] = 2;

  this->blob_bottom_->Reshape(num_images, channels, height, width);
  // consider the following 3 collections, 6 images of format 3x2
  // [1 0 1]  ||  [1 0 0]    [0 2 0]    [3 3 0]  ||  [1 0 2]    [ 2 1 5]
  // [0 0 1]  ||  [0 0 1]    [0 2 0]    [3 0 0]  ||  [0 4 1]    [-1 2 1]
  // where the channel index is added to each pixel
  // in order to produce different arrays for different channels
  Dtype* image = this->blob_bottom_->mutable_cpu_data();
  for (int i = 0; i < channels; ++i) {
    // image 0
    image[this->blob_bottom_->offset(0, i) + 0] = 1 + i;
    image[this->blob_bottom_->offset(0, i) + 1] = 0 + i;
    image[this->blob_bottom_->offset(0, i) + 2] = 1 + i;
    image[this->blob_bottom_->offset(0, i) + 3] = 0 + i;
    image[this->blob_bottom_->offset(0, i) + 4] = 0 + i;
    image[this->blob_bottom_->offset(0, i) + 5] = 1 + i;
    // image 1
    image[this->blob_bottom_->offset(1, i) + 0] = 1 + i;
    image[this->blob_bottom_->offset(1, i) + 1] = 0 + i;
    image[this->blob_bottom_->offset(1, i) + 2] = 0 + i;
    image[this->blob_bottom_->offset(1, i) + 3] = 0 + i;
    image[this->blob_bottom_->offset(1, i) + 4] = 0 + i;
    image[this->blob_bottom_->offset(1, i) + 5] = 1 + i;
    // image 2
    image[this->blob_bottom_->offset(2, i) + 0] = 0 + i;
    image[this->blob_bottom_->offset(2, i) + 1] = 2 + i;
    image[this->blob_bottom_->offset(2, i) + 2] = 0 + i;
    image[this->blob_bottom_->offset(2, i) + 3] = 0 + i;
    image[this->blob_bottom_->offset(2, i) + 4] = 2 + i;
    image[this->blob_bottom_->offset(2, i) + 5] = 0 + i;
    // image 3
    image[this->blob_bottom_->offset(3, i) + 0] = 3 + i;
    image[this->blob_bottom_->offset(3, i) + 1] = 3 + i;
    image[this->blob_bottom_->offset(3, i) + 2] = 0 + i;
    image[this->blob_bottom_->offset(3, i) + 3] = 3 + i;
    image[this->blob_bottom_->offset(3, i) + 4] = 0 + i;
    image[this->blob_bottom_->offset(3, i) + 5] = 0 + i;
    // image 4
    image[this->blob_bottom_->offset(4, i) + 0] = 1 + i;
    image[this->blob_bottom_->offset(4, i) + 1] = 0 + i;
    image[this->blob_bottom_->offset(4, i) + 2] = 2 + i;
    image[this->blob_bottom_->offset(4, i) + 3] = 0 + i;
    image[this->blob_bottom_->offset(4, i) + 4] = 4 + i;
    image[this->blob_bottom_->offset(4, i) + 5] = 1 + i;
    // image 5
    image[this->blob_bottom_->offset(5, i) + 0] = 2 + i;
    image[this->blob_bottom_->offset(5, i) + 1] = 1 + i;
    image[this->blob_bottom_->offset(5, i) + 2] = 5 + i;
    image[this->blob_bottom_->offset(5, i) + 3] =-1 + i;
    image[this->blob_bottom_->offset(5, i) + 4] = 2 + i;
    image[this->blob_bottom_->offset(5, i) + 5] = 1 + i;
  }


  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), num_collection);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), height);
  EXPECT_EQ(this->blob_top_->width(), width);
  if (this->blob_top_vec_.size() > 1) {
    EXPECT_EQ(this->blob_top_mask_->num(), num_collection);
    EXPECT_EQ(this->blob_top_mask_->channels(), channels);
    EXPECT_EQ(this->blob_top_mask_->height(), height);
    EXPECT_EQ(this->blob_top_mask_->width(), width);
  }
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output for channel = 0
  // [1 0 1]  ||  [3 3 0]  ||  [2 1 5]
  // [0 0 1]  ||  [3 2 1]  ||  [0 4 1]
  // adding channel index per pixel gives output for channel != 0
  const Dtype* pooled = this->blob_top_->cpu_data();
  for (int i = 0; i < channels; ++i) {
    // output 0
    EXPECT_EQ(pooled[this->blob_top_->offset(0, i) + 0], 1 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(0, i) + 1], 0 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(0, i) + 2], 1 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(0, i) + 3], 0 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(0, i) + 4], 0 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(0, i) + 5], 1 + i);
    // output 1
    EXPECT_EQ(pooled[this->blob_top_->offset(1, i) + 0], 3 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(1, i) + 1], 3 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(1, i) + 2], 0 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(1, i) + 3], 3 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(1, i) + 4], 2 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(1, i) + 5], 1 + i);
    // output 2
    EXPECT_EQ(pooled[this->blob_top_->offset(2, i) + 0], 2 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(2, i) + 1], 1 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(2, i) + 2], 5 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(2, i) + 3], 0 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(2, i) + 4], 4 + i);
    EXPECT_EQ(pooled[this->blob_top_->offset(2, i) + 5], 1 + i);
  }
  if (this->blob_top_vec_.size() > 1) {
    // test the mask
    // Expected output for every channel
    // [0 0 0]  ||  [3 3 1]  ||  [5 5 5]
    // [0 0 0]  ||  [3 2 1]  ||  [4 4 4]
    const Dtype *mask = this->blob_top_mask_->cpu_data();
    for (int i = 0; i < channels; ++i) {
      // output 0
      // All MAX values are in image 0.
      EXPECT_EQ(mask[this->blob_top_mask_->offset(0, i, 0, 0)],
                this->blob_bottom_->offset(0, i, 0, 0));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(0, i, 0, 1)],
                this->blob_bottom_->offset(0, i, 0, 1));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(0, i, 0, 2)],
                this->blob_bottom_->offset(0, i, 0, 2));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(0, i, 1, 0)],
                this->blob_bottom_->offset(0, i, 1, 0));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(0, i, 1, 1)],
                this->blob_bottom_->offset(0, i, 1, 1));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(0, i, 1, 2)],
                this->blob_bottom_->offset(0, i, 1, 2));
      // output 1
      EXPECT_EQ(mask[this->blob_top_mask_->offset(1, i, 0, 0)],
                this->blob_bottom_->offset(3, i, 0, 0));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(1, i, 0, 1)],
                this->blob_bottom_->offset(3, i, 0, 1));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(1, i, 0, 2)],
                this->blob_bottom_->offset(1, i, 0, 2));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(1, i, 1, 0)],
                this->blob_bottom_->offset(3, i, 1, 0));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(1, i, 1, 1)],
                this->blob_bottom_->offset(2, i, 1, 1));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(1, i, 1, 2)],
                this->blob_bottom_->offset(1, i, 1, 2));
      // output 2
      EXPECT_EQ(mask[this->blob_top_mask_->offset(2, i, 0, 0)],
                this->blob_bottom_->offset(5, i, 0, 0));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(2, i, 0, 1)],
                this->blob_bottom_->offset(5, i, 0, 1));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(2, i, 0, 2)],
                this->blob_bottom_->offset(5, i, 0, 2));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(2, i, 1, 0)],
                this->blob_bottom_->offset(4, i, 1, 0));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(2, i, 1, 1)],
                this->blob_bottom_->offset(4, i, 1, 1));
      EXPECT_EQ(mask[this->blob_top_mask_->offset(2, i, 1, 2)],
                this->blob_bottom_->offset(4, i, 1, 2));
    }
  }
  if (this->blob_top_vec_.size() > 2) {
    // test the key map
    // Expected output for every channel
    // [0] ||  [1]  ||  [2]
    const Dtype *key = this->blob_top_keys_->cpu_data();
    EXPECT_EQ(key[0], 0);
    EXPECT_EQ(key[1], 1);
    EXPECT_EQ(key[2], 2);
  }
}

TYPED_TEST(KeyPoolingLayerTest, TestBackwardMax) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_->Reshape(6, 3, 2, 3);
  this->blob_bottom_keys_->Reshape(6, 1, 1, 1);
  Dtype *key_data = this->blob_bottom_keys_->mutable_cpu_data();
  key_data[0] = 0;
  key_data[1] = 1; key_data[2] = 1; key_data[3] = 1;
  key_data[4] = 2; key_data[5] = 2;

  this->CreateTopData();
  this->TestBackwardMax();
}

}