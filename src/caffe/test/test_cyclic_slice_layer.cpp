#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/cyclic_slice_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CyclicSliceLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CyclicSliceLayerTest()
    : blob_bottom_(new Blob<Dtype>(1, 2, 3, 3)),
    blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    // fill the values
    Dtype input[] =  {1, 2, 3,
                      4, 5, 6,
                      7, 8, 9,
                      11, 12, 13,
                      14, 15, 16,
                      17, 18, 19};
    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
      this->blob_bottom_vec_[0]->mutable_cpu_data()[i] = input[i];
    }
  }
  virtual ~CyclicSliceLayerTest() { delete blob_bottom_;
    delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CyclicSliceLayerTest, TestDtypesAndDevices);

TYPED_TEST(CyclicSliceLayerTest, TestSetup) {
  this->SetUp();
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CyclicSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(0),
    this->blob_bottom_vec_[0]->shape(0)*4);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(1),
    this->blob_bottom_vec_[0]->shape(1));
  EXPECT_EQ(this->blob_top_vec_[0]->shape(2),
    this->blob_bottom_vec_[0]->shape(2));
  EXPECT_EQ(this->blob_top_vec_[0]->shape(3),
    this->blob_bottom_vec_[0]->shape(3));
}


TYPED_TEST(CyclicSliceLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CyclicSliceLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype expected_data[] ={1, 2, 3,
                          4, 5, 6,
                          7, 8, 9,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          7, 4, 1,
                          8, 5, 2,
                          9, 6, 3,
                          17, 14, 11,
                          18, 15, 12,
                          19, 16, 13,
                          9, 8, 7,
                          6, 5, 4,
                          3, 2, 1,
                          19, 18, 17,
                          16, 15, 14,
                          13, 12, 11,
                          3, 6, 9,
                          2, 5, 8,
                          1, 4, 7,
                          13, 16, 19,
                          12, 15, 18,
                          11, 14, 17};

  for (int i = 0; i < this->blob_top_vec_[0]->count(); ++i) {
    EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[i], expected_data[i]);
  }
}

}  // namespace caffe
