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
class OneOfNLayerTest : public ::testing::Test {
 protected:
  OneOfNLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 1, 1, 1024)),
        blob_top_(new Blob<Dtype>()),
        output_n_(69) {
    Caffe::set_mode(Caffe::CPU);
   
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(68);
    UniformFiller<Dtype> filler(filler_param);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~OneOfNLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t output_n_;
};

TYPED_TEST_CASE(OneOfNLayerTest, TestDtypes);

TYPED_TEST(OneOfNLayerTest, TestSetup) {
  
  // Create LayerParameter with the known parameters.
  // with output_n set as 69
  LayerParameter param;

  OneOfNParameter* one_of_n_param = param.mutable_one_of_n_param();
  int output_n = this->output_n_;
  one_of_n_param->set_output_n(output_n);

  // Test that the layer setup got the correct parameters.
  OneOfNLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), output_n);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1024);

}


TYPED_TEST(OneOfNLayerTest, TestCPU) {

  LayerParameter layer_param;

  OneOfNParameter* one_of_n_param = layer_param.mutable_one_of_n_param();
  int output_n = this->output_n_;
  one_of_n_param->set_output_n(output_n);

  OneOfNLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count(3, 4);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      int label = bottom_data[i * dim + j];
      for (int k = 0; k < output_n; ++k) {
        if (k == label) {
          EXPECT_EQ(top_data[i * dim * output_n + j + k * dim], 1);
        }
        else {
          EXPECT_EQ(top_data[i * dim * output_n + j + k * dim], 0);
        }
      }
      EXPECT_GE(top_data[i], 0);
    }
  }
}

}  // namespace caffe
