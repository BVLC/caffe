#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/layers/upsample_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"


namespace caffe {

template <typename TypeParam>
class UpsampleLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UpsampleLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 2, 2);
    blob_bottom_mask_->Reshape(2, 3, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_mask_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UpsampleLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_bottom_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_bottom_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square Upsample layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
    upsample_param->set_scale(2);
    const int num = 4;
    const int channels = 3;
    blob_bottom_->Reshape(num, channels, 2, 2);
    blob_bottom_mask_->Reshape(num, channels, 2, 2);
    // Input: 4x 3 channels of:
    //     [1 2]
    //     [9 4]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 9;
      blob_bottom_->mutable_cpu_data()[i +  3] = 4;
    }
    // Input mask: 4x 3 channels of:
    //     [2  5 ]
    //     [12 14]
    for (int i = 0; i < 4 * num * channels; i += 4) {
      blob_bottom_mask_->mutable_cpu_data()[i +  0] = 2;
      blob_bottom_mask_->mutable_cpu_data()[i +  1] = 5;
      blob_bottom_mask_->mutable_cpu_data()[i +  2] = 12;
      blob_bottom_mask_->mutable_cpu_data()[i +  3] = 14;
    }
    UpsampleLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 4);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 4x 3 channels of:
    //     [0 0 1 0]
    //     [0 2 0 0]
    //     [0 0 0 0]
    //     [9 0 4 0]
    for (int i = 0; i < 16 * num * channels; i += 16) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 0);
    }
  }
  int MapIndexBottomToTop(int bottom_idx, int scale_w, 
                          int scale_h, bool randomize) {
    const int input_width = bottom_idx % blob_bottom_->width();
    const int input_height = bottom_idx / blob_bottom_->width();
    const int top_w = scale_w * blob_bottom_->width();
    int out_w = scale_w * input_width + (randomize ? rand() % scale_w : 0);
    int out_h = scale_h * input_height + (randomize ? rand() % scale_h : 0);
    int out_idx = out_w + out_h * top_w;
//     std::cout << "mask i, iw, ih, ow, oh, topw, outidx: " 
//               << bottom_idx << " " << input_width << " "
//               << input_height << " "
//               << out_w << " "
//               << out_h << " "
//               << top_w << " "
//               << out_idx << "\n";
    return out_idx;
  }
  void FillBottomMask(int scale_w, int scale_h, bool randomize = false) {
    Dtype* mask_data = blob_bottom_mask_->mutable_cpu_data();
    for(int n = 0; n < blob_bottom_->num(); ++n) {
      for(int c = 0; c < blob_bottom_->channels(); ++c) {
        for(int i = 0; i < blob_bottom_->height() * blob_bottom_->width(); ++i) {
          int idx = MapIndexBottomToTop(i, scale_w, scale_h, randomize);
          mask_data[i] = idx;
          
        }
        mask_data += blob_bottom_mask_->offset(0, 1);
      }
    }
  }
};

TYPED_TEST_CASE(UpsampleLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpsampleLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
  upsample_param->set_scale_h(2);
  upsample_param->set_scale_w(3);
  UpsampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height() * 2);
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width() * 3);
}

/*
TYPED_TEST(UpsampleLayerTest, PrintBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
  upsample_param->set_scale(2);
  this->FillBottomMask(2, 2);
  UpsampleLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_bottom_mask_->count(); ++i) {
    cout << "bottom mask data " << i << " " 
         << this->blob_bottom_mask_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  std::vector<bool> prop_down;
  prop_down.push_back(true);
  prop_down.push_back(false);
  layer.Backward(this->blob_top_vec_, prop_down, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(UpsampleLayerTest, TestForward) {
  this->TestForwardSquare();
}

TYPED_TEST(UpsampleLayerTest, TestForwardFromPool) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2,3,4,4);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
//   pooling_param->set_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  PoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 2);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->num(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_mask_->height(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->width(), 2);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param = upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(4);
  upsample_param->set_upsample_w(4);
  UpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);

  pooling_layer.Forward(pool_bottom_vec, this->blob_bottom_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for(int i = 0; i < this->blob_top_->count(); ++i) {
    if(top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (16-4)*2*3);
}
    
TYPED_TEST(UpsampleLayerTest, TestForwardFromPoolOddShape) {
  typedef typename TypeParam::Dtype Dtype;
  int kernel_w = 2;
  int kernel_h = 2;
  Blob<Dtype>* input_blob = new Blob<Dtype>();
  input_blob->Reshape(2,3,5,4);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(input_blob);
  std::vector<Blob<Dtype>*> pool_bottom_vec;
  pool_bottom_vec.push_back(input_blob);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_h(kernel_h);
  pooling_param->set_kernel_w(kernel_w);
  pooling_param->set_stride(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  PoolingLayer<Dtype> pooling_layer(layer_param);
  pooling_layer.SetUp(pool_bottom_vec, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), 2);
  EXPECT_EQ(this->blob_bottom_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_->height(), 3);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->num(), 2);
  EXPECT_EQ(this->blob_bottom_mask_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_mask_->height(), 3);
  EXPECT_EQ(this->blob_bottom_mask_->width(), 2);

  LayerParameter upsample_layer_param;
  UpsampleParameter* upsample_param = upsample_layer_param.mutable_upsample_param();
  upsample_param->set_upsample_h(5);
  upsample_param->set_upsample_w(4);
  UpsampleLayer<Dtype> upsample_layer(upsample_layer_param);
  upsample_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 4);

  pooling_layer.Forward(pool_bottom_vec, this->blob_bottom_vec_);
  upsample_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* pool_bottom_data = input_blob->cpu_data();
  int num_zeros = 0;
  for(int i = 0; i < this->blob_top_->count(); ++i) {
    if(top_data[i] != 0) {
      EXPECT_EQ(top_data[i], pool_bottom_data[i]);
    } else {
      ++num_zeros;
    }
  }
  EXPECT_EQ(num_zeros, (5*4-6)*2*3);
}

TYPED_TEST(UpsampleLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  for (int scale_h = 2; scale_h <= 3; ++scale_h) {
    for (int scale_w = 2; scale_w <= 3; ++scale_w) {
      LayerParameter layer_param;
      UpsampleParameter* upsample_param = layer_param.mutable_upsample_param();
      upsample_param->set_scale_h(scale_h);
      upsample_param->set_scale_w(scale_w);
      this->FillBottomMask(scale_w, scale_h);
      UpsampleLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_, 0);
//       this->blob_top_vec_.pop_back();
    }
  }
}

}  // namespace caffe
