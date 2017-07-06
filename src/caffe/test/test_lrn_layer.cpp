#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lrn_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class LRNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LRNLayerTest()
      : epsilon_(Dtype(1e-3)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    if (std::is_same<Dtype, half_float::half>::value)
      epsilon_ = 5e-2;
  }
  virtual ~LRNLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceLRNForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void LRNLayerTest<TypeParam>::ReferenceLRNForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LRNParameter lrn_param = layer_param.lrn_param();
  Dtype alpha = lrn_param.alpha();
  Dtype beta = lrn_param.beta();
  int_tp size = lrn_param.local_size();
  switch (lrn_param.norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    for (int_tp n = 0; n < blob_bottom.num(); ++n) {
      for (int_tp c = 0; c < blob_bottom.channels(); ++c) {
        for (int_tp h = 0; h < blob_bottom.height(); ++h) {
          for (int_tp w = 0; w < blob_bottom.width(); ++w) {
            int_tp c_start = c - (size - 1) / 2;
            int_tp c_end = min(c_start + size, blob_bottom.channels());
            c_start = max(c_start, (int_tp)0);
            Dtype scale = 1.;
            for (int_tp i = c_start; i < c_end; ++i) {
              Dtype value = blob_bottom.data_at(n, i, h, w);
              scale += value * value * alpha / size;
            }
            *(top_data + blob_top->offset(n, c, h, w)) =
              blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
          }
        }
      }
    }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    for (int_tp n = 0; n < blob_bottom.num(); ++n) {
      for (int_tp c = 0; c < blob_bottom.channels(); ++c) {
        for (int_tp h = 0; h < blob_bottom.height(); ++h) {
          int_tp h_start = h - (size - 1) / 2;
          int_tp h_end = min(h_start + size, blob_bottom.height());
          h_start = max(h_start, (int_tp)0);
          for (int_tp w = 0; w < blob_bottom.width(); ++w) {
            Dtype scale = 1.;
            int_tp w_start = w - (size - 1) / 2;
            int_tp w_end = min(w_start + size, blob_bottom.width());
            w_start = max(w_start, (int_tp)0);
            for (int_tp nh = h_start; nh < h_end; ++nh) {
              for (int_tp nw = w_start; nw < w_end; ++nw) {
                Dtype value = blob_bottom.data_at(n, c, nh, nw);
                scale += value * value * alpha / (size * size);
              }
            }
            *(top_data + blob_top->offset(n, c, h, w)) =
              blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
          }
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

TYPED_TEST_CASE(LRNLayerTest, TestDtypesAndDevices);

TYPED_TEST(LRNLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  // for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
  //   std::cout << "CPU diff " << this->blob_bottom_->cpu_diff()[i]
  //       << std::endl;
  // }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestGradientAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  // for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
  //   std::cout << "CPU diff " << this->blob_bottom_->cpu_diff()[i]
  //       << std::endl;
  // }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestSetupWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestForwardWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestGradientWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNLRNLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNLRNLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNLRNLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceLRNForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename Dtype>
void CuDNNLRNLayerTest<Dtype>::ReferenceLRNForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LRNParameter lrn_param = layer_param.lrn_param();
  Dtype alpha = lrn_param.alpha();
  Dtype beta = lrn_param.beta();
  int_tp size = lrn_param.local_size();
  switch (lrn_param.norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    for (int_tp n = 0; n < blob_bottom.num(); ++n) {
      for (int_tp c = 0; c < blob_bottom.channels(); ++c) {
        for (int_tp h = 0; h < blob_bottom.height(); ++h) {
          for (int_tp w = 0; w < blob_bottom.width(); ++w) {
            int_tp c_start = c - (size - 1) / 2;
            int_tp c_end = min(c_start + size, blob_bottom.channels());
            c_start = max(c_start, (int_tp)0);
            Dtype scale = 1.;
            for (int_tp i = c_start; i < c_end; ++i) {
              Dtype value = blob_bottom.data_at(n, i, h, w);
              scale += value * value * alpha / size;
            }
            *(top_data + blob_top->offset(n, c, h, w)) =
              blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
          }
        }
      }
    }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    for (int_tp n = 0; n < blob_bottom.num(); ++n) {
      for (int_tp c = 0; c < blob_bottom.channels(); ++c) {
        for (int_tp h = 0; h < blob_bottom.height(); ++h) {
          int_tp h_start = h - (size - 1) / 2;
          int_tp h_end = min(h_start + size, blob_bottom.height());
          h_start = max(h_start, (int_tp)0);
          for (int_tp w = 0; w < blob_bottom.width(); ++w) {
            Dtype scale = 1.;
            int_tp w_start = w - (size - 1) / 2;
            int_tp w_end = min(w_start + size, blob_bottom.width());
            w_start = max(w_start, (int_tp)0);
            for (int_tp nh = h_start; nh < h_end; ++nh) {
              for (int_tp nw = w_start; nw < w_end; ++nw) {
                Dtype value = blob_bottom.data_at(n, c, nh, nw);
                scale += value * value * alpha / (size * size);
              }
            }
            *(top_data + blob_top->offset(n, c, h, w)) =
              blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
          }
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

TYPED_TEST_CASE(CuDNNLRNLayerTest, TestDtypes);

TYPED_TEST(CuDNNLRNLayerTest, TestForwardAcrossChannelsCuDNN) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    // typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    CuDNNLRNLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<TypeParam> top_reference;
    this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
        &top_reference);
    for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                  this->epsilon_);
    }
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestForwardAcrossChannelsLargeRegionCuDNN) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    typedef TypeParam Dtype;
    LayerParameter layer_param;
    layer_param.mutable_lrn_param()->set_local_size(15);
    CuDNNLRNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype> top_reference;
    this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
        &top_reference);
    for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                  this->epsilon_);
    }
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientAcrossChannelsCuDNN) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    typedef TypeParam Dtype;
    LayerParameter layer_param;
    CuDNNLRNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      this->blob_top_->mutable_cpu_diff()[i] = 1.;
    }
    vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
    layer.Backward(this->blob_top_vec_, propagate_down,
                   this->blob_bottom_vec_);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestForwardWithinChannel) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    typedef TypeParam Dtype;
    LayerParameter layer_param;
    layer_param.mutable_lrn_param()->set_norm_region(
        LRNParameter_NormRegion_WITHIN_CHANNEL);
    layer_param.mutable_lrn_param()->set_local_size(3);
    CuDNNLCNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype> top_reference;
    this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
        &top_reference);
    for (int_tp i = 0; i < this->blob_bottom_->count(); ++i) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                  this->epsilon_);
    }
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientWithinChannel) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    typedef TypeParam Dtype;
    LayerParameter layer_param;
    layer_param.mutable_lrn_param()->set_norm_region(
        LRNParameter_NormRegion_WITHIN_CHANNEL);
    layer_param.mutable_lrn_param()->set_local_size(3);
    CuDNNLCNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      this->blob_top_->mutable_cpu_diff()[i] = 1.;
    }
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientAcrossChannelsLargeRegionCuDNN) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    typedef TypeParam Dtype;
    LayerParameter layer_param;
    layer_param.mutable_lrn_param()->set_local_size(15);
    CuDNNLRNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      this->blob_top_->mutable_cpu_diff()[i] = 1.;
    }
    vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
    layer.Backward(this->blob_top_vec_, propagate_down,
                   this->blob_bottom_vec_);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

#endif

template <typename Dtype>
class LRNFuseLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  LRNFuseLayerTest()
      : epsilon_(Dtype(1e-3)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
    blob_bottom_->Reshape(1, 32, 55, 55);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LRNFuseLayerTest() { delete blob_bottom_; delete blob_top_; }

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LRNFuseLayerTest, TestDtypes);

TYPED_TEST(LRNFuseLayerTest, TestForwardAcrossChannelsFusePoolMax) {
  LayerParameter layer_param;

  Blob<TypeParam> top_reference;
  LRNLayer<TypeParam> lrnLayer(layer_param);

  // calculate reference value by lrn layer followed by pooling layer
  lrnLayer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  lrnLayer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LayerParameter pooling_param;
  pooling_param.mutable_pooling_param()->
    set_pool(PoolingParameter_PoolMethod_MAX);
  pooling_param.mutable_pooling_param()->add_kernel_size(3);
  pooling_param.mutable_pooling_param()->add_stride(2);
  PoolingLayer<TypeParam> pooling_layer(pooling_param);
  vector<Blob<TypeParam>*> top_reference_vec;
  top_reference_vec.push_back(&top_reference);
  pooling_layer.SetUp(this->blob_top_vec_, top_reference_vec);
  pooling_layer.Forward(this->blob_top_vec_, top_reference_vec);
  // calculate result by lrn fused with pooling layer.
  LayerParameter fused_layer_param;
  fused_layer_param.set_phase(TEST);
  fused_layer_param.mutable_lrn_param()->
    set_fuse_type(LRNParameter_FuseType_FUSED_POOL_MAX);
  fused_layer_param.mutable_lrn_param()->set_unit_test_mode(true);
  fused_layer_param.mutable_lrn_param()->mutable_pooling_param()->
    set_pool(PoolingParameter_PoolMethod_MAX);
  fused_layer_param.mutable_lrn_param()->mutable_pooling_param()->
    add_kernel_size(3);
  fused_layer_param.mutable_lrn_param()->mutable_pooling_param()->
    add_stride(2);

  bool test_fuse_kernel[2] = {true, false};
  for (int_tp index = 0; index < 2; index++) {
    fused_layer_param.mutable_lrn_param()->
      set_unit_test_fuse_kernel(test_fuse_kernel[index]);
    LRNLayer<TypeParam> fused_layer(fused_layer_param);
    fused_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    fused_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int_tp i = 0; i < top_reference.count(); ++i) {
      EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                  this->epsilon_);
    }
    caffe_set(top_reference.count(), TypeParam(0),
              this->blob_top_->mutable_cpu_data());
  }
}

}  // namespace caffe
