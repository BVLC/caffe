#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class InnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_nobatch_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~InnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

/** @brief TestSetUp while toggling tranpose flag
 */
TYPED_TEST(InnerProductLayerTest, TestSetUpTranposeFalse) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(2, this->blob_top_->num());
  EXPECT_EQ(1, this->blob_top_->height());
  EXPECT_EQ(1, this->blob_top_->width());
  EXPECT_EQ(10, this->blob_top_->channels());
  EXPECT_EQ(2, layer->blobs()[0]->num_axes());
  EXPECT_EQ(10, layer->blobs()[0]->shape(0));
  EXPECT_EQ(60, layer->blobs()[0]->shape(1));
}

/** @brief TestSetUp while toggling tranpose flag
 */
TYPED_TEST(InnerProductLayerTest, TestSetUpTranposeTrue) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(2, this->blob_top_->num());
  EXPECT_EQ(1, this->blob_top_->height());
  EXPECT_EQ(1, this->blob_top_->width());
  EXPECT_EQ(10, this->blob_top_->channels());
  EXPECT_EQ(2, layer->blobs()[0]->num_axes());
  EXPECT_EQ(60, layer->blobs()[0]->shape(0));
  EXPECT_EQ(10, layer->blobs()[0]->shape(1));
}

TYPED_TEST(InnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

/**
 * @brief Init. an IP layer without transpose + random weights,
 * run Forward, save the result.
 * Init. another IP layer with transpose.
 * manually copy and transpose the weights from the first IP layer,
 * then run Forward on the same input and check that the result is the same
 */
TYPED_TEST(InnerProductLayerTest, TestForwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->set_transpose(false);
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count = this->blob_top_->count();
    Blob<Dtype>* const top = new Blob<Dtype>();
    top->ReshapeLike(*this->blob_top_);
    caffe_copy(count, this->blob_top_->cpu_data(), top->mutable_cpu_data());
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(new Blob<Dtype>());
    inner_product_param->set_transpose(true);
    shared_ptr<InnerProductLayer<Dtype> > ip_t(
        new InnerProductLayer<Dtype>(layer_param));
    ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count_w = layer->blobs()[0]->count();
    EXPECT_EQ(count_w, ip_t->blobs()[0]->count());
    // manually copy and transpose the weights from 1st IP layer into 2nd
    const Dtype* w = layer->blobs()[0]->cpu_data();
    Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
    const int width = layer->blobs()[0]->shape(1);
    const int width_t = ip_t->blobs()[0]->shape(1);
    for (int i = 0; i < count_w; ++i) {
      int r = i / width;
      int c = i % width;
      w_t[c*width_t+r] = w[r*width+c];  // copy while transposing
    }
    // copy bias from 1st IP layer to 2nd IP layer
    ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
    caffe_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
        ip_t->blobs()[1]->mutable_cpu_data());
    ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(count, this->blob_top_->count())
        << "Invalid count for top blob for IP with transpose.";
    Blob<Dtype>* const top_t = new Blob<Dtype>();\
    top_t->ReshapeLike(*this->blob_top_vec_[0]);
    caffe_copy(count,
      this->blob_top_vec_[0]->cpu_data(),
      top_t->mutable_cpu_data());
    const Dtype* data = top->cpu_data();
    const Dtype* data_t = top_t->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(data[i], data_t[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestForwardNoBatch) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    InnerProductLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestGradientTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(11);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->set_transpose(true);
    InnerProductLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestBackwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->set_transpose(false);
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // copy top blob
    Blob<Dtype>* const top = new Blob<Dtype>();
    top->CopyFrom(*this->blob_top_, false, true);
    // fake top diff
    Blob<Dtype>* const diff = new Blob<Dtype>();
    diff->ReshapeLike(*this->blob_top_);
    {
      FillerParameter filler_param;
      UniformFiller<Dtype> filler(filler_param);
      filler.Fill(diff);
    }
    caffe_copy(this->blob_top_vec_[0]->count(),
      diff->cpu_data(),
      this->blob_top_vec_[0]->mutable_cpu_diff());
    vector<bool> propagate_down(1, true);
    layer->Backward(this->blob_top_vec_,
        propagate_down,
        this->blob_bottom_vec_);
    // copy first ip's weights and their diffs
    Blob<Dtype>* const w = new Blob<Dtype>();
    w->CopyFrom(*layer->blobs()[0], false, true);
    w->CopyFrom(*layer->blobs()[0], true, true);
    // copy bottom diffs
    Blob<Dtype>* const bottom_diff = new Blob<Dtype>();
    bottom_diff->CopyFrom(*this->blob_bottom_vec_[0], true, true);
    // repeat original top with tranposed ip
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(new Blob<Dtype>());
    inner_product_param->set_transpose(true);
    shared_ptr<InnerProductLayer<Dtype> > ip_t(
        new InnerProductLayer<Dtype>(layer_param));
    ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // manually copy and transpose the weights from 1st IP layer into 2nd
    {
      const Dtype* w_src = w->cpu_data();
      Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
      const int width = layer->blobs()[0]->shape(1);
      const int width_t = ip_t->blobs()[0]->shape(1);
      for (int i = 0; i < layer->blobs()[0]->count(); ++i) {
        int r = i / width;
        int c = i % width;
        w_t[c*width_t+r] = w_src[r*width+c];  // copy while transposing
      }
      // copy bias from 1st IP layer to 2nd IP layer
      ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
      caffe_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
          ip_t->blobs()[1]->mutable_cpu_data());
    }
    ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    caffe_copy(this->blob_top_vec_[0]->count(),
      diff->cpu_data(),
      this->blob_top_vec_[0]->mutable_cpu_diff());
    ip_t->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
    const Dtype* data = w->cpu_diff();
    const Dtype* data_t = ip_t->blobs()[0]->cpu_diff();
    const int WIDTH = layer->blobs()[0]->shape(1);
    const int WIDTH_T = ip_t->blobs()[0]->shape(1);
    for (int i = 0; i < layer->blobs()[0]->count(); ++i) {
      int r = i / WIDTH;
      int c = i % WIDTH;
      EXPECT_NE(Dtype(0.), data[r*WIDTH+c]);
      EXPECT_FLOAT_EQ(data[r*WIDTH+c], data_t[c*WIDTH_T+r]);
    }
    data = bottom_diff->cpu_diff();
    data_t = this->blob_bottom_vec_[0]->cpu_diff();
    for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
      EXPECT_NE(Dtype(0.), data[i]);
      EXPECT_FLOAT_EQ(data[i], data_t[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
