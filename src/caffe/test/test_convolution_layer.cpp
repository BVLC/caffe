#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_size()) {
    kernel_h = kernel_w = conv_param->kernel_size();
  } else {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  int pad_h, pad_w;
  if (!conv_param->has_pad_h()) {
    pad_h = pad_w = conv_param->pad();
  } else {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  int stride_h, stride_w;
  if (!conv_param->has_stride_h()) {
    stride_h = stride_w = conv_param->stride();
  } else {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;
  // Convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < out->height(); y++) {
            for (int x = 0; x < out->width(); x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < in->height()
                    && in_x >= 0 && in_x < in->width()) {
                    out_data[out->offset(n, o + o_head, y, x)] +=
                        in_data[in->offset(n, k + k_head, in_y, in_x)]
                        * weight_data[weights[0]->offset(o + o_head, k, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
            out_data[out->offset(n, o, y, x)] += bias_data[o];
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class ConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestSobelConvolution) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<Dtype> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<Dtype>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 convolution.
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 3));
  Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i +  0] = -1;
    weights[i +  1] =  0;
    weights[i +  2] =  1;
    weights[i +  3] = -2;
    weights[i +  4] =  0;
    weights[i +  5] =  2;
    weights[i +  6] = -1;
    weights[i +  7] =  0;
    weights[i +  8] =  1;
  }
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
  // (1) the [1 2 1] column filter
  vector<Blob<Dtype>*> sep_blob_bottom_vec;
  vector<Blob<Dtype>*> sep_blob_top_vec;
  shared_ptr<Blob<Dtype> > blob_sep(new Blob<Dtype>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  convolution_param->clear_kernel_size();
  convolution_param->clear_stride();
  convolution_param->set_kernel_h(3);
  convolution_param->set_kernel_w(1);
  convolution_param->set_stride_h(2);
  convolution_param->set_stride_w(1);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 1));
  Dtype* weights_1 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 2;
    weights_1[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  convolution_param->set_kernel_h(1);
  convolution_param->set_kernel_w(3);
  convolution_param->set_stride_h(1);
  convolution_param->set_stride_w(2);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 1, 3));
  Dtype* weights_2 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 1 x 3 filter
    weights_2[i +  0] = -1;
    weights_2[i +  1] =  0;
    weights_2[i +  2] =  1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // Test equivalence of full and separable filters.
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(ConvolutionLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

#ifdef USE_CUDNN

template <typename Dtype>
class CuDNNConvolutionLayerTest : public ::testing::Test {
 protected:
  CuDNNConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CuDNNConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNConvolutionLayerTest, TestDtypes);

TYPED_TEST(CuDNNConvolutionLayerTest, TestSetupCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionCuDNN) {
  // We will simply see if the convolution layer carries out averaging well.
  Caffe::set_mode(Caffe::GPU);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionGroupCuDNN) {
  // We will simply see if the convolution layer carries out averaging well.
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(CuDNNConvolutionLayerTest, TestSobelConvolutionCuDNN) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
  Caffe::set_mode(Caffe::GPU);
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<TypeParam> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<TypeParam>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 convolution.
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 3));
  TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i +  0] = -1;
    weights[i +  1] =  0;
    weights[i +  2] =  1;
    weights[i +  3] = -2;
    weights[i +  4] =  0;
    weights[i +  5] =  2;
    weights[i +  6] = -1;
    weights[i +  7] =  0;
    weights[i +  8] =  1;
  }
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
  // (1) the [1 2 1] column filter
  vector<Blob<TypeParam>*> sep_blob_bottom_vec;
  vector<Blob<TypeParam>*> sep_blob_top_vec;
  shared_ptr<Blob<TypeParam> > blob_sep(new Blob<TypeParam>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  convolution_param->clear_kernel_size();
  convolution_param->clear_stride();
  convolution_param->set_kernel_h(3);
  convolution_param->set_kernel_w(1);
  convolution_param->set_stride_h(2);
  convolution_param->set_stride_w(1);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 1));
  TypeParam* weights_1 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 2;
    weights_1[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  convolution_param->set_kernel_h(1);
  convolution_param->set_kernel_w(3);
  convolution_param->set_stride_h(1);
  convolution_param->set_stride_w(2);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 1, 3));
  TypeParam* weights_2 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 1 x 3 filter
    weights_2[i +  0] = -1;
    weights_2[i +  1] =  0;
    weights_2[i +  2] =  1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // Test equivalence of full and separable filters.
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const TypeParam* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}

TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientGroupCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

#endif

}  // namespace caffe
