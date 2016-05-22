#ifdef USE_LIBDNN

#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/libdnn_conv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// Comparative check difference limit
#define kappa 0.05
// Comparative check shape size limit
#define element_limit 10000000


namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void libdnn_convtest(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int_tp kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int_tp pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int_tp stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int_tp dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int_tp kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int_tp groups = conv_param->group();
  int_tp o_g = out->shape(1) / groups;
  int_tp k_g = in->shape(1) / groups;
  int_tp o_head, k_head;
  // Convolution
  vector<int_tp> weight_offset(4 + has_depth);
  vector<int_tp> in_offset(4 + has_depth);
  vector<int_tp> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int_tp n = 0; n < out->shape(0); n++) {
    for (int_tp g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int_tp o = 0; o < o_g; o++) {
        for (int_tp k = 0; k < k_g; k++) {
          for (int_tp z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int_tp y = 0; y < out->shape(2 + has_depth); y++) {
              for (int_tp x = 0; x < out->shape(3 + has_depth); x++) {
                for (int_tp r = 0; r < kernel_d; r++) {
                  for (int_tp p = 0; p < kernel_h; p++) {
                    for (int_tp q = 0; q < kernel_w; q++) {
                      int_tp in_z = z * stride_d - pad_d + r * dilation_d;
                      int_tp in_y = y * stride_h - pad_h + p * dilation_h;
                      int_tp in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                      }
                    }
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
    for (int_tp n = 0; n < out->shape(0); n++) {
      for (int_tp o = 0; o < out->shape(1); o++) {
        for (int_tp z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int_tp y = 0; y < out->shape(2 + has_depth); y++) {
            for (int_tp x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void libdnn_convtest(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void libdnn_convtest(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename Dtype>
class LibDNNConvolutionLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  LibDNNConvolutionLayerTest()
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

  virtual ~LibDNNConvolutionLayerTest() {
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

TYPED_TEST_CASE(LibDNNConvolutionLayerTest, TestDtypes);

TYPED_TEST(LibDNNConvolutionLayerTest, TestSetupLibDNN) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<TypeParam> > layer(
      new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
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
  layer.reset(new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(LibDNNConvolutionLayerTest, TestSimpleConvolutionLibDNN) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  libdnn_convtest(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  libdnn_convtest(this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(LibDNNConvolutionLayerTest, TestSimpleConvolutionGroupLibDNN) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  libdnn_convtest(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(LibDNNConvolutionLayerTest, TestSobelConvolutionLibDNN) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
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
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(1);
  convolution_param->set_bias_term(false);
  shared_ptr<Layer<TypeParam> > layer(
      new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 3));
  TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();
  for (int_tp c = 0; c < 3; ++c) {
    int_tp i = c * 9;  // 3 x 3 filter
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
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
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
  layer.reset(new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 1));
  TypeParam* weights_1 = layer->blobs()[0]->mutable_cpu_data();
  for (int_tp c = 0; c < 3; ++c) {
    int_tp i = c * 3;  // 3 x 1 filter
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 2;
    weights_1[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
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
  layer.reset(new LibDNNConvolutionLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 1, 1, 3));
  TypeParam* weights_2 = layer->blobs()[0]->mutable_cpu_data();
  weights_2[0] = -1;
  weights_2[1] =  0;
  weights_2[2] =  1;
  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
  // Test equivalence of full and separable filters.
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const TypeParam* sep_top_data = this->blob_top_2_->cpu_data();
  for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}

TYPED_TEST(LibDNNConvolutionLayerTest, TestGradientLibDNN) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  LibDNNConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LibDNNConvolutionLayerTest, TestGradientGroupLibDNN) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  LibDNNConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

template<typename TypeParam>
class LibDNNConvolutionNDLayerTest : public GPUDeviceTest<TypeParam> {
 protected:
  LibDNNConvolutionNDLayerTest()
      : blob_bottom_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()) {
  }

  virtual void SetUp() {
    BlobShape shape;
    shape.add_dim(1);  // Batch
    shape.add_dim(1);  // Channels
    shape.add_dim(5);  // Depth
    shape.add_dim(5);  // Height
    shape.add_dim(5);  // Width
    blob_bottom_->Reshape(shape);

    shape.add_dim(1);  // Batch
    shape.add_dim(1);  // Channels
    shape.add_dim(1);  // Depth
    shape.add_dim(1);  // Height
    shape.add_dim(1);  // Width
    blob_top_->Reshape(shape);

    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~LibDNNConvolutionNDLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  void TestForward() {
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(3);

    convolution_param->add_dilation(2);
    convolution_param->add_dilation(2);
    convolution_param->add_dilation(2);

    convolution_param->set_num_output(1);

    convolution_param->set_axis(1);

    convolution_param->mutable_weight_filler()->set_type("constant");
    convolution_param->mutable_weight_filler()->set_value(1);
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0);

    LibDNNConvolutionLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int_tp d = blob_bottom_->shape(2);
    int_tp h = blob_bottom_->shape(3);
    int_tp w = blob_bottom_->shape(4);

    TypeParam *bottom_data = blob_bottom_->mutable_cpu_data();

    TypeParam checksum = 0;

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          bottom_data[cw + ch * w + cd * w * h] =
              cw + ch * w + cd * w * h;
          if (cw % 2 == 0 && ch % 2 == 0 && cd % 2 == 0) {
            checksum += cw + ch * w + cd * w * h;
          }
        }
      }
    }

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const TypeParam *top_data = blob_top_->cpu_data();

    EXPECT_EQ(checksum, top_data[0]);
  }

  void TestBackward() {
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(3);
    convolution_param->add_kernel_size(3);

    convolution_param->add_dilation(2);
    convolution_param->add_dilation(2);
    convolution_param->add_dilation(2);

    convolution_param->set_num_output(1);

    convolution_param->set_axis(1);

    convolution_param->mutable_weight_filler()->set_type("constant");
    convolution_param->mutable_weight_filler()->set_value(1);
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0);

    LibDNNConvolutionLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    TypeParam *top_diff = blob_top_->mutable_cpu_diff();

    *top_diff = 1;

    std::vector<bool> prop_down;
    prop_down.push_back(true);

    layer.Backward(this->blob_top_vec_, prop_down, this->blob_bottom_vec_);

    const TypeParam *bottom_diff = blob_bottom_->cpu_diff();

    int_tp d = blob_bottom_->shape(2);
    int_tp h = blob_bottom_->shape(3);
    int_tp w = blob_bottom_->shape(4);

    for (int_tp cd = 0; cd < d; ++cd) {
      for (int_tp ch = 0; ch < h; ++ch) {
        for (int_tp cw = 0; cw < w; ++cw) {
          if (cw % 2 == 0 && ch % 2 == 0 && cd % 2 == 0) {
            EXPECT_EQ(1, bottom_diff[cw + ch * w + cd * w * h]);
          }
        }
      }
    }
  }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_top_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_;
};

TYPED_TEST_CASE(LibDNNConvolutionNDLayerTest, TestDtypes);

TYPED_TEST(LibDNNConvolutionNDLayerTest, TestSetup) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(3);
  convolution_param->add_kernel_size(3);
  convolution_param->add_kernel_size(3);

  convolution_param->add_dilation(2);
  convolution_param->add_dilation(2);
  convolution_param->add_dilation(2);

  convolution_param->set_num_output(4);


  LibDNNConvolutionLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(1, this->blob_top_->shape(2));
  EXPECT_EQ(1, this->blob_top_->shape(3));
  EXPECT_EQ(1, this->blob_top_->shape(4));
}

TYPED_TEST(LibDNNConvolutionNDLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(LibDNNConvolutionNDLayerTest, TestBackward) {
  this->TestBackward();
}


template<typename TypeParam>
class LibDNNComparativeTest : public GPUDeviceTest<TypeParam> {
 protected:
  LibDNNComparativeTest()
      : blob_bottom_(new Blob<TypeParam>()),
        blob_bottom_ref_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()),
        blob_top_ref_(new Blob<TypeParam>()),
        rng_(rd_()) {
  }

  virtual void SetUp() {
    // fill the values
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_ref_.push_back(blob_bottom_ref_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_ref_.push_back(blob_top_ref_);
  }

  virtual ~LibDNNComparativeTest() {
    delete blob_bottom_;
    delete blob_bottom_ref_;
    delete blob_top_;
    delete blob_top_ref_;
  }

  bool TestForward(int_tp testIdx) {
    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;

    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();

    std::uniform_int_distribution<int_tp> dimsRand(1, 3);
    std::uniform_int_distribution<int_tp> dilationRand(1, 8);
    std::uniform_int_distribution<int_tp> kernelRand(1, 7);
    std::uniform_int_distribution<int_tp> padRand(0, 5);
    std::uniform_int_distribution<int_tp> strideRand(1, 6);
    std::uniform_int_distribution<int_tp> biasRand(0, 1);
    std::uniform_int_distribution<int_tp> groupRand(1, 4);

    std::uniform_int_distribution<int_tp> batchRand(1, 10);
    std::uniform_int_distribution<int_tp> fmapRand(1, 64);

    int_tp batchsize = batchRand(this->rng_);
    int_tp groups = groupRand(this->rng_);
    int_tp fmaps_in = fmapRand(this->rng_) * groups;
    int_tp fmaps_out = fmapRand(this->rng_) * groups;

    int dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> sizeRand(1,
                pow(element_limit / (fmaps_in * fmaps_out * batchsize),
                1.0 / (static_cast<double>(dims))));


    BlobShape shape;
    shape.add_dim(batchsize);  // Batch
    shape.add_dim(fmaps_in);   // Channels

    convolution_param->set_group(groups);

    for (int_tp i = 0; i < dims; ++i) {
      convolution_param->add_kernel_size(kernelRand(this->rng_));
      convolution_param->add_dilation(dilationRand(this->rng_));
      convolution_param->add_pad(padRand(this->rng_));
      convolution_param->add_stride(strideRand(this->rng_));

      int_tp size = sizeRand(this->rng_);
      int_tp kernel_extent = convolution_param->dilation(i)
          * (convolution_param->kernel_size(i) - 1) + 1;
      size = std::max((int_tp)size,
                      (int_tp)(kernel_extent - 2 * convolution_param->pad(i)));
      shape.add_dim(size);
    }

    std::cout << "Shape in: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << shape.dim(i);
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Kernel: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->kernel_size(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Dilation: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->dilation(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Stride: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->stride(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Pad: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->pad(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Group: " << groups << std::endl;

    blob_bottom_->Reshape(shape);
    blob_bottom_ref_->Reshape(shape);

    convolution_param->set_num_output(fmaps_out);

    convolution_param->set_axis(1);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_value(1);

    int_tp grand = biasRand(this->rng_);
    if (grand == 0) {
      convolution_param->mutable_bias_filler()->set_type("constant");
      convolution_param->mutable_bias_filler()->set_value(0);
      convolution_param->set_bias_term(false);
    } else {
      convolution_param->mutable_bias_filler()->set_type("gaussian");
      convolution_param->mutable_bias_filler()->set_value(1);
      convolution_param->set_bias_term(true);
    }

    LibDNNConvolutionLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    ConvolutionLayer<TypeParam> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    for (int_tp i = 0; i < layer.blobs().size(); ++i) {
      caffe_cpu_copy(layer.blobs()[i]->count(),
                     layer.blobs()[i]->cpu_data(),
                     ref_layer.blobs()[i]->mutable_cpu_data());
    }

    caffe_rng_uniform(blob_bottom_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_bottom_->mutable_cpu_data());

    caffe_cpu_copy(blob_bottom_->count(), blob_bottom_->cpu_data(),
                   blob_bottom_ref_->mutable_cpu_data());

    caffe_set(blob_top_->count(),
              (TypeParam)0.0, blob_top_->mutable_cpu_data());
    caffe_set(blob_top_ref_->count(),
              (TypeParam)0.0, blob_top_ref_->mutable_cpu_data());

    /*layer.Tune(this->blob_top_vec_[0]->mutable_gpu_data(), nullptr,
               this->blob_bottom_vec_[0]->mutable_gpu_data(), nullptr,
               batchsize);*/

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ref_layer.Forward(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    EXPECT_EQ(blob_top_->count(), blob_top_ref_->count());

    const TypeParam *top_data = blob_top_->cpu_data();
    const TypeParam *ref_top_data = blob_top_ref_->cpu_data();

    std::cout << "Shape out: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << blob_top_->shape()[i];
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    bool failure = false;
    double tot_error = 0;
    double tot_value = 0;
    double tot_value_ref = 0;
    int_tp failure_count = 0;

    for (int_tp i = 0; i < blob_top_->count(); ++i) {
      bool fail = (fabs(top_data[i] - ref_top_data[i]) >= kappa);
      if (fail) {
        std::cout << "Value: " << top_data[i]
                  << ", expected: " << ref_top_data[i] << " (at " << i << ")"
                  << std::endl;
        tot_error += fabs(top_data[i] - ref_top_data[i]);
        tot_value += fabs(top_data[i]);
        tot_value_ref += fabs(ref_top_data[i]);
        ++failure_count;
      }
      failure |= fail;
    }
    std::cout << "Error count: " << failure_count
              << "/" << blob_top_->count() << std::endl;
    std::cout << "Difference: " << tot_error
              << " (value: " << tot_value << " vs " << tot_value_ref << ")"
              << std::endl;

    EXPECT_EQ(failure, false);
    return failure;
  }

  bool TestBackward(int_tp testIdx) {
    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;

    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();

    std::uniform_int_distribution<int_tp> dimsRand(1, 3);
    std::uniform_int_distribution<int_tp> dilationRand(1, 8);
    std::uniform_int_distribution<int_tp> kernelRand(1, 7);
    std::uniform_int_distribution<int_tp> padRand(0, 5);
    std::uniform_int_distribution<int_tp> strideRand(1, 6);
    std::uniform_int_distribution<int_tp> biasRand(0, 1);
    std::uniform_int_distribution<int_tp> groupRand(1, 4);

    std::uniform_int_distribution<int_tp> batchRand(1, 10);
    std::uniform_int_distribution<int_tp> fmapRand(1, 64);

    int_tp batchsize = batchRand(this->rng_);
    int_tp groups = groupRand(this->rng_);
    int_tp fmaps_in = fmapRand(this->rng_) * groups;
    int_tp fmaps_out = fmapRand(this->rng_) * groups;

    int dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> sizeRand(1,
                pow(element_limit / (fmaps_in * fmaps_out * batchsize),
                1.0 / (static_cast<double>(dims))));


    BlobShape shape;
    shape.add_dim(batchsize);  // Batch
    shape.add_dim(fmaps_in);   // Channels

    convolution_param->set_group(groups);

    for (int_tp i = 0; i < dims; ++i) {
      convolution_param->add_kernel_size(kernelRand(this->rng_));
      convolution_param->add_dilation(dilationRand(this->rng_));
      convolution_param->add_pad(padRand(this->rng_));
      convolution_param->add_stride(strideRand(this->rng_));

      int_tp size = sizeRand(this->rng_);
      int_tp kernel_extent = convolution_param->dilation(i)
          * (convolution_param->kernel_size(i) - 1) + 1;
      size = std::max((int_tp)size,
                      (int_tp)(kernel_extent - 2 * convolution_param->pad(i)));
      shape.add_dim(size);
    }

    std::cout << "Shape in: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << shape.dim(i);
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Kernel: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->kernel_size(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Dilation: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->dilation(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Stride: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->stride(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Pad: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->pad(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Group: " << groups << std::endl;

    blob_bottom_->Reshape(shape);
    blob_bottom_ref_->Reshape(shape);

    convolution_param->set_num_output(fmaps_out);

    convolution_param->set_axis(1);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_value(1);

    int_tp grand = biasRand(this->rng_);
    if (grand == 0) {
      convolution_param->mutable_bias_filler()->set_type("constant");
      convolution_param->mutable_bias_filler()->set_value(0);
      convolution_param->set_bias_term(false);
    } else {
      convolution_param->mutable_bias_filler()->set_type("gaussian");
      convolution_param->mutable_bias_filler()->set_value(1);
      convolution_param->set_bias_term(true);
    }

    LibDNNConvolutionLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    ConvolutionLayer<TypeParam> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    for (int_tp i = 0; i < layer.blobs().size(); ++i) {
      caffe_cpu_copy(layer.blobs()[i]->count(),
                     layer.blobs()[i]->cpu_data(),
                     ref_layer.blobs()[i]->mutable_cpu_data());
    }

    caffe_rng_uniform(blob_top_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_top_->mutable_cpu_diff());

    caffe_cpu_copy(blob_top_->count(), blob_top_->cpu_diff(),
                   blob_top_ref_->mutable_cpu_diff());

    caffe_rng_uniform(blob_bottom_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_bottom_->mutable_cpu_data());

    caffe_cpu_copy(blob_bottom_->count(), blob_bottom_->cpu_data(),
                   blob_bottom_ref_->mutable_cpu_data());


    caffe_set(blob_top_->count(),  (TypeParam)0.0,
              blob_top_->mutable_cpu_data());
    caffe_set(blob_top_ref_->count(), (TypeParam)0.0,
              blob_top_ref_->mutable_cpu_data());

    caffe_set(blob_bottom_->count(),  (TypeParam)0.0,
              blob_bottom_->mutable_cpu_diff());
    caffe_set(blob_bottom_ref_->count(), (TypeParam)0.0,
              blob_bottom_ref_->mutable_cpu_diff());


    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ref_layer.Forward(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    std::vector<bool> prop_down(1, true);

    layer.Backward(blob_top_vec_, prop_down, blob_bottom_vec_);
    ref_layer.Backward(blob_top_vec_ref_, prop_down, blob_bottom_vec_ref_);

    EXPECT_EQ(blob_bottom_->count(), blob_bottom_ref_->count());

    const TypeParam *bottom_diff = blob_bottom_->cpu_diff();
    const TypeParam *ref_bottom_diff = blob_bottom_ref_->cpu_diff();

    const TypeParam *weight_diff = layer.blobs()[0]->cpu_diff();
    const TypeParam *ref_weight_diff = ref_layer.blobs()[0]->cpu_diff();

    const TypeParam *bias_diff = nullptr;
    const TypeParam *ref_bias_diff = nullptr;

    if (grand == 0) {
    } else {
      bias_diff = layer.blobs()[1]->cpu_diff();
      ref_bias_diff = ref_layer.blobs()[1]->cpu_diff();
    }

    std::cout << "Shape out: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << blob_top_->shape()[i];
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    bool failure = false;
    double tot_error = 0;
    double tot_value = 0;
    double tot_value_ref = 0;
    int_tp failure_count = 0;

    for (int_tp i = 0; i < blob_bottom_->count(); ++i) {
      bool fail = (fabs(bottom_diff[i] - ref_bottom_diff[i]) >= kappa);
      if (fail) {
        std::cout << "Value: " << bottom_diff[i]
                  << ", expected: " << ref_bottom_diff[i] << " (at " << i << ")"
                  << std::endl;
        tot_error += fabs(bottom_diff[i] - ref_bottom_diff[i]);
        tot_value += fabs(bottom_diff[i]);
        tot_value_ref += fabs(ref_bottom_diff[i]);
        ++failure_count;
      }
      failure |= fail;
    }

    for (int_tp i = 0; i < layer.blobs()[0]->count(); ++i) {
      bool fail = (fabs(weight_diff[i] - ref_weight_diff[i]) >= kappa);
      if (fail) {
        std::cout << "Value: " << weight_diff[i]
                  << ", expected: " << ref_weight_diff[i] << " (at " << i << ")"
                  << std::endl;
        tot_error += fabs(weight_diff[i] - ref_weight_diff[i]);
        tot_value += fabs(weight_diff[i]);
        tot_value_ref += fabs(ref_weight_diff[i]);
        ++failure_count;
      }
      failure |= fail;
    }

    if (grand == 0) {
    } else {
      for (int_tp i = 0; i < layer.blobs()[1]->count(); ++i) {
        bool fail = (fabs(bias_diff[i] - ref_bias_diff[i]) >= kappa);
        if (fail) {
          std::cout << "Value: " << bias_diff[i]
                    << ", expected: " << ref_bias_diff[i] << " (at " << i << ")"
                    << std::endl;
          tot_error += fabs(bias_diff[i] - ref_bias_diff[i]);
          tot_value += fabs(bias_diff[i]);
          tot_value_ref += fabs(ref_bias_diff[i]);
          ++failure_count;
        }
        failure |= fail;
      }
    }

    std::cout << "Error count: " << failure_count
        << "/" << blob_bottom_->count() << std::endl;
    std::cout << "Difference: " << tot_error
        << " (value: " << tot_value << " vs " << tot_value_ref << ")"
        << std::endl;

    EXPECT_EQ(failure, false);
    return failure;
  }

  Blob<TypeParam>* const blob_bottom_;
  Blob<TypeParam>* const blob_bottom_ref_;
  Blob<TypeParam>* const blob_top_;
  Blob<TypeParam>* const blob_top_ref_;

  vector<Blob<TypeParam>*> blob_bottom_vec_;
  vector<Blob<TypeParam>*> blob_bottom_vec_ref_;
  vector<Blob<TypeParam>*> blob_top_vec_;
  vector<Blob<TypeParam>*> blob_top_vec_ref_;

  std::random_device rd_;
  std::mt19937 rng_;
};

TYPED_TEST_CASE(LibDNNComparativeTest, TestDtypes);

TYPED_TEST(LibDNNComparativeTest, TestForward) {
  for (int i = 0; i < 100; ++i) {
    if (this->TestForward(i)) {
      break;
    }
  }
}

TYPED_TEST(LibDNNComparativeTest, TestBackward) {
  for (int i = 0; i < 100; ++i) {
    if (this->TestBackward(i)) {
      break;
    }
  }
}



}  // namespace caffe
#endif  // USE_LIBDNN
