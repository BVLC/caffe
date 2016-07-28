#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_spatial_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#if defined(USE_GREENTEA) && defined(USE_INTEL_SPATIAL)

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype> static
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int_tp kernel_h, kernel_w;
  if (conv_param->has_kernel_w() || conv_param->has_kernel_h()) {
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
  // Groups
  int_tp groups = conv_param->group();
  int_tp o_g = out->shape(1) / groups;
  int_tp k_g = in->shape(1) / groups;
  int_tp o_head, k_head;
  // Convolution
  vector<int_tp> weight_offset(4);
  vector<int_tp> in_offset(4);
  vector<int_tp> out_offset(4);

  Dtype* out_data = out->mutable_cpu_data();
  for (int_tp n = 0; n < out->shape(0); n++) {
    for (int_tp g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int_tp o = 0; o < o_g; o++) {
        for (int_tp k = 0; k < k_g; k++) {
          for (int_tp y = 0; y < out->shape(2); y++) {
            for (int_tp x = 0; x < out->shape(3); x++) {
              for (int_tp p = 0; p < kernel_h; p++) {
                for (int_tp q = 0; q < kernel_w; q++) {
                  int_tp in_y = y * stride_h - pad_h + p;
                  int_tp in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < in->height()
                    && in_x >= 0 && in_x < in->width()) {
                    weight_offset[0] = o + o_head;
                    weight_offset[1] = k;
                    weight_offset[2] = p;
                    weight_offset[3] = q;
                    in_offset[0] = n;
                    in_offset[1] = k + k_head;
                    in_offset[2] = in_y;
                    in_offset[3] = in_x;
                    out_offset[0] = n;
                    out_offset[1] = o + o_head;
                    out_offset[2] = y;
                    out_offset[3] = x;
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
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int_tp n = 0; n < out->shape(0); n++) {
      for (int_tp o = 0; o < out->shape(1); o++) {
        for (int_tp y = 0; y < out->shape(2); y++) {
          for (int_tp x = 0; x < out->shape(3); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              out_offset[2] = y;
              out_offset[3] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
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
class ConvolutionLayerTest_Spatial : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvolutionLayerTest_Spatial()
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

  virtual ~ConvolutionLayerTest_Spatial() {
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

TYPED_TEST_CASE(ConvolutionLayerTest_Spatial, TestFloatAndDevices);

TYPED_TEST(ConvolutionLayerTest_Spatial, TestSetup_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->set_num_output(4);
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
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
    layer.reset(new ConvolutionLayerSpatial<Dtype>(layer_param));
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
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestSimpleConvolution_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->set_num_output(256);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestSimpleConvolution_Spatial3x3) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(1);
    convolution_param->set_num_output(1024);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}


TYPED_TEST(ConvolutionLayerTest_Spatial,
    TestSimpleConvolution_Spatial3x3xPad1) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(1);
    convolution_param->add_pad(3);
    convolution_param->set_num_output(4);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial,
    TestSimpleConvolution_Spatial11x11x1x2_caffenet_Conv1) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(11);
    convolution_param->set_group(1);
    convolution_param->add_stride(4);
    convolution_param->add_pad(9);
    convolution_param->set_num_output(96);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}


TYPED_TEST(ConvolutionLayerTest_Spatial,
    TestSimpleConvolution_Spatial5x5x1x2_caffenet_Conv2) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(5);
    convolution_param->set_group(1);
    convolution_param->add_stride(1);
    convolution_param->add_pad(3);
    convolution_param->set_num_output(96);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.7);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial,
    TestSimpleConvolution_Spatial3x3x1_caffenet_Conv3) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->set_group(1);
    convolution_param->add_stride(1);
    convolution_param->add_pad(1);
    convolution_param->set_num_output(384);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial,
    TestSimpleConvolution_Spatial3x3x1_caffenet_Conv4) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->set_group(3);
    convolution_param->add_stride(1);
    convolution_param->add_pad(1);
    convolution_param->set_num_output(384);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.7);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial,
    TestSimpleConvolution_Spatial3x3x2_caffenet_Conv5) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->set_group(1);
    convolution_param->add_stride(2);
    convolution_param->add_pad(1);
    convolution_param->set_num_output(256);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.7);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestSimpleConvolution_Spatial5x5) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(5);
    convolution_param->set_group(1);
    convolution_param->add_stride(2);
    convolution_param->add_pad(5);
    convolution_param->set_num_output(1024);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.7);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
    caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_2_));
    top_data = this->blob_top_2_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, Test1x1Convolution_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(1);
    convolution_param->add_stride(1);
    convolution_param->set_num_output(4);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("constant");
    convolution_param->mutable_bias_filler()->set_value(0.1);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestSimpleConvolutionGroup_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
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
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Check against reference convolution.
    const Dtype* top_data;
    const Dtype* ref_top_data;
    caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
        this->MakeReferenceTop(this->blob_top_));
    top_data = this->blob_top_->cpu_data();
    ref_top_data = this->ref_blob_top_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestSobelConvolution_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
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
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->set_num_output(1);
    convolution_param->set_bias_term(false);
    shared_ptr<Layer<Dtype> > layer(
        new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->blobs().resize(1);
    layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 3));
    Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
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
    layer.reset(new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->blobs().resize(1);
    layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 1));
    Dtype* weights_1 = layer->blobs()[0]->mutable_cpu_data();
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
    layer.reset(new ConvolutionLayerSpatial<Dtype>(layer_param));
    layer->blobs().resize(1);
    layer->blobs()[0].reset(new Blob<Dtype>(1, 1, 1, 3));
    Dtype* weights_2 = layer->blobs()[0]->mutable_cpu_data();
    weights_2[0] = -1;
    weights_2[1] =  0;
    weights_2[2] =  1;
    layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
    layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
    // Test equivalence of full and separable filters.
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype* sep_top_data = this->blob_top_2_->cpu_data();
    for (int_tp i = 0; i < this->blob_top_->count(); ++i) {
      EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
    }
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestGradient_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
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
    ConvolutionLayerSpatial<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, Test1x1Gradient_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
    this->blob_top_vec_.push_back(this->blob_top_2_);
    convolution_param->add_kernel_size(1);
    convolution_param->add_stride(1);
    convolution_param->set_num_output(2);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("gaussian");
    ConvolutionLayerSpatial<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

TYPED_TEST(ConvolutionLayerTest_Spatial, TestGradientGroup_Spatial) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_OpenCL) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();
    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(2);
    convolution_param->set_num_output(3);
    convolution_param->set_group(3);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_bias_filler()->set_type("gaussian");
    ConvolutionLayerSpatial<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

}  // namespace caffe
#endif
