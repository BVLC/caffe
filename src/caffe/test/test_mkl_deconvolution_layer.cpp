/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef MKL2017_SUPPORTED
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/mkl_layers.hpp"
#include "caffe/layers/deconv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Since ConvolutionLayerTest checks the shared conv/deconv code in detail,
// we'll just do a simple forward test and a gradient check.
template <typename TypeParam>
class MKLDeconvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MKLDeconvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        ref_blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        ref_blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->ref_blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    ref_blob_bottom_vec_.push_back(ref_blob_bottom_);
    ref_blob_top_vec_.push_back(ref_blob_top_);

  }

  virtual ~MKLDeconvolutionLayerTest() {
    delete blob_bottom_;
    delete ref_blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete ref_blob_top_;
    delete blob_top_2_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const ref_blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const ref_blob_top_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> ref_blob_bottom_vec_;
  vector<Blob<Dtype>*> ref_blob_top_vec_;

};

TYPED_TEST_CASE(MKLDeconvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(MKLDeconvolutionLayerTest, TestSetup) {
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
      new MKLDeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 13);
  EXPECT_EQ(this->blob_top_->width(), 9);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 13);
  EXPECT_EQ(this->blob_top_2_->width(), 9);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new MKLDeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 13);
  EXPECT_EQ(this->blob_top_->width(), 9);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 13);
  EXPECT_EQ(this->blob_top_2_->width(), 9);
}

TYPED_TEST(MKLDeconvolutionLayerTest, TestSimpleMKLDeconvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new MKLDeconvolutionLayer<Dtype>(layer_param));
  shared_ptr<Layer<Dtype> > ref_layer(
      new DeconvolutionLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer->SetUp(this->ref_blob_bottom_vec_, this->ref_blob_top_vec_);
  // constant-fill the bottom blobs
  FillerParameter filler_param;
  filler_param.set_value(1.);

  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->ref_blob_bottom_);

  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  ref_layer->Forward(this->ref_blob_bottom_vec_, this->ref_blob_top_vec_);
  // simply check that accumulation works with overlapping filters
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* ref_top_data = this->ref_blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          Dtype expected = 3.1;
          bool h_overlap = h % 2 == 0 && h > 0
            && h < this->blob_top_->height() - 1;
          bool w_overlap = w % 2 == 0 && w > 0
            && w < this->blob_top_->width() - 1;
          if (h_overlap && w_overlap) {
            expected += 9;
          } else if (h_overlap || w_overlap) {
            expected += 3;
          }

          EXPECT_NEAR(top_data[this->blob_top_->offset(n, c, h, w)],
              expected, 1e-4);
          EXPECT_NEAR(ref_top_data[this->blob_top_->offset(n, c, h, w)],
              expected, 1e-4);
        }
      }
    }
  }

  // set top_diff
  Dtype* top_diff = this->blob_top_->mutable_cpu_diff();
  Dtype* ref_top_diff = this->ref_blob_top_->mutable_cpu_diff();
  for( int n = 0; n < this->blob_top_->num(); ++n) {
      for( int c = 0; c < this->blob_top_->channels(); ++c) {
          for( int h=0; h < this->blob_top_->height(); ++h) {
              for(int w = 0; w < this->blob_top_->width(); ++w) {
                top_diff[this->blob_top_->offset(n, c, h, w)] = ref_top_data[this->blob_top_->offset(n, c, h, w)];
                ref_top_diff[this->blob_top_->offset(n, c, h, w)] = ref_top_data[this->blob_top_->offset(n, c, h, w)];
              }
          }
      }
  }


  vector<bool> need_backward({true});
  layer->Backward(this->blob_top_vec_, need_backward, this->blob_bottom_vec_);
  ref_layer->Backward(this->ref_blob_top_vec_, need_backward, this->ref_blob_bottom_vec_);

  // check backward data
  const Dtype* bottom_diff = this->blob_bottom_->cpu_diff();
  const Dtype* ref_bottom_diff = this->ref_blob_bottom_->cpu_diff();
  for( int n = 0; n < this->blob_bottom_->num(); ++n) {
      for( int c = 0; c < this->blob_bottom_->channels(); ++c) {
          for( int h=0; h < this->blob_bottom_->height(); ++h) {
              for(int w = 0; w < this->blob_bottom_->width(); ++w) {
                EXPECT_NEAR(bottom_diff[this->blob_bottom_->offset(n, c, h, w)],
                        ref_bottom_diff[this->blob_bottom_->offset(n, c, h, w)],
                        1e-4);
              }
          }
      }
  }
  // check backward weights
  for (int i = 0; i < layer->blobs().size(); ++i) {
      Blob<Dtype>* blob = layer->blobs()[i].get();
      Blob<Dtype>* ref_blob = ref_layer->blobs()[i].get();
      const Dtype* weights_diff = blob->cpu_diff();
      const Dtype* ref_weights_diff = ref_blob->cpu_diff();
      for( int n = 0; n < blob->num(); ++n) {
          for( int c = 0; c <blob->channels(); ++c) {
              for( int h = 0; h < blob->height(); ++h) {
                  for( int w =0; w < blob->width(); ++w) {
                    //printf("%.4f ", weights_diff[blob->offset(n, c, h, w)] - ref_weights_diff[ref_blob->offset(n, c, h, w)]);
                    EXPECT_NEAR(weights_diff[blob->offset(n, c, h, w)],
                            ref_weights_diff[ref_blob->offset(n, c, h, w)],
                            1e-3);
                  }
              }
          }
      }
  }
}
TYPED_TEST(MKLDeconvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(2);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  MKLDeconvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
TYPED_TEST(MKLDeconvolutionLayerTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 11;
  const int kernel_w = 13;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 15;
  bottom_shape[1] = 12;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_num_output(18);
  convolution_param->set_bias_term(false);
  convolution_param->set_group(6);
  convolution_param->set_kernel_h(kernel_h);
  convolution_param->set_kernel_w(kernel_w);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  Blob<Dtype> weights;
  Blob<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    MKLDeconvolutionLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    top_diff.ReshapeLike(*this->blob_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1, layer.blobs().size());
    copy_diff = false; reshape = true;
    weights.CopyFrom(*layer.blobs()[0], copy_diff, reshape);
  }
  vector<bool> propagate_down(1, true);
  Blob<Dtype> result_2d;
  Blob<Dtype> backward_result_2d;
  Blob<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_2d.
    convolution_param->set_force_nd_im2col(false);
    MKLDeconvolutionLayer<Dtype> layer_2d(layer_param);
    layer_2d.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_2d.blobs().size());
    copy_diff = false; reshape = false;
    layer_2d.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_2d.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_2d.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_2d.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_2d.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  Blob<Dtype> result_nd;
  Blob<Dtype> backward_result_nd;
  Blob<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_nd.
    convolution_param->set_force_nd_im2col(true);
    MKLDeconvolutionLayer<Dtype> layer_nd(layer_param);
    layer_nd.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_nd.blobs().size());
    copy_diff = false; reshape = false;
    layer_nd.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_nd.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_nd.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_nd.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_nd.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (int i = 0; i < result_2d.count(); ++i)  {
    EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (int i = 0; i < backward_result_2d.count(); ++i) {
    EXPECT_EQ(backward_result_2d.cpu_diff()[i],
              backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(),
            backward_weight_result_2d.count());
  for (int i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i],
              backward_weight_result_nd.cpu_diff()[i]);
  }
}

#if 0
TYPED_TEST(MKLDeconvolutionLayerTest, TestGradient3D) {
  typedef typename TypeParam::Dtype Dtype;
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 2;
  bottom_shape[3] = 3;
  bottom_shape[4] = 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(2);
  convolution_param->add_stride(2);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  MKLDeconvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
#endif
}  // namespace caffe

#endif
