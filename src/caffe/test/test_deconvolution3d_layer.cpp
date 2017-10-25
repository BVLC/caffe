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

#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/deconv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Since ConvolutionLayerTest checks the shared conv/deconv code in detail,
// we'll just do a simple forward test and a gradient check.
int initial_list[]={2,3,6,4,6};
vector<int> initial_vector(initial_list,initial_list+5);
template <typename TypeParam>
class DeconvolutionLayerTest3d : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DeconvolutionLayerTest3d()
      : blob_bottom_(new Blob<Dtype>(initial_vector)),
        blob_bottom_2_(new Blob<Dtype>(initial_vector)),
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

  virtual ~DeconvolutionLayerTest3d() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DeconvolutionLayerTest3d, TestDtypesAndDevices);

TYPED_TEST(DeconvolutionLayerTest3d, TestSetup) {
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
      new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  int iarray[] = {2,4,13,9,13};
  vector<int> desired_shape(iarray, iarray+5);
  for(size_t i=0; i<desired_shape.size();i++){
    EXPECT_EQ(this->blob_top_->shape()[i], desired_shape[i]);
    EXPECT_EQ(this->blob_top_2_->shape()[i], desired_shape[i]);
  }

  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  desired_shape[1]=3;
  for(size_t i=0; i<desired_shape.size();i++){
    EXPECT_EQ(this->blob_top_->shape()[i], desired_shape[i]);
    EXPECT_EQ(this->blob_top_2_->shape()[i], desired_shape[i]);
  }
}

TYPED_TEST(DeconvolutionLayerTest3d, TestSimpleDeconvolution) {
  typedef typename TypeParam::Dtype Dtype;
  float fill_weight = 0.85;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  // test for a set of convolution parameter. kernel_size = 3, stride = 2, pad = 0, output_num = 4
  // std::cout << "Test for another kernel parameter  kernel_size = 3, stride = 2, pad = 0, output_num = 4"<<std::endl;
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(fill_weight);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // simply check that accumulation works with overlapping filters
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->shape()[0]; ++n) {
    for (int c = 0; c < this->blob_top_->shape()[1]; ++c) {
      for (int d = 0; d < this->blob_top_->shape()[2]; ++d) {
        for (int h = 0; h < this->blob_top_->shape()[3]; ++h) {
          for (int w = 0; w < this->blob_top_->shape()[4]; ++w) {
            Dtype expected = 0.1 + 3 * fill_weight;
            bool d_overlap = d % 2 == 0 && d > 0
              && d < this->blob_top_->shape()[2] - 1;
            bool h_overlap = h % 2 == 0 && h > 0
              && h < this->blob_top_->shape()[3] - 1;
            bool w_overlap = w % 2 == 0 && w > 0
              && w < this->blob_top_->shape()[4] - 1;
          
            if (d_overlap && h_overlap && w_overlap){
              expected += 21 * fill_weight;
            } else if ((d_overlap && h_overlap) || (d_overlap && w_overlap) || (h_overlap && w_overlap)){
              expected += 9 * fill_weight;
            } else if (d_overlap || h_overlap || w_overlap){
              expected += 3 * fill_weight;
            }
            int off_list[]={n,c,d,h,w};
            vector<int> off_set(off_list, off_list + 5);
            EXPECT_NEAR(top_data[this->blob_top_->offset(off_set)], expected, 1e-4);
            //std::cout << "self_cal: " << expected << "caffe_cal: " << top_data[this->blob_top_->offset(off_set)] << std::endl;
          }
        }
      }
    }
  }

  // Test for another kernel parameter  kernel_size = 2, stride = 1, pad = 0, output_num = 3
  // std::cout << "Test for another kernel parameter  kernel_size = 2, stride = 1, pad = 0, output_num = 3"<<std::endl;
  convolution_param->clear_kernel_size();
  convolution_param->add_kernel_size(2);
  convolution_param->clear_stride();
  convolution_param->add_stride(1);
  convolution_param->set_num_output(3);
  layer.reset(new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  filler_param.set_value(1.);
  caffe_set(this->blob_bottom_->count(), Dtype(0), this->blob_bottom_->mutable_cpu_data());
  caffe_set(this->blob_bottom_2_->count(), Dtype(0), this->blob_bottom_2_->mutable_cpu_data());
  caffe_set(this->blob_top_->count(), Dtype(0), this->blob_top_->mutable_cpu_data());
  caffe_set(this->blob_top_2_->count(), Dtype(0), this->blob_top_2_->mutable_cpu_data());
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // simply check that accumulation works with overlapping filters
  top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->shape()[0]; ++n) {
    for (int c = 0; c < this->blob_top_->shape()[1]; ++c) {
      for (int d = 0; d < this->blob_top_->shape()[2]; ++d) {
        for (int h = 0; h < this->blob_top_->shape()[3]; ++h) {
          for (int w = 0; w < this->blob_top_->shape()[4]; ++w) {
            Dtype expected = 0.1 + 3 * fill_weight;
            bool d_overlap = d > 0 && d < this->blob_top_->shape()[2] - 1;
            bool h_overlap = h > 0 && h < this->blob_top_->shape()[3] - 1;
            bool w_overlap = w > 0 && w < this->blob_top_->shape()[4] - 1;

            if (d_overlap && h_overlap && w_overlap){
              expected += 21 * fill_weight;
            } else if ((d_overlap && h_overlap) || (d_overlap && w_overlap) || (h_overlap && w_overlap)){
              expected += 9 * fill_weight;
            } else if (d_overlap || h_overlap || w_overlap){
              expected += 3 * fill_weight;
            }
            int off_list[]={n,c,d,h,w};
            vector<int> off_set(off_list, off_list + 5);
            EXPECT_NEAR(top_data[this->blob_top_->offset(off_set)], expected, 1e-4);
            // std::cout << "self_cal: " << expected << "caffe_cal: " << top_data[this->blob_top_->offset(off_set)] << std::endl;
          }
        }
      }
    }
  }

  // Test for another kernel parameter  kernel_size = 3, stride = 1, pad = 1, output_num = 5
  // std::cout << "Test for another kernel parameter  kernel_size = 3, stride = 1, pad = 1, output_num = 5"<<std::endl;
  convolution_param->clear_kernel_size();
  convolution_param->add_kernel_size(3);
  convolution_param->clear_stride();
  convolution_param->add_stride(1);
  convolution_param->clear_pad();
  convolution_param->add_pad(1);
  convolution_param->set_num_output(5);
  layer.reset(new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  filler_param.set_value(1.);
  caffe_set(this->blob_bottom_->count(), Dtype(0), this->blob_bottom_->mutable_cpu_data());
  caffe_set(this->blob_bottom_2_->count(), Dtype(0), this->blob_bottom_2_->mutable_cpu_data());
  caffe_set(this->blob_top_->count(), Dtype(0), this->blob_top_->mutable_cpu_data());
  caffe_set(this->blob_top_2_->count(), Dtype(0), this->blob_top_2_->mutable_cpu_data());
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // simply check that accumulation works with overlapping filters
  top_data = this->blob_top_->cpu_data();
  for (int n = 0; n < this->blob_top_->shape()[0]; ++n) {
    for (int c = 0; c < this->blob_top_->shape()[1]; ++c) {
      for (int d = 0; d < this->blob_top_->shape()[2]; ++d) {
        for (int h = 0; h < this->blob_top_->shape()[3]; ++h) {
          for (int w = 0; w < this->blob_top_->shape()[4]; ++w) {
            Dtype expected = 0.1 + 3 * 8 * fill_weight;
            bool d_overlap = d > 0 && d < this->blob_top_->shape()[2] - 1;
            bool h_overlap = h > 0 && h < this->blob_top_->shape()[3] - 1;
            bool w_overlap = w > 0 && w < this->blob_top_->shape()[4] - 1;

            if (d_overlap && h_overlap && w_overlap){
              expected += 3 * 19 * fill_weight;
            } else if ((d_overlap && h_overlap) || (d_overlap && w_overlap) || (h_overlap && w_overlap)){
              expected += 3 * 10 * fill_weight;
            } else if (d_overlap || h_overlap || w_overlap){
              expected += 3 * 4 * fill_weight;
            }
            int off_list[]={n,c,d,h,w};
            vector<int> off_set(off_list, off_list + 5);
            EXPECT_NEAR(top_data[this->blob_top_->offset(off_set)], expected, 1e-4);
            // std::cout << "self_cal: " << expected << "caffe_cal: " << top_data[this->blob_top_->offset(off_set)] << std::endl;
          }
        }
      }
    }
  }

  // Test for another kernel parameter  kernel_size = 2, stride = 2, pad = 0, output_num = 4
  // std::cout << "Test for another kernel parameter  kernel_size = 2, stride = 2, pad = 0, output_num = 4"<<std::endl;
  // reset the convolution parameters
  convolution_param->clear_kernel_size();
  convolution_param->add_kernel_size(2);
  convolution_param->clear_stride();
  convolution_param->add_stride(2);
  convolution_param->clear_pad();
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_mean(0.0);
  convolution_param->mutable_weight_filler()->set_std(0.1);
  // convolution_param->mutable_weight_filler()->set_type("constant");
  // convolution_param->mutable_weight_filler()->set_value(fill_weight);

  // reset layer
  layer.reset(new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  filler_param.set_value(1.);
  caffe_set(this->blob_bottom_->count(), Dtype(0), this->blob_bottom_->mutable_cpu_data());
  caffe_set(this->blob_bottom_2_->count(), Dtype(0), this->blob_bottom_2_->mutable_cpu_data());
  caffe_set(this->blob_top_->count(), Dtype(0), this->blob_top_->mutable_cpu_data());
  caffe_set(this->blob_top_2_->count(), Dtype(0), this->blob_top_2_->mutable_cpu_data());
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype* gaussian_weight = layer->blobs()[0]->mutable_cpu_data();
  EXPECT_EQ(layer->blobs()[0]->count(), 96);
  // check that deconvolution works as the transport of convolution
  top_data = this->blob_top_->cpu_data();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  for (int n = 0; n < this->blob_top_->shape()[0]; ++n) {
    for (int c = 0; c < this->blob_top_->shape()[1]; ++c) {
      for (int d = 0; d < this->blob_top_->shape()[2]; ++d) {
        for (int h = 0; h < this->blob_top_->shape()[3]; ++h) {
          for (int w = 0; w < this->blob_top_->shape()[4]; ++w) {
            Dtype expected = 0.1;
            for (int u = 0; u <= 1; u++) {
              for (int v = 0; v <= 1; v++) {
                for (int l = 0; l <= 1; l++) {
                  bool in_zone = ((d - u) >= 0 && (d - u) / 2 < this->blob_bottom_->shape()[2]) \
                              && ((h - v) >= 0 && (h - v) / 2 < this->blob_bottom_->shape()[3]) \
                              && ((w - l) >= 0 && (w - l) / 2 < this->blob_bottom_->shape()[4]);
                  bool at_pixel = ((d - u) % 2 == 0) && ((h - v) % 2 == 0) && ((w - l) % 2 == 0);
                  if (in_zone && at_pixel) {
                    for (int cb = 0; cb < this->blob_bottom_->shape()[1]; cb++) {
                      int bottom_list[] = {n, cb, (d - u) / 2, (h - v) / 2, (w - l) / 2};
                      vector<int> bottom_offset(bottom_list, bottom_list + 5);
                      int weight_offset = (((cb * this->blob_top_->shape()[1] + c) * 2 + u) * 2 + v) * 2 + l;
                      expected += gaussian_weight[weight_offset] * bottom_data[this->blob_bottom_->offset(bottom_offset)];
                    }
                  }
                }
              }
            }
            int off_list[] = {n, c, d, h, w};
            vector<int> off_set(off_list, off_list + 5);
            EXPECT_NEAR(top_data[this->blob_top_->offset(off_set)], expected, 1e-4);
          }
        }
      }
    }
  }

  // Test for another kernel parameter  kernel_size = 3, stride = 2, pad = 1, output_num = 4
  // std::cout << "Test for another kernel parameter  kernel_size = 2, stride = 2, pad = 0, output_num = 4"<<std::endl;
  // reset the convolution parameters
  convolution_param->clear_kernel_size();
  convolution_param->add_kernel_size(3);
  convolution_param->clear_stride();
  convolution_param->add_stride(2);
  convolution_param->clear_pad();
  convolution_param->add_pad(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_mean(0.0);
  convolution_param->mutable_weight_filler()->set_std(0.1);  
  // reset layer
  layer.reset(new DeconvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  filler_param.set_value(1.);
  caffe_set(this->blob_bottom_->count(), Dtype(0), this->blob_bottom_->mutable_cpu_data());
  caffe_set(this->blob_bottom_2_->count(), Dtype(0), this->blob_bottom_2_->mutable_cpu_data());
  caffe_set(this->blob_top_->count(), Dtype(0), this->blob_top_->mutable_cpu_data());
  caffe_set(this->blob_top_2_->count(), Dtype(0), this->blob_top_2_->mutable_cpu_data());
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  gaussian_weight = layer->blobs()[0]->mutable_cpu_data();
  EXPECT_EQ(layer->blobs()[0]->count(), 324);
  // check that deconvolution works as the transport of convolution
  top_data = this->blob_top_->cpu_data();
  bottom_data = this->blob_bottom_->cpu_data();
  for (int n = 0; n < this->blob_top_->shape()[0]; ++n) {
    for (int c = 0; c < this->blob_top_->shape()[1]; ++c) {
      for (int d = 0; d < this->blob_top_->shape()[2]; ++d) {
        for (int h = 0; h < this->blob_top_->shape()[3]; ++h) {
          for (int w = 0; w < this->blob_top_->shape()[4]; ++w) {
            Dtype expected = 0.1;
            for (int u = 0; u <= 2; u++) {
              for (int v = 0; v <= 2; v++) {
                for (int l = 0; l <= 2; l++) {
                  bool in_zone = ((d + 1 - u) >= 0 && (d + 1 - u) / 2 < this->blob_bottom_->shape()[2]) \
                              && ((h + 1 - v) >= 0 && (h + 1 - v) / 2 < this->blob_bottom_->shape()[3]) \
                              && ((w + 1 - l) >= 0 && (w + 1 - l) / 2 < this->blob_bottom_->shape()[4]);
                  bool at_pixel = ((d + 1 - u) % 2 == 0) && ((h + 1 - v) % 2 == 0) && ((w + 1 - l) % 2 == 0);
                  if (in_zone && at_pixel) {
                    for (int cb = 0; cb < this->blob_bottom_->shape()[1]; cb++) {
                      int bottom_list[] = {n, cb, (d + 1 - u) / 2, (h + 1 - v) / 2, (w + 1 - l) / 2};
                      vector<int> bottom_offset(bottom_list, bottom_list + 5);
                      int weight_offset = (((cb * this->blob_top_->shape()[1] + c) * 3 + u) * 3 + v) * 3 + l;
                      expected += gaussian_weight[weight_offset] * bottom_data[this->blob_bottom_->offset(bottom_offset)];
                    }
                  }
                }
              }
            }
            int off_list[] = {n, c, d, h, w};
            vector<int> off_set(off_list, off_list + 5);
            EXPECT_NEAR(top_data[this->blob_top_->offset(off_set)], expected, 1e-4);
          }
        }
      }
    }
  }
}

TYPED_TEST(DeconvolutionLayerTest3d, TestGradient3D) {
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
  DeconvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
