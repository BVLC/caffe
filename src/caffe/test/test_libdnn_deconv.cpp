#ifdef USE_LIBDNN

#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/libdnn_deconv_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// Comparative check difference limit
#define kappa 0.05
// Comparative check shape size limit
#define ELEMENT_LIMIT 1000000

namespace caffe {

template <typename Dtype>
class LibDNNDeconvolutionLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  LibDNNDeconvolutionLayerTest()
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

  virtual ~LibDNNDeconvolutionLayerTest() {
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

TYPED_TEST_CASE(LibDNNDeconvolutionLayerTest, TestDtypesFloat);

TYPED_TEST(LibDNNDeconvolutionLayerTest, TestSetup) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<TypeParam, TypeParam, TypeParam> > layer(
      new LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam>(layer_param));
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
  layer.reset(new LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam>(layer_param));
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

TYPED_TEST(LibDNNDeconvolutionLayerTest, TestSimpleDeconvolution) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
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
  shared_ptr<Layer<TypeParam, TypeParam, TypeParam> > layer(
      new LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // constant-fill the bottom blobs
  FillerParameter filler_param;
  filler_param.set_value(1.);
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  filler.Fill(this->blob_bottom_2_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // simply check that accumulation works with overlapping filters
  const TypeParam* top_data = this->blob_top_->cpu_data();
  for (int_tp n = 0; n < this->blob_top_->num(); ++n) {
    for (int_tp c = 0; c < this->blob_top_->channels(); ++c) {
      for (int_tp h = 0; h < this->blob_top_->height(); ++h) {
        for (int_tp w = 0; w < this->blob_top_->width(); ++w) {
          TypeParam expected = 3.1;
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
        }
      }
    }
  }
}

TYPED_TEST(LibDNNDeconvolutionLayerTest, TestGradient) {
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
  LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(LibDNNDeconvolutionLayerTest, TestGradient3D) {
  vector<int_tp> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 2;
  bottom_shape[3] = 3;
  bottom_shape[4] = 2;
  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  for (int_tp i = 0; i < this->blob_bottom_vec_.size(); ++i) {
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
  LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

template<typename TypeParam>
class LibDNNComparativeDeconvTest : public GPUDeviceTest<TypeParam> {
 protected:
  LibDNNComparativeDeconvTest()
      : blob_bottom_(new Blob<TypeParam>()),
        blob_bottom_ref_(new Blob<TypeParam>()),
        blob_top_(new Blob<TypeParam>()),
        blob_top_ref_(new Blob<TypeParam>()),
        rng_(rd_()) {
  }

  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_ref_.push_back(blob_bottom_ref_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_ref_.push_back(blob_top_ref_);
  }

  virtual ~LibDNNComparativeDeconvTest() {
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
    std::uniform_int_distribution<int_tp> dilationRand(1, 1);
    std::uniform_int_distribution<int_tp> kernelRand(1, 3);
    std::uniform_int_distribution<int_tp> padRand(0, 2);
    std::uniform_int_distribution<int_tp> strideRand(1, 3);
    std::uniform_int_distribution<int_tp> biasRand(0, 1);
    std::uniform_int_distribution<int_tp> groupRand(1, 4);

    std::uniform_int_distribution<int_tp> batchRand(1, 10);
    std::uniform_int_distribution<int_tp> fmapRand(1, 64);

    int_tp batchsize = batchRand(this->rng_);
    int_tp groups = groupRand(this->rng_);
    int_tp fmaps_in = fmapRand(this->rng_) * groups;
    int_tp fmaps_out = fmapRand(this->rng_) * groups;

    int dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> sizeRand(5,
                std::max(static_cast<int>(pow(ELEMENT_LIMIT /
                  (fmaps_in * fmaps_out * batchsize),
                  1.0 / (static_cast<double>(dims)))), 5));


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

    LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    DeconvolutionLayer<TypeParam, TypeParam, TypeParam> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    for (int_tp i = 0; i < layer.blobs().size(); ++i) {
      caffe_copy(layer.blobs()[i]->count(),
                     layer.blobs()[i]->cpu_data(),
                     ref_layer.blobs()[i]->mutable_cpu_data());
    }

    caffe_rng_uniform(blob_bottom_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_bottom_->mutable_cpu_data());

    caffe_copy(blob_bottom_->count(), blob_bottom_->cpu_data(),
                   blob_bottom_ref_->mutable_cpu_data());

    caffe_set(blob_top_->count(),
              (TypeParam)0.0, blob_top_->mutable_cpu_data());
    caffe_set(blob_top_ref_->count(),
              (TypeParam)0.0, blob_top_ref_->mutable_cpu_data());

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
    std::uniform_int_distribution<int_tp> dilationRand(1, 1);
    std::uniform_int_distribution<int_tp> kernelRand(1, 3);
    std::uniform_int_distribution<int_tp> padRand(0, 2);
    std::uniform_int_distribution<int_tp> strideRand(1, 3);
    std::uniform_int_distribution<int_tp> biasRand(0, 1);
    std::uniform_int_distribution<int_tp> groupRand(1, 4);

    std::uniform_int_distribution<int_tp> batchRand(1, 10);
    std::uniform_int_distribution<int_tp> fmapRand(1, 64);

    int_tp batchsize = batchRand(this->rng_);
    int_tp groups = groupRand(this->rng_);
    int_tp fmaps_in = fmapRand(this->rng_) * groups;
    int_tp fmaps_out = fmapRand(this->rng_) * groups;

    int dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> sizeRand(5,
                std::max(static_cast<int>(pow(ELEMENT_LIMIT /
                  (fmaps_in * fmaps_out * batchsize),
                  1.0 / (static_cast<double>(dims)))), 5));

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

    LibDNNDeconvolutionLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    DeconvolutionLayer<TypeParam, TypeParam, TypeParam> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_ref_, this->blob_top_vec_ref_);

    for (int_tp i = 0; i < layer.blobs().size(); ++i) {
      caffe_copy(layer.blobs()[i]->count(),
                     layer.blobs()[i]->cpu_data(),
                     ref_layer.blobs()[i]->mutable_cpu_data());
    }

    caffe_rng_uniform(blob_top_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_top_->mutable_cpu_diff());

    caffe_copy(blob_top_->count(), blob_top_->cpu_diff(),
                   blob_top_ref_->mutable_cpu_diff());

    caffe_rng_uniform(blob_bottom_->count(), (TypeParam)-5.0, (TypeParam)5.0,
                      blob_bottom_->mutable_cpu_data());

    caffe_copy(blob_bottom_->count(), blob_bottom_->cpu_data(),
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

    vector<bool> prop_down(1, true);

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

TYPED_TEST_CASE(LibDNNComparativeDeconvTest, TestDtypesFloat);

TYPED_TEST(LibDNNComparativeDeconvTest, TestForward) {
  for (int i = 0; i < 100; ++i) {
    if (this->TestForward(i)) {
      break;
    }
  }
}

TYPED_TEST(LibDNNComparativeDeconvTest, TestBackward) {
  for (int i = 0; i < 100; ++i) {
    if (this->TestBackward(i)) {
      break;
    }
  }
}

}  // namespace caffe
#endif  // USE_LIBDNN

