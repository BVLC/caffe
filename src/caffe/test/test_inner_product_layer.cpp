#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#ifdef USE_CUDA
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class InnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~InnerProductLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void InnerProductLayerTestSetup(int num_images, int num_channels, int im_width, int im_height) {

	  blob_bottom_->Reshape(num_images, num_channels, im_height, im_width);

	  FillerParameter filler_param;
	  UniformFiller<Dtype> filler(filler_param);
	  filler.Fill(this->blob_bottom_);

	  blob_bottom_vec_.clear();
	  blob_bottom_vec_.push_back(blob_bottom_);
  }

  void InnerProductLayerTestForwardPerformance(int num_images, int num_channels, int im_width, int im_height) {

	  this->InnerProductLayerTestSetup(num_images, num_channels, im_width, im_height);

	  LayerParameter layer_param;
	  InnerProductParameter* inner_product_param = layer_param.mutable_inner_product_param();
	  inner_product_param->set_num_output(10);
	  inner_product_param->mutable_weight_filler()->set_type("uniform");
	  inner_product_param->mutable_bias_filler()->set_type("uniform");
	  inner_product_param->mutable_bias_filler()->set_min(1);
	  inner_product_param->mutable_bias_filler()->set_max(2);
	  shared_ptr<InnerProductLayer<Dtype> > layer(new InnerProductLayer<Dtype>(layer_param));

	  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

#if defined(USE_CUDA) || defined(USE_OPENCL)
			blob_bottom_->mutable_gpu_data();
			blob_bottom_->mutable_gpu_diff();
			blob_top_->mutable_gpu_data();
			blob_top_->mutable_gpu_diff();
#endif

	  record r;
	  r.type 			= std::string(typeid(Dtype).name());
	  r.num_images 		= num_images;
	  r.num_channels 	= num_channels;
	  r.img_width		= im_width;
	  r.img_height		= im_height;

	  BENCH(r, {
			  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	  });
  }

};

TYPED_TEST_CASE(InnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
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

TYPED_TEST(InnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifdef USE_CUDA
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
#ifdef USE_OPENCL
  IS_VALID_CUDA = true;
#endif
  if (Caffe::mode() == Caffe::CPU || IS_VALID_CUDA) {
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

#if defined(USE_CUDA) || defined(USE_OPENCL)
TYPED_TEST(InnerProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#if defined(USE_CUDA)
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
#ifdef USE_OPENCL
  IS_VALID_CUDA = true;
#endif
  if (Caffe::mode() == Caffe::CPU || IS_VALID_CUDA) {
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
#endif

TYPED_TEST(InnerProductLayerTest, TestForwardPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=1024; i*=2 ) {
		this->InnerProductLayerTestForwardPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

}  // namespace caffe
