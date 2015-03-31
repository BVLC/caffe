#include <algorithm>
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

template <typename TypeParam>
class EltwiseLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EltwiseLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_c_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    filler.Fill(this->blob_bottom_c_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_c_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EltwiseLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_bottom_c_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void EltwiseLayerTestSetup(int num_images, int num_channels, int im_width, int im_height) {

	  blob_bottom_a_->Reshape(num_images, num_channels, im_height, im_width);
	  blob_bottom_b_->Reshape(num_images, num_channels, im_height, im_width);
	  blob_bottom_c_->Reshape(num_images, num_channels, im_height, im_width);

	  FillerParameter filler_param;
	  UniformFiller<Dtype> filler(filler_param);
	  filler.Fill(this->blob_bottom_a_);
	  filler.Fill(this->blob_bottom_b_);
	  filler.Fill(this->blob_bottom_c_);

	  blob_bottom_vec_.clear();
	  blob_bottom_vec_.push_back(blob_bottom_a_);
	  blob_bottom_vec_.push_back(blob_bottom_b_);
	  blob_bottom_vec_.push_back(blob_bottom_c_);
  }

  void EltwiseLayerTestSumPerformance(int num_images, int num_channels, int im_width, int im_height) {

	  this->EltwiseLayerTestSetup(num_images, num_channels, im_width, im_height);

	  LayerParameter layer_param;
	  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
	  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
	  shared_ptr<EltwiseLayer<Dtype> > layer(new EltwiseLayer<Dtype>(layer_param));
	  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

	  record r;
	  r.type 			= std::string(typeid(Dtype).name());
	  r.num_images 	= num_images;
	  r.num_channels 	= num_channels;
	  r.img_width		= im_width;
	  r.img_height	= im_height;

#if defined(USE_CUDA) || defined(USE_OPENCL)
			blob_bottom_a_->mutable_gpu_data();
			blob_bottom_a_->mutable_gpu_diff();
			blob_bottom_b_->mutable_gpu_data();
			blob_bottom_b_->mutable_gpu_diff();
			blob_bottom_c_->mutable_gpu_data();
			blob_bottom_c_->mutable_gpu_diff();
			blob_top_->mutable_gpu_data();
			blob_top_->mutable_gpu_diff();
#endif

	  BENCH(r, {
			  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	  });
  }

  void EltwiseLayerTestProdPerformance(int num_images, int num_channels, int im_width, int im_height) {

	  this->EltwiseLayerTestSetup(num_images, num_channels, im_width, im_height);

	  LayerParameter layer_param;
	  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
	  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
	  shared_ptr<EltwiseLayer<Dtype> > layer(new EltwiseLayer<Dtype>(layer_param));
	  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

	  record r;
	  r.type 			= std::string(typeid(Dtype).name());
	  r.num_images 	= num_images;
	  r.num_channels 	= num_channels;
	  r.img_width		= im_width;
	  r.img_height	= im_height;

	  BENCH(r, {
			  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	  });
  }

  void EltwiseLayerTestMaxPerformance(int num_images, int num_channels, int im_width, int im_height) {

	  this->EltwiseLayerTestSetup(num_images, num_channels, im_width, im_height);

	  LayerParameter layer_param;
	  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
	  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_MAX);
	  shared_ptr<EltwiseLayer<Dtype> > layer(new EltwiseLayer<Dtype>(layer_param));
	  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

	  record r;
	  r.type 			= std::string(typeid(Dtype).name());
	  r.num_images 	= num_images;
	  r.num_channels 	= num_channels;
	  r.img_width		= im_width;
	  r.img_height	= im_height;

	  BENCH(r, {
			  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	  });
  }

};

TYPED_TEST_CASE(EltwiseLayerTest, TestDtypesAndDevices);

TYPED_TEST(EltwiseLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  shared_ptr<EltwiseLayer<Dtype> > layer(
      new EltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(EltwiseLayerTest, TestProd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  shared_ptr<EltwiseLayer<Dtype> > layer(
      new EltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data_a[i] * in_data_b[i] * in_data_c[i]);
  }
}

TYPED_TEST(EltwiseLayerTest, TestSum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  shared_ptr<EltwiseLayer<Dtype> > layer(
      new EltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i], in_data_a[i] + in_data_b[i] + in_data_c[i]);
  }
}

TYPED_TEST(EltwiseLayerTest, TestSumCoeff) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  shared_ptr<EltwiseLayer<Dtype> > layer(
      new EltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], in_data_a[i] - 0.5*in_data_b[i] + 2*in_data_c[i],
        1e-4);
  }
}

TYPED_TEST(EltwiseLayerTest, TestStableProdGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  eltwise_param->set_stable_prod_grad(true);
  EltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EltwiseLayerTest, TestUnstableProdGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
  eltwise_param->set_stable_prod_grad(false);
  EltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EltwiseLayerTest, TestSumGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  EltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EltwiseLayerTest, TestSumCoeffGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);
  eltwise_param->add_coeff(1);
  eltwise_param->add_coeff(-0.5);
  eltwise_param->add_coeff(2);
  EltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EltwiseLayerTest, TestMax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_MAX);
  shared_ptr<EltwiseLayer<Dtype> > layer(
      new EltwiseLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  const Dtype* in_data_c = this->blob_bottom_c_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(data[i],
              std::max(in_data_a[i], std::max(in_data_b[i], in_data_c[i])));
  }
}

TYPED_TEST(EltwiseLayerTest, TestMaxGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
  eltwise_param->set_operation(EltwiseParameter_EltwiseOp_MAX);
  EltwiseLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EltwiseLayerTest, TestSumPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=TEST_IMAGE_WIDTH_MAX*4; i*=2 ) {
		this->EltwiseLayerTestSumPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

TYPED_TEST(EltwiseLayerTest, TestProdPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=TEST_IMAGE_WIDTH_MAX*4; i*=2 ) {
		this->EltwiseLayerTestProdPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

TYPED_TEST(EltwiseLayerTest, TestMaxPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=TEST_IMAGE_WIDTH_MAX*4; i*=2 ) {
		this->EltwiseLayerTestMaxPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

}  // namespace caffe
