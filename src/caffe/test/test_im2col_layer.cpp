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

template <typename TypeParam>
class Im2colLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  Im2colLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~Im2colLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

	void Im2colLayerTestForwardPerformance(int num_images, int num_channels, int im_width, int im_height) {

		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		Im2colLayer<Dtype> layer(layer_param);

		blob_bottom_->Reshape(num_images, num_channels, im_height, im_width);
		blob_bottom_vec_.clear();
		blob_bottom_vec_.push_back(blob_bottom_);

		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

#if defined(USE_CUDA) || defined(USE_OPENCL)
		blob_bottom_->mutable_gpu_data();
		blob_bottom_->mutable_gpu_diff();
		blob_top_->mutable_gpu_data();
		blob_top_->mutable_gpu_diff();
#endif

		record r;
		r.type 			= std::string(typeid(Dtype).name());
		r.num_images 	= num_images;
		r.num_channels 	= num_channels;
		r.img_width		= im_width;
		r.img_height	= im_height;

		BENCH(r, {
				layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		});

	}
};

TYPED_TEST_CASE(Im2colLayerTest, TestDtypesAndDevices);

TYPED_TEST(Im2colLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  Im2colLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 27);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(Im2colLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  Im2colLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We are lazy and will only check the top left block
  for (int c = 0; c < 27; ++c) {
    EXPECT_EQ(this->blob_bottom_->data_at(0, (c / 9), (c / 3) % 3, c % 3),
        this->blob_top_->data_at(0, c, 0, 0));
  }
}

TYPED_TEST(Im2colLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  Im2colLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(Im2colLayerTest, TestRect) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_h(5);
  convolution_param->set_kernel_w(3);
  convolution_param->set_stride(2);
  Im2colLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // We are lazy and will only check the top left block
  for (int c = 0; c < 45; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, (c / 15), (c / 3) % 5, c % 3));
  }
}


TYPED_TEST(Im2colLayerTest, TestRectGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_h(5);
  convolution_param->set_kernel_w(3);
  convolution_param->set_stride(2);
  Im2colLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(Im2colLayerTest, TestForwardPerformance){

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=1024; i*=2 ) {
		this->Im2colLayerTestForwardPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

}  // namespace caffe
