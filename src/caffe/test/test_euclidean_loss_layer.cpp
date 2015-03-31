#include <cmath>
#include <cstdlib>
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
class EuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~EuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    EuclideanLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    EuclideanLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void EuclideanLossLayerTestSetup(int num_images, int num_channels, int im_width, int im_height) {

	  blob_bottom_data_->Reshape(num_images, num_channels, im_height, im_width);
	  blob_bottom_label_->Reshape(num_images, num_channels, im_height, im_width);

	  FillerParameter filler_param;
	  UniformFiller<Dtype> filler(filler_param);
	  filler.Fill(this->blob_bottom_data_);
	  filler.Fill(this->blob_bottom_label_);

	  blob_bottom_vec_.clear();
	  blob_bottom_vec_.push_back(blob_bottom_data_);
	  blob_bottom_vec_.push_back(blob_bottom_label_);
  }

  void EuclideanLossLayerTestForwardPerformance(int num_images, int num_channels, int im_width, int im_height) {

	  this->EuclideanLossLayerTestSetup(num_images, num_channels, im_width, im_height);

	  LayerParameter layer_param;
	  EuclideanLossLayer<Dtype> layer(layer_param);
	  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

#if defined(USE_CUDA) || defined(USE_OPENCL)
			blob_bottom_data_->mutable_gpu_data();
			blob_bottom_data_->mutable_gpu_diff();
			blob_bottom_label_->mutable_gpu_data();
			blob_bottom_label_->mutable_gpu_diff();
			blob_top_loss_->mutable_gpu_data();
			blob_top_loss_->mutable_gpu_diff();
#endif

	  record r;
	  r.type 			= std::string(typeid(Dtype).name());
	  r.num_images 		= num_images;
	  r.num_channels 	= num_channels;
	  r.img_width		= im_width;
	  r.img_height		= im_height;

	  BENCH(r, {
			  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	  });
  }
};

TYPED_TEST_CASE(EuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(EuclideanLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(EuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  EuclideanLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EuclideanLossLayerTest, TestForwardPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=TEST_IMAGE_WIDTH_MAX; i*=2 ) {
		this->EuclideanLossLayerTestForwardPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

}  // namespace caffe
