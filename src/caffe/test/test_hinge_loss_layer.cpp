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
class HingeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HingeLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~HingeLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

	void HingeLossLayerTestSetup(int num_images, int num_channels, int im_width, int im_height) {

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

	void HingeLossLayerTestForwardPerformance(int num_images, int num_channels, int im_width, int im_height) {

		this->HingeLossLayerTestSetup(num_images, num_channels, im_width, im_height);

		LayerParameter layer_param;
		HingeLossLayer<Dtype> layer(layer_param);
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
		r.type = std::string(typeid(Dtype).name());
		r.num_images = num_images;
		r.num_channels = num_channels;
		r.img_width = im_width;
		r.img_height = im_height;

	BENCH(r, {
				layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
			});
	}

};

TYPED_TEST_CASE(HingeLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(HingeLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701, 1, 0.01);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HingeLossLayerTest, TestGradientL2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // Set norm to L2
  HingeLossParameter* hinge_loss_param = layer_param.mutable_hinge_loss_param();
  hinge_loss_param->set_norm(HingeLossParameter_Norm_L2);
  HingeLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(HingeLossLayerTest, TestForwardPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=1024; i*=2 ) {
		this->HingeLossLayerTestForwardPerformance(TEST_NUM_IMAGES, TEST_NUM_CHANNELS, i, i);
	}
}

}  // namespace caffe
