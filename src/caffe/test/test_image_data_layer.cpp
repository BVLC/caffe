#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << i;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 1;
    reshapefile.close();
  }

  virtual ~ImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

	void ImageDataLayerTestReadPerformance() {

		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		ImageDataParameter* image_data_param = param.mutable_image_data_param();
		image_data_param->set_batch_size(5);
		image_data_param->set_source(this->filename_.c_str());
		image_data_param->set_shuffle(false);
		ImageDataLayer<Dtype> layer(param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

#if defined(USE_CUDA) || defined(USE_OPENCL)
		blob_top_data_->mutable_gpu_data();
		blob_top_data_->mutable_gpu_diff();
		blob_top_label_->mutable_gpu_data();
		blob_top_label_->mutable_gpu_diff();
#endif

		record r;
		r.type = std::string(typeid(Dtype).name());
		r.num_images 	= this->blob_top_data_->num();
		r.num_channels 	= this->blob_top_data_->channels();
		r.img_width 	= this->blob_top_data_->width();
		r.img_height 	= this->blob_top_data_->height();

		BENCH(r, {
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_)
			;
		});
	}

	void ImageDataLayerTestResizePerformance(int scaled_width, int scaled_height) {

		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		ImageDataParameter* image_data_param = param.mutable_image_data_param();
		image_data_param->set_batch_size(5);
		image_data_param->set_source(this->filename_.c_str());
		image_data_param->set_new_height(scaled_width);
		image_data_param->set_new_width(scaled_height);
		image_data_param->set_shuffle(false);
		ImageDataLayer<Dtype> layer(param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		record r;
		r.type = std::string(typeid(Dtype).name());
		r.num_images 	= this->blob_top_data_->num();
		r.num_channels 	= this->blob_top_data_->channels();
		r.img_width 	= this->blob_top_data_->width();
		r.img_height 	= this->blob_top_data_->height();

		BENCH(r, {
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_)
			;
		});
	}

	void ImageDataLayerTestShufflePerformance() {

		typedef typename TypeParam::Dtype Dtype;
		LayerParameter param;
		ImageDataParameter* image_data_param = param.mutable_image_data_param();
		image_data_param->set_batch_size(5);
		image_data_param->set_source(this->filename_.c_str());
		image_data_param->set_shuffle(true);
		ImageDataLayer<Dtype> layer(param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		record r;
		r.type = std::string(typeid(Dtype).name());
		r.num_images 	= this->blob_top_data_->num();
		r.num_channels 	= this->blob_top_data_->channels();
		r.img_width 	= this->blob_top_data_->width();
		r.img_height 	= this->blob_top_data_->height();

		BENCH(r, {
			layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_)
			;
		});
	}

};

TYPED_TEST_CASE(ImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

TYPED_TEST(ImageDataLayerTest, TestReadPerformance) {

	this->ImageDataLayerTestReadPerformance();
}

TYPED_TEST(ImageDataLayerTest, TestResizePerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=TEST_IMAGE_WIDTH_MAX; i*=2 ) {
		this->ImageDataLayerTestResizePerformance(i, i);
	}
}

TYPED_TEST(ImageDataLayerTest, TestShufflePerformance) {

	this->ImageDataLayerTestShufflePerformance();
}

}  // namespace caffe
