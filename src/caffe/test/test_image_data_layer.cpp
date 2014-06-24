#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::map;
using std::string;

namespace caffe {

template <typename TypeParam>
class ImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataLayerTest()
      : seed_(1701),
        filename_(new string(tmpnam(NULL))),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create a Vector of files with labels
    std::ofstream outfile(filename_->c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << *filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << "examples/images/cat.jpg " << i;
      labels_.push_back(i);
    }
    outfile.close();
    image_ = cv::imread("examples/images/cat.jpg", CV_LOAD_IMAGE_COLOR);
  }

  virtual ~ImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  shared_ptr<string> filename_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  cv::Mat image_;
  vector<int> labels_;
};

TYPED_TEST_CASE(ImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  cv::Mat image = this->image_;
  // Go through the data 5 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const TypeParam* data = this->blob_top_data_->cpu_data();
    for (int i = 0, index = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < image.rows; ++h) {
          for (int w = 0; w < image.cols; ++w) {
            EXPECT_EQ(static_cast<uint8_t>(image.at<cv::Vec3b>(h, w)[c]),
                      static_cast<uint8_t>(data[index++]));
          }
        }
      }
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
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
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_->c_str());
  image_data_param->set_shuffle(true);
  ImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  cv::Mat image = this->image_;
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
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
    const TypeParam* data = this->blob_top_data_->cpu_data();
    for (int i = 0, index = 0; i < 5; ++i) {
      EXPECT_GE(this->blob_top_label_->cpu_data()[i], 0);
      EXPECT_LE(this->blob_top_label_->cpu_data()[i], 5);
      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < image.rows; ++h) {
          for (int w = 0; w < image.cols; ++w) {
            EXPECT_EQ(static_cast<uint8_t>(image.at<cv::Vec3b>(h, w)[c]),
                      data[index++]);
          }
        }
      }
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestAddImagesAndLabels) {
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_shuffle(true);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 0);
  EXPECT_EQ(this->blob_top_data_->channels(), 0);
  EXPECT_EQ(this->blob_top_data_->height(), 0);
  EXPECT_EQ(this->blob_top_data_->width(), 0);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  cv::Mat image = this->image_;
  vector<cv::Mat> images(5, image);
  layer.AddImagesAndLabels(images, this->labels_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), this->image_.rows);
  EXPECT_EQ(this->blob_top_data_->width(), this->image_.cols);
  // Go through the data 5 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const TypeParam* data = this->blob_top_data_->cpu_data();
    for (int i = 0, index = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < image.rows; ++h) {
          for (int w = 0; w < image.cols; ++w) {
            EXPECT_EQ(static_cast<uint8_t>(image.at<cv::Vec3b>(h, w)[c]),
                      data[index++]);
          }
        }
      }
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestAddImagesAndLabelsResize) {
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_shuffle(false);
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 0);
  EXPECT_EQ(this->blob_top_data_->channels(), 0);
  EXPECT_EQ(this->blob_top_data_->height(), 0);
  EXPECT_EQ(this->blob_top_data_->width(), 0);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  cv::Mat image = this->image_;
  vector<cv::Mat> images(5, image);
  layer.AddImagesAndLabels(images, this->labels_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), image_data_param->new_height());
  EXPECT_EQ(this->blob_top_data_->width(), image_data_param->new_width());
  // Go through the data 50 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const TypeParam* data = this->blob_top_data_->cpu_data();
    for (int i = 0, index = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(ImageDataLayerTest, TestAddImagesAndLabelsShuffle) {
  LayerParameter param;
  ImageDataParameter* image_data_param = param.mutable_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_shuffle(true);
  ImageDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 0);
  EXPECT_EQ(this->blob_top_data_->channels(), 0);
  EXPECT_EQ(this->blob_top_data_->height(), 0);
  EXPECT_EQ(this->blob_top_data_->width(), 0);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  cv::Mat image = this->image_;
  vector<cv::Mat> images(5, image);
  layer.AddImagesAndLabels(images, this->labels_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), this->image_.rows);
  EXPECT_EQ(this->blob_top_data_->width(), this->image_.cols);
  // Go through the data 5 times
  for (int iter = 0; iter < 5; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    const TypeParam* data = this->blob_top_data_->cpu_data();
    for (int i = 0, index = 0; i < 5; ++i) {
      EXPECT_GE(this->blob_top_label_->cpu_data()[i], 0);
      EXPECT_LE(this->blob_top_label_->cpu_data()[i], 5);
      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < image.rows; ++h) {
          for (int w = 0; w < image.cols; ++w) {
            EXPECT_EQ(static_cast<uint8_t>(image.at<cv::Vec3b>(h, w)[c]),
                      data[index++]);
          }
        }
      }
    }
  }
}

}  // namespace caffe
