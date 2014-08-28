#include <string>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class IOTest : public ::testing::Test {
 protected:
  IOTest() : image_file_path_("examples/images/cat.jpg"), resize_height_(256),
    resize_width_(256) {
  }
  virtual ~IOTest() {}
  string image_file_path_;
  const int resize_height_;
  const int resize_width_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(IOTest, Dtypes);

TYPED_TEST(IOTest, TestOpenCVImageToDatum) {
  cv::Mat cv_img = cv::imread(this->image_file_path_, CV_LOAD_IMAGE_COLOR);
  Datum* datum;
  int label = 1001;
  string data;
  int index;
  datum = new Datum();
  OpenCVImageToDatum(cv_img, label, this->resize_height_, this->resize_width_,
                     datum);
  EXPECT_EQ(datum->channels(), 3);
  EXPECT_EQ(datum->height(), this->resize_height_);
  EXPECT_EQ(datum->width(), this->resize_width_);
  EXPECT_EQ(datum->label(), label);
  delete datum;
  // Cases without resizing
  int heights[] = {-1, 0, cv_img.rows, cv_img.rows, cv_img.rows};
  int widths[] = {cv_img.cols, cv_img.cols, 0, -1, cv_img.cols};
  for (int i = 0; i < 3; ++i) {
    datum = new Datum();
    OpenCVImageToDatum(cv_img, ++label, heights[i], widths[i], datum);
    EXPECT_EQ(datum->channels(), 3);
    EXPECT_EQ(datum->height(), cv_img.rows);
    EXPECT_EQ(datum->width(), cv_img.cols);
    EXPECT_EQ(datum->label(), label);
    data = datum->data();
    index = 0;
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          EXPECT_EQ(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]),
                    data[index++]);
        }
      }
    }
    delete datum;
  }
}

TYPED_TEST(IOTest, OpenCVImageToBlob) {
  cv::Mat cv_img = cv::imread(this->image_file_path_, CV_LOAD_IMAGE_COLOR);
  Blob<TypeParam> blob;
  int label = 1001;
  OpenCVImageToBlob<TypeParam>(cv_img, label, this->resize_height_,
                               this->resize_width_, &blob);
  EXPECT_EQ(blob.num(), 1);
  EXPECT_EQ(blob.channels(), 3);
  EXPECT_EQ(blob.height(), this->resize_height_);
  EXPECT_EQ(blob.width(), this->resize_width_);
  // Cases without resizing
  int heights[] = {-1, 0, cv_img.rows, cv_img.rows, cv_img.rows};
  int widths[] = {cv_img.cols, cv_img.cols, 0, -1, cv_img.cols};
  for (int i = 0; i < 3; ++i) {
    OpenCVImageToBlob<TypeParam>(cv_img, ++label, heights[i], widths[i],
                                 &blob);
    EXPECT_EQ(blob.num(), 1);
    EXPECT_EQ(blob.channels(), 3);
    EXPECT_EQ(blob.height(), cv_img.rows);
    EXPECT_EQ(blob.width(), cv_img.cols);
    const TypeParam* data = blob.cpu_data();
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          EXPECT_EQ(static_cast<TypeParam>(cv_img.at<cv::Vec3b>(h, w)[c]),
                    data[blob.offset(0, c, h, w)]);
        }
      }
    }
  }
}

}  // namespace caffe
