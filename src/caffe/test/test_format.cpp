// Copyright 2014 BVLC and contributors

#include <cuda_runtime.h>

#include <string>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/format.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
using std::string;

template <typename Dtype>
class FormatTest : public ::testing::Test {
 protected:
  FormatTest() : image_file_path_("src/caffe/test/test_data/lena.png") {
  }
  virtual ~FormatTest() {}
  string image_file_path_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(FormatTest, Dtypes);

TYPED_TEST(FormatTest, TestOpenCVImageToDatum) {
  cv::Mat cv_img = cv::imread(this->image_file_path_, CV_LOAD_IMAGE_COLOR);
  Datum* datum;
  int label = 1001;
  string data;
  int index;
  datum = new Datum();
  OpenCVImageToDatum(cv_img, label, 128, 256, datum);
  EXPECT_EQ(datum->channels(), 3);
  EXPECT_EQ(datum->height(), 128);
  EXPECT_EQ(datum->width(), 256);
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

}  // namespace caffe
