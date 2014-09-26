#include <string>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class IOTest : public ::testing::Test {};

TEST_F(IOTest, TestReadImageToDatum) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestReadImageToDatumContent) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int c = 0; c < datum.channels(); ++c) {
    for (int h = 0; h < datum.height(); ++h) {
      for (int w = 0; w < datum.width(); ++w) {
        EXPECT_TRUE(data[index++] == static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumContentGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(datum.channels(), cv_img.channels());
  EXPECT_EQ(datum.height(), cv_img.rows);
  EXPECT_EQ(datum.width(), cv_img.cols);

  const string& data = datum.data();
  int index = 0;
  for (int h = 0; h < datum.height(); ++h) {
    for (int w = 0; w < datum.width(); ++w) {
      EXPECT_TRUE(data[index++] == static_cast<char>(cv_img.at<uchar>(h, w)));
    }
  }
}

TEST_F(IOTest, TestReadImageToDatumResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 100, 200, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 100);
  EXPECT_EQ(datum.width(), 200);
}


TEST_F(IOTest, TestReadImageToDatumResizedSquare) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  ReadImageToDatum(filename, 0, 256, 256, &datum);
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToDatumGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TEST_F(IOTest, TestReadImageToDatumResizedGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  Datum datum;
  const bool is_color = false;
  ReadImageToDatum(filename, 0, 256, 256, is_color, &datum);
  EXPECT_EQ(datum.channels(), 1);
  EXPECT_EQ(datum.height(), 256);
  EXPECT_EQ(datum.width(), 256);
}

TEST_F(IOTest, TestReadImageToCVMat) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestReadImageToCVMatResized) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 100, 200);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 100);
  EXPECT_EQ(cv_img.cols, 200);
}

TEST_F(IOTest, TestReadImageToCVMatResizedSquare) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256);
  EXPECT_EQ(cv_img.channels(), 3);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

TEST_F(IOTest, TestReadImageToCVMatGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 360);
  EXPECT_EQ(cv_img.cols, 480);
}

TEST_F(IOTest, TestReadImageToCVMatResizedGray) {
  string filename = EXAMPLES_SOURCE_DIR "images/cat.jpg";
  const bool is_color = false;
  cv::Mat cv_img = ReadImageToCVMat(filename, 256, 256, is_color);
  EXPECT_EQ(cv_img.channels(), 1);
  EXPECT_EQ(cv_img.rows, 256);
  EXPECT_EQ(cv_img.cols, 256);
}

}  // namespace caffe
