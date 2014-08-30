#include <string>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class IOTest : public ::testing::Test {
 protected:
  IOTest() : resize_height_(256),
    resize_width_(256) {
  }
  virtual ~IOTest() {}

  const int resize_height_;
  const int resize_width_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(IOTest, Dtypes);

#define TestOpenCVImageToDatumChannels(num_channels) \
  do { \
    const int channels = num_channels; \
    const int height = 200; \
    const int width = 300; \
    cv::Mat image(height, width, CV_MAKETYPE(CV_8U, channels)); \
    typedef cv::Vec<uchar, channels> VecUchar; \
    for (int c = 0; c < channels; ++c) { \
      for (int h = 0; h < height; ++h) { \
        for (int w = 0; w < width; ++w) { \
          image.at<VecUchar>(h, w)[c] = ((c * height + h) * width + w) % 256; \
        } \
      } \
    } \
    Datum* datum; \
    int label = 1001; \
    string data; \
    int index; \
    datum = new Datum(); \
    OpenCVImageToDatum(image, label, this->resize_height_, \
                       this->resize_width_, datum); \
    EXPECT_EQ(datum->channels(), channels); \
    EXPECT_EQ(datum->height(), this->resize_height_); \
    EXPECT_EQ(datum->width(), this->resize_width_); \
    EXPECT_EQ(datum->label(), label); \
    delete datum; \
    /* Cases without resizing */ \
    int heights[] = {-1, 0, image.rows, image.rows, image.rows}; \
    int widths[] = {image.cols, image.cols, 0, -1, image.cols}; \
    for (int i = 0; i < 5; ++i) { \
      datum = new Datum(); \
      OpenCVImageToDatum(image, ++label, heights[i], widths[i], datum); \
      EXPECT_EQ(datum->channels(), channels); \
      EXPECT_EQ(datum->height(), image.rows); \
      EXPECT_EQ(datum->width(), image.cols); \
      EXPECT_EQ(datum->label(), label); \
      data = datum->data(); \
      index = 0; \
      for (int c = 0; c < channels; ++c) { \
        for (int h = 0; h < image.rows; ++h) { \
          for (int w = 0; w < image.cols; ++w) { \
            EXPECT_EQ(static_cast<char>(image.at<VecUchar>(h, w)[c]), \
                      data[index++]); \
          } \
        } \
      } \
      delete datum; \
    } \
  } while (0)

TYPED_TEST(IOTest, TestOpenCVImageToDatum) {
  TestOpenCVImageToDatumChannels(3);
  TestOpenCVImageToDatumChannels(1);
}

TYPED_TEST(IOTest, OpenCVMatToBlob) {
  const int channels = 10;
  const int height = 200;
  const int width = 300;
  Blob<TypeParam> blob;
  cv::Mat mat(height, width, CV_MAKETYPE(CV_32F, channels));
  typedef cv::Vec<float, channels> VecFloat;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        mat.at<VecFloat>(h, w)[c] = (c * height + h) * width + w;
      }
    }
  }
  OpenCVMatToBlob<float, TypeParam>(mat, &blob);
  EXPECT_EQ(blob.num(), 1);
  EXPECT_EQ(blob.channels(), mat.channels());
  EXPECT_EQ(blob.height(), mat.rows);
  EXPECT_EQ(blob.width(), mat.cols);
  for (int c = 0; c < mat.channels(); ++c) {
    for (int h = 0; h < mat.rows; ++h) {
      for (int w = 0; w < mat.cols; ++w) {
        EXPECT_EQ(static_cast<TypeParam>(mat.at<VecFloat>(h, w)[c]),
                  blob.cpu_data()[blob.offset(0, c, h, w)]);
      }
    }
  }

  if (sizeof(TypeParam) == sizeof(double)) {
    mat.create(height, width, CV_MAKETYPE(CV_64F, channels));
    typedef cv::Vec<double, channels> VecDouble;
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          mat.at<VecDouble>(h, w)[c] = (c * height + h) * width + w;
        }
      }
    }
    OpenCVMatToBlob<double, TypeParam>(mat, &blob);
    EXPECT_EQ(blob.num(), 1);
    EXPECT_EQ(blob.channels(), mat.channels());
    EXPECT_EQ(blob.height(), mat.rows);
    EXPECT_EQ(blob.width(), mat.cols);
    for (int c = 0; c < mat.channels(); ++c) {
      for (int h = 0; h < mat.rows; ++h) {
        for (int w = 0; w < mat.cols; ++w) {
          EXPECT_EQ(static_cast<TypeParam>(mat.at<VecDouble>(h, w)[c]),
                    blob.cpu_data()[blob.offset(0, c, h, w)]);
        }
      }
    }
  }
}

TYPED_TEST(IOTest, BlobToOpenCVMat) {
  const int channels = 10;
  const int height = 200;
  const int width = 300;
  Blob<TypeParam> blob(1, channels, height, width);
  // fill the values
  FillerParameter filler_param;
  GaussianFiller<TypeParam> filler(filler_param);
  filler.Fill(&blob);
  cv::Mat mat;
  BlobToOpenCVMat<TypeParam>(blob, &mat);
  EXPECT_EQ(mat.channels(), channels);
  EXPECT_EQ(mat.rows, height);
  EXPECT_EQ(mat.cols, width);
  const TypeParam* data = blob.cpu_data();
  // opencv multi channel element access
  // http://stackoverflow.com/a/1836580
  typedef cv::Vec<TypeParam, channels> VecType;
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        EXPECT_EQ(data[blob.offset(0, c, h, w)], mat.at<VecType>(h, w)[c]);
      }
    }
  }
}


}  // namespace caffe
