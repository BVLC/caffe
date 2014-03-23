// Copyright 2014 BVLC and contributors.

#include <opencv2/opencv.hpp>
#include <string>

#include "caffe/common.hpp"
#include "caffe/util/format.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::string;

bool OpenCVImageToDatum(
    const cv::Mat& image, const int label, const int height,
    const int width, Datum* datum) {
  cv::Mat cv_img;
  CHECK(image.data) << "Image data must not be NULL";
  CHECK_GT(image.rows, 0) << "Image height must be positive";
  CHECK_GT(image.cols, 0) << "Image width must be positive";
  if (height > 0 && width > 0 &&
      image.rows != height && image.cols != width) {
    cv::resize(image, cv_img, cv::Size(width, height));
  } else {
    cv_img = image;
  }
  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  return true;
}

}  // namespace caffe
