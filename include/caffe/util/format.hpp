// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_FORMAT_H_
#define CAFFE_UTIL_FORMAT_H_

#include <opencv2/opencv.hpp>
#include <string>

#include "caffe/blob.hpp"

#include "caffe/proto/caffe.pb.h"

namespace caffe {

bool OpenCVImageToDatum(
    const cv::Mat& image, const int label, const int height,
    const int width, Datum* datum, const bool is_color = true);

template <typename Dtype>
bool OpenCVImageToBlob(
    const cv::Mat& image, const int label, const int height,
    const int width, Blob<Dtype>* blob, const bool is_color = true);

}  // namespace caffe

#endif   // CAFFE_UTIL_FORMAT_H_
