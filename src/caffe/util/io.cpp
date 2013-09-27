#include <stdint.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using cv::Mat;
using cv::Vec3b;
using std::string;

namespace caffe {

void ReadImageToProto(const string& filename, BlobProto* proto) {
  Mat cv_image;
  cv_image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  CHECK(cv_image.data) << "Could not open or find the image.";
  DCHECK_EQ(cv_image.channels(), 3);
  proto->set_num(1);
  proto->set_channels(3);
  proto->set_height(cv_image.rows);
  proto->set_width(cv_image.cols);
  proto->clear_data();
  proto->clear_diff();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_image.rows; ++h) {
      for (int w = 0; w < cv_image.cols; ++w) {
        proto->add_data(float(cv_image.at<Vec3b>(h, w)[c]) / 255.);
      }
    }
  }
}

void WriteProtoToImage(const string& filename, const BlobProto& proto) {
  CHECK_EQ(proto.num(), 1);
  CHECK_EQ(proto.channels(), 3);
  CHECK_GT(proto.height(), 0);
  CHECK_GT(proto.width(), 0);
  Mat cv_image(proto.height(), proto.width(), CV_8UC3);
  // TODO: copy the blob data to image.
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_image.rows; ++h) {
      for (int w = 0; w < cv_image.cols; ++w) {
        cv_image.at<Vec3b>(h, w)[c] =
            uint8_t(proto.data((c * cv_image.rows + h) * cv_image.cols + w)
                * 255.);
      }
    }
  }
  CHECK(cv::imwrite(filename, cv_image));
}


}  // namespace caffe
