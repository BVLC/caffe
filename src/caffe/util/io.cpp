// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
    ::google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(536870912, 268435456);

  CHECK(proto->ParseFromCodedStream(coded_input));

  delete coded_input;
  delete raw_input;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
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

template <>
void hd5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim,
        boost::scoped_ptr<float>* array, std::vector<hsize_t>& out_dims) {
    herr_t status;

    int ndims;
    status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
    CHECK_GE(ndims, min_dim);
    CHECK_LE(ndims, max_dim);

    boost::scoped_ptr<hsize_t> dims(new hsize_t[ndims]);

    H5T_class_t class_;
    status = H5LTget_dataset_info(
        file_id, dataset_name_, dims.get(), &class_, NULL);
    CHECK_EQ(class_, H5T_FLOAT) << "Epected float data";

    int array_size = 1;
    for (int i=0; i<ndims; ++i) {
      out_dims.push_back(dims.get()[i]);
      array_size *= dims.get()[i];
    }

    array->reset(new float[array_size]);
    status = H5LTread_dataset_float(
        file_id, dataset_name_, array->get());
}

template <>
void hd5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim,
        boost::scoped_ptr<double>* array, std::vector<hsize_t>& out_dims) {
    herr_t status;

    int ndims;
    status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
    CHECK_GE(ndims, min_dim);
    CHECK_LE(ndims, max_dim);

    boost::scoped_ptr<hsize_t> dims(new hsize_t[ndims]);

    H5T_class_t class_;
    status = H5LTget_dataset_info(
        file_id, dataset_name_, dims.get(), &class_, NULL);
    CHECK_EQ(class_, H5T_FLOAT) << "Epected float data";

    int array_size = 1;
    for (int i=0; i<ndims; ++i) {
      out_dims.push_back(dims.get()[i]);
      array_size *= dims.get()[i];
    }

    array->reset(new double[array_size]);
    status = H5LTread_dataset_double(
        file_id, dataset_name_, array->get());
}

}  // namespace caffe
