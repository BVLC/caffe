#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(1073741824, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

cv::Mat AspectResizeToSquare(const cv::Mat &in_img, const int new_size) {
    cv::Mat img_resized, out_img;

    if (in_img.channels() == 3) {  // Color
        out_img = cv::Mat::zeros(new_size, new_size, CV_8UC3);
      } else {  // Grayscale
        out_img = cv::Mat::zeros(new_size, new_size, CV_8U);
      }
    cv::Rect roi;
    float origAspect = static_cast<double> (in_img.cols) /
                       static_cast<double> (in_img.rows);
    if (origAspect > 1) {
        cv::resize(in_img, img_resized,
          cv::Size(new_size, floor(new_size/origAspect)), 0, 0, CV_INTER_AREA);
        cv::Size resSize = img_resized.size();
        int padding = floor((new_size - resSize.height) / 2.0);
        roi = cv::Rect(0, padding, new_size, resSize.height);
    } else {
        cv::resize(in_img, img_resized,
          cv::Size(floor(new_size*origAspect), new_size), 0, 0, CV_INTER_AREA);
        cv::Size resSize = img_resized.size();
        int padding = floor((new_size - resSize.width) / 2.0);
        roi = cv::Rect(padding, 0, resSize.width, new_size);
    }
    cv::Mat roiImg = out_img(roi);
    img_resized.copyTo(roiImg);
    return out_img;
}

bool cvMatToDatum(const cv::Mat & cv_img, const int label, Datum* datum) {
  const unsigned int num_channels = cv_img.channels();
  const unsigned int height = cv_img.rows;
  const unsigned int width = cv_img.cols;

  datum->set_channels(num_channels);
  datum->set_height(height);
  datum->set_width(width);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();

  if (num_channels == 3) {
  for (unsigned int c = 0; c < num_channels; ++c) {
      for (unsigned int h = 0; h < height; ++h) {
          const cv::Vec3b *cv_img_data = cv_img.ptr<cv::Vec3b>(h);
          for (unsigned int w = 0; w < width; ++w) {
              datum_string->push_back(static_cast<char>(cv_img_data[w][c]));
              // Much faster, than at<>;
            }
        }
    }
  } else {
        for (unsigned int h = 0; h < height; ++h) {
            const uchar *cv_img_data = cv_img.ptr<uchar>(h);
            for (unsigned int w = 0; w < width; ++w) {
                datum_string->push_back(static_cast<uchar>(cv_img_data[w]));
                // Much faster, than at<>;
              }
          }
  }
  return true;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    Datum* datum, const bool keep_aspect_ratio) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {
        if (keep_aspect_ratio) {
            cv_img = AspectResizeToSquare(cv_img_origin, height);
          } else {
            cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
          }
  } else {
    cv_img = cv_img_origin;
  }
  return cvMatToDatum(cv_img, label, datum);
}

leveldb::Options GetLevelDBOptions() {
  // In default, we will return the leveldb option and set the max open files
  // in order to avoid using up the operating system's limit.
  leveldb::Options options;
  options.max_open_files = 100;
  return options;
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the dataset exists.
  CHECK(H5LTfind_dataset(file_id, dataset_name_))
      << "Failed to find HDF5 dataset " << dataset_name_;
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  blob->Reshape(
    dims[0],
    (dims.size() > 1) ? dims[1] : 1,
    (dims.size() > 2) ? dims[2] : 1,
    (dims.size() > 3) ? dims[3] : 1);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
