#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <unistd.h>
#include <string>



#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/msvc.hpp"

#define HDF5_NUM_DIMS 4

namespace caffe {

using ::google::protobuf::Message;



inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
#ifndef _MSC_VER
  *temp_filename = "/tmp/caffe_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());
  int fd = mkstemp(temp_filename_cstr);
  CHECK_GE(fd, 0) << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete[] temp_filename_cstr;
#else
  // has per gtest
  char temp_dir_path[MAX_PATH + 1] = { '\0' };  // NOLINT
  char temp_file_path[MAX_PATH + 1] = { '\0' };  // NOLINT

  ::GetTempPathA(sizeof(temp_dir_path), temp_dir_path);
  const UINT success = ::GetTempFileNameA(temp_dir_path,
      "caffe_test",
      0,  // Generate unique file name.
      temp_file_path);
  CHECK(success != 0) << "Unable to create a temporary file in " << temp_dir_path;
  *temp_filename = temp_dir_path;
#endif
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
#ifndef _MSC_VER
  *temp_dirname = "/tmp/caffe_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);
  CHECK(mkdtemp_result != NULL)
      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete[] temp_dirname_cstr;
#else
  // has per gtest
  char base_temp_dir_path[MAX_PATH + 1] = { '\0' };  // NOLINT
  char temp_dir_path[MAX_PATH + 1] = { '\0' };  // NOLINT

  ::GetTempPathA(sizeof(base_temp_dir_path), base_temp_dir_path);
  // there is not GetTempDirectory in windows so fallback on mktemp
  strcat_s(temp_dir_path, sizeof(temp_dir_path), base_temp_dir_path);
  strcat_s(temp_dir_path, sizeof(temp_dir_path), "\\caffe_test.XXXXXX");
  
  errno_t success = _mktemp_s(temp_dir_path, sizeof(temp_dir_path));
  CHECK(success != 0) << "Unable to create a temporary directory in " << temp_dir_path;

  BOOL directory_created = ::CreateDirectory(temp_dir_path, NULL);
  CHECK(!directory_created) << "Unable to create a temporary directory in " << temp_dir_path;

  *temp_dirname = temp_dir_path;
#endif
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
