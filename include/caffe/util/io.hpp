#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#define HDF5_NUM_DIMS 4

namespace leveldb {
// Forward declaration for leveldb::Options to be used in GetlevelDBOptions().
struct Options;
}

namespace caffe {

using ::google::protobuf::Message;

static inline boost::filesystem::path unique_tmp_path() {
  boost::system::error_code error;
  const boost::filesystem::path tmp_path =
      boost::filesystem::temp_directory_path(error);
  const string pattern =
      (tmp_path / "caffe_test.%%%%-%%%%-%%%%-%%%%").string();
  boost::filesystem::path path;
  do {
    path = boost::filesystem::unique_path(pattern, error);
  } while (boost::system::errc::success != error.value()
      || boost::filesystem::exists(path));
  return path;
}

inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
  *temp_filename = unique_tmp_path().string();
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const boost::filesystem::path path = unique_tmp_path();
  boost::system::error_code error;
  do {
    try {
      if (boost::filesystem::create_directories(path, error)) {
        break;
      }
    } catch (...) {
      LOG(ERROR) << "Failed to create a temporary directory at: "
          << path.string();
    }
  } while (boost::system::errc::success != error.value()
      || !boost::filesystem::exists(path));
  *temp_dirname = path.string();
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
inline void WriteProtoToBinaryFile(const Message& proto,
                                   const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadImageToDatum(const string& filename, const int label, const int height,
                      const int width, const bool is_color, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
                             const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
                             Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

leveldb::Options GetLevelDBOptions();

template<typename Dtype>
void hdf5_load_nd_dataset_helper(hid_t file_id, const char* dataset_name_,
                                 int min_dim, int max_dim, Blob<Dtype>* blob);

template<typename Dtype>
void hdf5_load_nd_dataset(hid_t file_id, const char* dataset_name_, int min_dim,
                          int max_dim, Blob<Dtype>* blob);

template<typename Dtype>
void hdf5_save_nd_dataset(const hid_t file_id, const string dataset_name,
                          const Blob<Dtype>& blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
