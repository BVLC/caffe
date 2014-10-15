#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#ifndef OSX
#include <opencv2/core/core.hpp>
#endif

#include <unistd.h>
#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#define HDF5_NUM_DIMS 4

namespace caffe {

using ::google::protobuf::Message;

inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
  *temp_filename = "/tmp/caffe_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());
  int fd = mkstemp(temp_filename_cstr);
  CHECK_GE(fd, 0) << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete[] temp_filename_cstr;
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  *temp_dirname = "/tmp/caffe_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);
  CHECK(mkdtemp_result != NULL)
      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete[] temp_dirname_cstr;
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
    const int height, const int width, const bool is_color, Datum* datum);

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

bool DecodeDatum(const int height, const int width, const bool is_color,
  Datum* datum);

inline bool DecodeDatum(const int height, const int width, Datum* datum) {
  return DecodeDatum(height, width, true, datum);
}

inline bool DecodeDatum(const bool is_color, Datum* datum) {
  return DecodeDatum(0, 0, is_color, datum);
}

inline bool DecodeDatum(Datum* datum) {
  return DecodeDatum(0, 0, true, datum);
}

#ifndef OSX
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

inline cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

inline cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

inline cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

cv::Mat DecodeDatumToCVMat(const Datum& datum,
    const int height, const int width, const bool is_color);

inline cv::Mat DecodeDatumToCVMat(const Datum& datum,
    const int height, const int width) {
  return DecodeDatumToCVMat(datum, height, width, true);
}

inline cv::Mat DecodeDatumToCVMat(const Datum& datum,
    const bool is_color) {
  return DecodeDatumToCVMat(datum, 0, 0, is_color);
}

inline cv::Mat DecodeDatumToCVMat(const Datum& datum) {
  return DecodeDatumToCVMat(datum, 0, 0, true);
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
#endif

/**
 * @brief Shapes a Blob to read "num" rows of HDF5 data.  If num == -1, take
 *        the num of the HDF5 dataset.
 *
 * @param file_id      the HDF5 file handle
 * @param dataset_name the name of the HDF5 dataset to read
 * @param num          the number of rows to read: either num >= 0,
 *                     or num == -1 for the number of rows in the HDF5 dataset
 * @param blob         the Blob to shape
 *
 * The HDF5 dataset must have 1-4 dimensions. blob will be shaped like the
 * the HDF5 dataset, except that the HDF5 dataset's first dimension is ignored
 * and replaced by num, and if the dataset has \@$ D < 4 \@$ dimensions, the
 * remaining \@$ 4 - D \@$ dimensions are replaced with 1's -- so an
 * \@$ N \times D \times H \@$ HDF5 dataset will result in a
 * \@$ \mathrm{num} \times D \times H \times 1 \@$ Blob.
 */
template <typename Dtype>
void HDF5PrepareBlob(hid_t file_id, const char* dataset_name, int num,
                     Blob<Dtype>* blob);

/**
 * @brief Reads rows [offset, offset + data->num() - 1] into Blob* data, which
 *        must have been pre-shaped using HDF5PrepareBlob (or otherwise).
 */
template <typename Dtype>
int HDF5ReadRowsToBlob(hid_t file_id, const char* dataset_name,
                       int h5_offset, int blob_offset, Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_nd_dataset(
  const hid_t file_id, const string dataset_name, const Blob<Dtype>& blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
