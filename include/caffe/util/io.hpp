// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <string>

#include <google/protobuf/message.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include <H5Cpp.h>
#include <boost/scoped_ptr.hpp>

#include "caffe/util/format.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/blob.hpp"

using std::string;
using ::google::protobuf::Message;

namespace caffe {
using namespace H5;

void ReadProtoFromTextFile(const char* filename,
    Message* proto);
inline void ReadProtoFromTextFile(const string& filename,
    Message* proto) {
  ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

void ReadProtoFromBinaryFile(const char* filename,
    Message* proto);
inline void ReadProtoFromBinaryFile(const string& filename,
    Message* proto) {
  ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

template <typename Dtype>
void hd5_load_nd_dataset(
  hid_t file_id, const char* dataset_name_,
  int min_dim,//inclusive
  int max_dim,//inclusive
  //output:
  boost::scoped_ptr<Dtype>* array,
  std::vector<hsize_t>& dims
  );

template <typename Dtype>
int WriteBlobToHDF5File(
    const Blob<Dtype>& blob, const string& hdf5_file,
    const string& hdf5_dataset_name,  const PredType hdf5_data_type) {
  LOG(ERROR) << "WriteBlobToHDF5File";
  try {
     Exception::dontPrint();
     H5File* file = new H5File(hdf5_file, H5F_ACC_TRUNC);
     int fill_value = 0;
     DSetCreatPropList plist;
     plist.setFillValue(hdf5_data_type, &fill_value);
     int num = blob.num();
     int channels = blob.channels();
     int height = blob.height();
     int width = blob.width();
     const int num_dims = 4;
     hsize_t dims[] = {num, channels, height, width};
     DataSpace data_space(num_dims, dims);
     DataSet* data_set = new DataSet(
         file->createDataSet(hdf5_dataset_name, hdf5_data_type, data_space,
                             plist));
     BlobToHDF5(blob, hdf5_data_type, data_space, data_set);
     file->close();
     delete data_set;
     delete file;
   } catch(FileIException& error) {
     error.printError();
     return -1;
   } catch(DataSetIException& error) {
     error.printError();
     return -2;
   } catch(DataSpaceIException& error) {
     error.printError();
     return -3;
   } catch (Exception& error) {
     error.printError();
     return -4;
   } catch (...) {
     return 1;
   }
   return 0;
}

template <typename Dtype>
int ReadBlobFromHDF5File(const string& hdf5_file,
                          const string& hdf5_dataset_name,
                          Blob<Dtype>* blob) {
  LOG(ERROR) << "ReadBlobFromHDF5File";
  try {
     Exception::dontPrint();
     H5File file(hdf5_file, H5F_ACC_RDONLY);
     DataSet data_set = file.openDataSet(hdf5_dataset_name);
     LOG(ERROR) << "ReadBlobFromHDF5File";
     PredType hdf5_data_type = PredType::NATIVE_FLOAT;
     if (sizeof(Dtype) == sizeof(double)) {
       hdf5_data_type = PredType::NATIVE_DOUBLE;
     }
     HDF5ToBlob(data_set, hdf5_data_type, blob);
     LOG(ERROR) << "ReadBlobFromHDF5File";
     file.close();
   } catch(FileIException& error) {
     error.printError();
     return -1;
   } catch(DataSetIException& error) {
     error.printError();
     return -2;
   } catch(DataSpaceIException& error) {
     error.printError();
     return -3;
   } catch (Exception& error) {
     error.printError();
     return -4;
   } catch (...) {
     return 1;
   }
   return 0;
}

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
