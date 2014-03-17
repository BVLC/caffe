// Copyright 2014 kloudkl@github

#ifndef CAFFE_FORMAT_IO_H_
#define CAFFE_FORMAT_IO_H_

#include <H5Cpp.h>

#include "caffe/blob.hpp"

namespace caffe {
using namespace H5;
using std::string;
using std::vector;

template<typename Dtype>
int BlobToHDF5(const Blob<Dtype>& blob, const DataType hdf5_data_type,
               const DataSpace& hdf5_data_space, DataSet* data_set) {
  try {
    Exception::dontPrint();
    const Dtype* data = blob.cpu_data();
    data_set->write(data, hdf5_data_type, hdf5_data_space, hdf5_data_space);
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

template<typename Dtype>
int HDF5ToBlob(const DataSet& data_set, const PredType hdf5_data_type,
               Blob<Dtype>* blob) {
  LOG(ERROR) << "HDF5ToBlob";
  try {
    Exception::dontPrint();
    LOG(ERROR) << "HDF5ToBlob";
    DataSpace dataspace = data_set.getSpace();
    hsize_t dims_out[4];
    LOG(ERROR) << "HDF5ToBlob";
    int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    LOG(ERROR) << "blob->Reshape ";
    LOG(ERROR) << dims_out[0] << dims_out[1]  << dims_out[2] << dims_out[3];
    blob->Reshape(dims_out[0], dims_out[1], dims_out[2], dims_out[3]);
    Dtype* data_out = blob->mutable_cpu_data();
    data_set.read(data_out, hdf5_data_type, dataspace, dataspace);
//    data_set.read(data_out, hdf5_data_type);
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
    LOG(ERROR) << "HDF5ToBlob Unknown error ";
    return 1;
  }
  return 0;
}

}  // namespace caffe

#endif   // CAFFE_FORMAT_IO_H_
