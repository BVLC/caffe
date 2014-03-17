// Copyright 2014 kloudkl@github

#include <H5Cpp.h>
#include <opencv2/opencv.hpp>

#include "caffe/util/format.hpp"

namespace caffe {

template <>
int BlobToHDF5<float>(const Blob<float>& blob, const DataType hdf5_data_type,
               const DataSpace& hdf5_data_space, DataSet* data_set);
template <>
int BlobToHDF5<double>(const Blob<double>& blob, const DataType hdf5_data_type,
               const DataSpace& hdf5_data_space, DataSet* data_set);

template<>
int HDF5ToBlob<float>(const DataSet& data_set, const PredType hdf5_data_type,
Blob<float>* blob);
template<>
int HDF5ToBlob<double>(const DataSet& data_set, const PredType hdf5_data_type,
                       Blob<double>* blob);

}  // namespace caffe
