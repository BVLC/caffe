// Copyright 2014 BVLC and contributors.

#include <string>

#include "caffe/data/data.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::string;

template <typename Dtype>
DataIterator<DataBatch<Dtype> >* GetDataIterator(const DataIteratorParameter& param) {
  const string& name = param.name();
  const DataIteratorParameter_DataIteratorType& type = param.type();
  switch (type) {
    case DataIteratorParameter_DataIteratorType_HDF5:
      return new HDF5DataIterator<DataBatch<Dtype> >(param);
  case DataIteratorParameter_DataIteratorType_IMAGE:
    return new ImageDataIterator<DataBatch<Dtype> >(param);
  case DataIteratorParameter_DataIteratorType_LEVELDB:
    return new LeveldbDataIterator<DataBatch<Dtype> >(param);
  case DataIteratorParameter_DataIteratorType_MEMORY:
    return new MemoryDataIterator<DataBatch<Dtype> >(param);
  case DataIteratorParameter_DataIteratorType_WINDOW:
    return new WindowDataIterator<DataBatch<Dtype> >(param);
  default:
    LOG(FATAL) << "DataIterator " << name << " has unknown type " << type;
  }
  return (DataIterator<DataBatch<Dtype> >*)(NULL);
}

template
DataIterator<DataBatch<float> >* GetDataIterator(
    const DataIteratorParameter& param);
template
DataIterator<DataBatch<double> >* GetDataIterator(
    const DataIteratorParameter& param);


}  // namespace caffe
