// Copyright 2014 BVLC and contributors.

#include <string>

#include "caffe/data/data.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::string;

template<typename Dtype>
DataIterator<Dtype>* GetDataIterator(const DataIteratorParameter& param) {
	const string& name = param.name();
	const DataIteratorParameter_DataIteratorType& type = param.type();
	switch (type) {
	case DataIteratorParameter_DataIteratorType_HDF5:
		return new HDF5DataIterator<Dtype>(param);
	case DataIteratorParameter_DataIteratorType_IMAGE:
		return new ImageDataIterator<Dtype>(param);
	case DataIteratorParameter_DataIteratorType_LEVELDB:
		return new LeveldbDataIterator<Dtype>(param);
	case DataIteratorParameter_DataIteratorType_MEMORY:
		return new MemoryDataIterator<Dtype>(param);
	case DataIteratorParameter_DataIteratorType_WINDOW:
		return new WindowDataIterator<Dtype>(param);
	default:
		LOG(FATAL) << "DataIterator " << name << " has unknown type " << type;
	}
	return (DataIterator<Dtype>*) (NULL);
}

template
DataIterator<float>* GetDataIterator(const DataIteratorParameter& param);
template
DataIterator<double>* GetDataIterator(const DataIteratorParameter& param);

}  // namespace caffe
