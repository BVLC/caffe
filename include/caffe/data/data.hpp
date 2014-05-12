// Copyright 2014 BVLC and contributors.
/*
 * Adapted from cxxnet
 */
#ifndef CAFFE_UTIL_DATA_H_
#define CAFFE_UTIL_DATA_H_

#include <vector>
#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

namespace caffe {
using std::vector;

template<typename DataBatchType>
class DataIterator {
 public:
  DataIterator(const DataIteratorParameter& param) {}
  virtual ~DataIterator() {}
  virtual void Init() = 0;
  virtual void BeforeFirst() = 0;
  virtual bool Next() = 0;
  virtual const DataBatchType& Value() const = 0;
};

template<typename Dtype>
class DataInstance {
 public:
  float label;
  uint32_t index;
  Blob<Dtype> data;
};

template<typename Dtype>
class DataBatch {
 public:
  DataBatch(): labels(), indices(), batch_size(0) {
  }

  inline void AllocSpace(const int batch_size, const int channels,
                         const int height, const int width) {
    data.Reshape(batch_size, channels, height, width);
    labels.resize(batch_size);
    indices.resize(batch_size);
    this->batch_size = batch_size;
  }

  inline void FreeSpace() {
  }

  inline void CopyFrom(const DataBatch& src) {
    CHECK_EQ(batch_size, src.batch_size);
    labels = src.labels;
    indices = src.indices;
    data.CopyFrom(src.data, false, true);
  }

 public:
 vector<float> labels;
 vector<uint32_t> indices;
 size_t batch_size;
 Blob<Dtype> data;
};

template<typename DataBatchType>
class HDF5DataIterator: public DataIterator<DataBatchType> {
public:
 HDF5DataIterator(const DataIteratorParameter& param):
	 DataIterator<DataBatchType>(param) {}
 virtual ~HDF5DataIterator() {}
 virtual void Init() {}
 virtual void BeforeFirst() {}
 virtual bool Next() {}
 virtual const DataBatchType& Value() const {}
};

template<typename DataBatchType>
class ImageDataIterator: public DataIterator<DataBatchType> {
public:
 ImageDataIterator(const DataIteratorParameter& param):
	 DataIterator<DataBatchType>(param) {}
 virtual ~ImageDataIterator() {}
 virtual void Init() {}
 virtual void BeforeFirst() {}
 virtual bool Next() {}
 virtual const DataBatchType& Value() const {}
};

template<typename DataBatchType>
class LeveldbDataIterator: public DataIterator<DataBatchType> {
public:
 LeveldbDataIterator(const DataIteratorParameter& param):
	 DataIterator<DataBatchType>(param) {}
 virtual ~LeveldbDataIterator() {}
 virtual void Init() {}
 virtual void BeforeFirst() {}
 virtual bool Next() {}
 virtual const DataBatchType& Value() const {}
};

template<typename DataBatchType>
class MemoryDataIterator: public DataIterator<DataBatchType> {
public:
 MemoryDataIterator(const DataIteratorParameter& param):
	 DataIterator<DataBatchType>(param) {}
 virtual ~MemoryDataIterator() {}
 virtual void Init() {}
 virtual void BeforeFirst() {}
 virtual bool Next() {}
 virtual const DataBatchType& Value() const {}
};

template<typename DataBatchType>
class WindowDataIterator: public DataIterator<DataBatchType> {
public:
 WindowDataIterator(const DataIteratorParameter& param):
	 DataIterator<DataBatchType>(param) {}
 virtual ~WindowDataIterator() {}
 virtual void Init() {}
 virtual void BeforeFirst() {}
 virtual bool Next() {}
 virtual const DataBatchType& Value() const {}
};

template <typename Dtype>
DataIterator<DataBatch<Dtype> >* GetDataIterator(const DataIteratorParameter& param);

}  // namespace caffe

#endif   // CAFFE_UTIL_DATA_H_
