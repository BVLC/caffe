// Copyright 2014 BVLC and contributors.
/*
 * Adapted from cxxnet
 */
#ifndef CAFFE_UTIL_DATA_H_
#define CAFFE_UTIL_DATA_H_

#include <vector>
#include "mshadow/tensor.h"
#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

namespace caffe {
using std::vector;

template<typename Dtype>
class DataIterator {
 public:
  virtual ~DataIterator() {}
  virtual void Init() = 0;
  virtual void BeforeFirst() = 0;
  virtual bool Next() = 0;
  virtual const Dtype& Value() const = 0;
};

template<typename Dtype>
class DataInstance {
 public:
  float label;
  uint32_t index;
  Blob<Dtype> data;
};

TensorShape

template<typename Dtype>
class DataBatch {
 public:
  DataBatch(): labels(), indices(), batch_size(0) {
  }

  inline void AllocSpace(mshadow::Shape<4> shape, const size_t batch_size) {
    data = Blob::NewBlob(shape);
    labele.resize(batch_size);
    indices.resize(batch_size);
    this->batch_size = batch_size;
  }

  inline void FreeSpace() {
  }

  inline void CopyFrom(const DataBatch& src) {
    CHECK_EQ(batch_size, src.batch_size);
    labels = src.labels;
    indices = src.indices;
    data = src.data;
  }

 public:
 vector<float> labels;
 vector<uint32_t> indices;
 size_t batch_size;
 Blob<Dtype> data;
};

template <typename Dtype>
DataIterator<DataBatch>* GetDataIterator(const DataIteratorParameter& param);

}  // namespace caffe

#endif   // CAFFE_UTIL_DATA_H_
