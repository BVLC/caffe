// Copyright 2014 BVLC and contributors.
/*
 * Adapted from cxxnet
 */
#ifndef CAFFE_UTIL_DATA_H_
#define CAFFE_UTIL_DATA_H_

#include <string>
#include <vector>
#include "caffe/proto/caffe.pb.h"

#include "caffe/blob.hpp"

namespace caffe {
using std::string;
using std::vector;

template<typename Dtype>
class DataBatch {
 public:
  DataBatch(): data(), label(), batch_size(0) {
  }

  inline void AllocSpace(const int batch_size, const int channels,
                         const int height, const int width,
                         const int max_labels) {
	this->batch_size = batch_size;
	data.Reshape(batch_size, channels, height, width);
	label.Reshape(batch_size, max_labels, 1, 1);
  }

  inline void FreeSpace() {
  }

  inline void CopyFrom(const DataBatch& src) {
    CHECK_EQ(batch_size, src.batch_size);
    data.CopyFrom(src.data, false, true);
    label.CopyFrom(src.label, false, true);
  }

 public:
  size_t batch_size;
  Blob<Dtype> data;
  Blob<Dtype> label;
};

template<typename Dtype>
class DataIterator {
 public:
  DataIterator(const DataIteratorParameter& param):
	  data_iterator_param_(param) {}
  virtual ~DataIterator() {}
  virtual void Init() = 0;
  virtual bool HasNext() = 0;
  virtual void Shuffle();
  virtual const DataBatch<Dtype>& Next() const = 0;
 protected:
  DataIteratorParameter data_iterator_param_;
};

template<typename Dtype>
class HDF5DataIterator: public DataIterator<Dtype> {
public:
 HDF5DataIterator(const DataIteratorParameter& param):
	 DataIterator<Dtype>(param) {}
 virtual ~HDF5DataIterator() {}
 virtual void Init() {}
 virtual bool HasNext();
 virtual void Shuffle();
 virtual const DataBatch<Dtype>& Next() const;
};

template<typename Dtype>
class ImageDataIterator: public DataIterator<Dtype> {
public:
 ImageDataIterator(const DataIteratorParameter& param);
 virtual ~ImageDataIterator() {}
 virtual void Init();
 virtual bool HasNext();
 virtual void Shuffle();
 virtual const DataBatch<Dtype>& Next() const;
private:
 string base_path_;
 vector<std::pair<string, int> > lines_;
};

template<typename Dtype>
class LeveldbDataIterator: public DataIterator<Dtype> {
public:
 LeveldbDataIterator(const DataIteratorParameter& param):
	 DataIterator<Dtype>(param) {}
 virtual ~LeveldbDataIterator() {}
 virtual void Init() {}
 virtual bool HasNext();
 virtual void Shuffle();
 virtual const DataBatch<Dtype>& Next() const;
};

template<typename Dtype>
class MemoryDataIterator: public DataIterator<Dtype> {
public:
 MemoryDataIterator(const DataIteratorParameter& param):
	 DataIterator<Dtype>(param) {}
 virtual ~MemoryDataIterator() {}
 virtual void Init() {}
 virtual bool HasNext();
 virtual void Shuffle();
 virtual const DataBatch<Dtype>& Next() const;
};

template<typename Dtype>
class WindowDataIterator: public DataIterator<Dtype> {
public:
 WindowDataIterator(const DataIteratorParameter& param):
	 DataIterator<Dtype>(param) {}
 virtual ~WindowDataIterator() {}
 virtual void Init() {}
 virtual bool HasNext();
 virtual void Shuffle();
 virtual const DataBatch<Dtype>& Next() const;
};

template <typename Dtype>
DataIterator<Dtype>* GetDataIterator(const DataIteratorParameter& param);

}  // namespace caffe

#endif   // CAFFE_UTIL_DATA_H_
