// Copyright 2014 kloudkl@github

#ifndef CAFFE_DATA_PROCESSORS_HPP_
#define CAFFE_DATA_PROCESSORS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::vector;

template<typename Dtype>
class DataProcessor {
 public:
  DataProcessor(const DataProcessorParameter& param)
      : param_(param) {
  }
  virtual ~DataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs) = 0;
 protected:
  const DataProcessorParameter& param_;
};

template<typename Dtype>
class MeanSubtractionDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MeanSubtractionDataProcessor(const DataProcessorParameter& param);
  virtual ~MeanSubtractionDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

template<typename Dtype>
class ScalingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit ScalingDataProcessor(const DataProcessorParameter& param);
  virtual ~ScalingDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

template<typename Dtype>
class MirroringDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MirroringDataProcessor(const DataProcessorParameter& param);
  virtual ~MirroringDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

template<typename Dtype>
class RotationDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit RotationDataProcessor(const DataProcessorParameter& param);
  virtual ~RotationDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

template<typename Dtype>
class MeanZeroingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MeanZeroingDataProcessor(const DataProcessorParameter& param);
  virtual ~MeanZeroingDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

template<typename Dtype>
class ResizingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit ResizingDataProcessor(const DataProcessorParameter& param);
  virtual ~ResizingDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

template<typename Dtype>
class CroppingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit CroppingDataProcessor(const DataProcessorParameter& param);
  virtual ~CroppingDataProcessor() {
  }

  virtual void Process(vector<Blob<Dtype>*>& blobs);
};

// The data sink factory function
template<typename Dtype>
DataProcessor<Dtype>* GetDataProcessor(const DataProcessorParameter& param);

}  // namespace caffe

#endif /* CAFFE_DATA_PROCESSORS_HPP_ */
