// Copyright 2014 BVLC.
/*
 Contributors:
 - Yangqing Jia, 2013.
 - kloudkl@github, 2014.
 */

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
      : processor_param_(param) {
  }
  virtual ~DataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output) = 0;
 protected:
  const DataProcessorParameter& processor_param_;
};

template<typename Dtype>
class CroppingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit CroppingDataProcessor(const DataProcessorParameter& param);
  virtual ~CroppingDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);

  inline uint32_t crop_size() { return crop_size_; }
  inline uint32_t height_offset() { return height_offset_; }
  inline uint32_t width_offset() { return width_offset_; }
 protected:
  uint32_t crop_size_;
  uint32_t height_offset_;
  uint32_t width_offset_;
};

template<typename Dtype>
class MirroringDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MirroringDataProcessor(const DataProcessorParameter& param);
  virtual ~MirroringDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
 protected:
  bool mirror_;
};

template<typename Dtype>
class MeanSubtractionDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MeanSubtractionDataProcessor(const DataProcessorParameter& param);
  virtual ~MeanSubtractionDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
};

template<typename Dtype>
class ScalingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit ScalingDataProcessor(const DataProcessorParameter& param);
  virtual ~ScalingDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
};

template<typename Dtype>
class RotationDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit RotationDataProcessor(const DataProcessorParameter& param);
  virtual ~RotationDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
};

template<typename Dtype>
class MeanZeroingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MeanZeroingDataProcessor(const DataProcessorParameter& param);
  virtual ~MeanZeroingDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
};

template<typename Dtype>
class ResizingDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit ResizingDataProcessor(const DataProcessorParameter& param);
  virtual ~ResizingDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
};

// The data sink factory function
template<typename Dtype>
DataProcessor<Dtype>* GetDataProcessor(const DataProcessorParameter& param);

}  // namespace caffe

#endif /* CAFFE_DATA_PROCESSORS_HPP_ */
