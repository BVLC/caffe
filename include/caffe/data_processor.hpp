// Copyright 2014 BVLC.
/*
 Contributors:
 - Yangqing Jia, 2013.
 - kloudkl@github, 2014.
 */

#ifndef CAFFE_DATA_PROCESSORS_HPP_
#define CAFFE_DATA_PROCESSORS_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::string;
using std::vector;

#define BLOB_DATUM_DIMS_LOOP_BEGIN(channels, height, width) \
      for (int c = 0; c < channels; ++c) { \
        for (int h = 0; h < height; ++h) { \
          for (int w = 0; w < width; ++w) {

#define BLOB_DATUM_DIMS_LOOP_END \
          }  /* for (int w = 0; w < crop_size_; ++w) { */ \
        }  /* for (int h = 0; h < crop_size_; ++h) { */ \
      }  /* for (int c = 0; c < channels; ++c) { */

#define BLOB_ALL_DIMS_LOOP_BEGIN(num, channels, height, width) \
    for (int n = 0; n < num; ++n) { \
      BLOB_DATUM_DIMS_LOOP_BEGIN(channels, height, width)

#define BLOB_ALL_DIMS_LOOP_END \
      BLOB_DATUM_DIMS_LOOP_END \
    }  /* for (int n = 0; n < num; ++n) { */

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

  inline uint32_t crop_size() const { return crop_size_; }
  inline uint32_t height_offset() const { return height_offset_; }
  inline uint32_t width_offset() const { return width_offset_; }
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

  inline MirroringParameter::MirroringType type() const { return type_; }
  inline float random_sampling_ratio() const { return random_sampling_ratio_; }
  inline vector<bool> is_mirrored() const { return is_mirrored_; }
 protected:
  FillerParameter filler_param_;
  UniformFiller<Dtype> filler_;
  Blob<Dtype>* random_one_to_zero_;
  MirroringParameter::MirroringType type_;
  float random_sampling_ratio_;
  vector<bool> is_mirrored_;
};

template<typename Dtype>
class MeanSubtractionDataProcessor : public DataProcessor<Dtype> {
 public:
  explicit MeanSubtractionDataProcessor(const DataProcessorParameter& param);
  virtual ~MeanSubtractionDataProcessor() {
  }

  virtual void Process(const shared_ptr<Blob<Dtype> >& input,
                       shared_ptr<Blob<Dtype> > output);
  inline string mean_file() const { return mean_file_; }
  inline const Dtype* mean_blob_data() const { return mean_blob_->cpu_data(); }
  inline void set_mean_blob(const shared_ptr<Blob<Dtype> > blob) { mean_blob_ = blob; }
 protected:
  string mean_file_;
  shared_ptr<Blob<Dtype> > mean_blob_;
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
