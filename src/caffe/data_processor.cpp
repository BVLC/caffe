// Copyright 2014 kloudkl@github

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_processor.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::string;
using std::vector;

template<typename Dtype>
MeanSubtractionDataProcessor<Dtype>::MeanSubtractionDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void MeanSubtractionDataProcessor<Dtype>::Process(
    vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
ScalingDataProcessor<Dtype>::ScalingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void ScalingDataProcessor<Dtype>::Process(vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
MirroringDataProcessor<Dtype>::MirroringDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void MirroringDataProcessor<Dtype>::Process(vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
RotationDataProcessor<Dtype>::RotationDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void RotationDataProcessor<Dtype>::Process(vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
MeanZeroingDataProcessor<Dtype>::MeanZeroingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void MeanZeroingDataProcessor<Dtype>::Process(vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
ResizingDataProcessor<Dtype>::ResizingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void ResizingDataProcessor<Dtype>::Process(vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
CroppingDataProcessor<Dtype>::CroppingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void CroppingDataProcessor<Dtype>::Process(vector<Blob<Dtype>*>& blobs) {
}

template<typename Dtype>
DataProcessor<Dtype>* GetDataProcessor(const DataProcessorParameter& param) {
  const string& name = param.name();
  const DataProcessorParameter_DataProcessorType& type = param.type();
  switch (type) {
  case DataProcessorParameter_DataProcessorType_MEAN_SUBTRACTION: {
    return new MeanSubtractionDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_SCALING: {
    return new ScalingDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_MIRRORING: {
    return new MirroringDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_ROTATION: {
    return new RotationDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_MEAN_ZEROING: {
    return new MeanZeroingDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_RESIZING: {
    return new ResizingDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_CROPPING: {
    return new CroppingDataProcessor<Dtype>(param);
  }
  case DataProcessorParameter_DataProcessorType_NONE: {
    LOG(FATAL)<< "Data processor " << name << " has unspecified type.";
    break;
  }
  default: {
    LOG(FATAL)<< "Unknown data processor type: " << type;
    break;
  }
  }
  return (DataProcessor<Dtype>*) (NULL);
}

template
DataProcessor<float>* GetDataProcessor(const DataProcessorParameter& param);
template
DataProcessor<double>* GetDataProcessor(const DataProcessorParameter& param);

INSTANTIATE_CLASS(MeanSubtractionDataProcessor);
INSTANTIATE_CLASS(ScalingDataProcessor);
INSTANTIATE_CLASS(MirroringDataProcessor);
INSTANTIATE_CLASS(RotationDataProcessor);
INSTANTIATE_CLASS(MeanZeroingDataProcessor);
INSTANTIATE_CLASS(ResizingDataProcessor);
INSTANTIATE_CLASS(CroppingDataProcessor);

}  // namespace caffe
