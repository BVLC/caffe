// Copyright 2014 BVLC.
/*
 Contributors:
 - Yangqing Jia, 2013.
 - kloudkl@github, 2014.
 */

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
CroppingDataProcessor<Dtype>::CroppingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param),
      crop_size_(this->processor_param_.cropping_param().crop_size()) {
}

/*
 * Adapted from the original implementation in the DataLayerPrefetch
 *   authored by Yangqing Jia
 */
template<typename Dtype>
void CroppingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
  if (crop_size_) {
    int height = input->height();
    int width = input->width();
    // We only do random crop when we do training.
    if (Caffe::phase() == Caffe::TRAIN) {
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      height_offset_ = rand() % (height - crop_size_);
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      width_offset_ = rand() % (width - crop_size_);
    } else {
      height_offset_ = (height - crop_size_) / 2;
      width_offset_ = (width - crop_size_) / 2;
    }
    const Dtype* data = input->cpu_data();
    int num = input->num();
    int channels = input->channels();
    output->Reshape(num, channels, crop_size_, crop_size_);
    Dtype* output_data = output->mutable_cpu_data();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size_; ++h) {
          for (int w = 0; w < crop_size_; ++w) {
            output_data[((
                n * channels + c) * crop_size_ + h) * crop_size_ + w] =
                    data[((n * channels + c) * height + h + height_offset_) *
                            width + w + width_offset_];
          }  // for (int w = 0; w < crop_size_; ++w) {
        }  // for (int h = 0; h < crop_size_; ++h) {
      }  // for (int c = 0; c < channels; ++c) {
    }  // for (int n = 0; n < num; ++n) {
  }  // if (crop_size_) {
}

template<typename Dtype>
MirroringDataProcessor<Dtype>::MirroringDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param),
      mirror_(this->processor_param_.mirroring_param().mirror()) {
}

template<typename Dtype>
void MirroringDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
MeanSubtractionDataProcessor<Dtype>::MeanSubtractionDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void MeanSubtractionDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
ScalingDataProcessor<Dtype>::ScalingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void ScalingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
RotationDataProcessor<Dtype>::RotationDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void RotationDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
MeanZeroingDataProcessor<Dtype>::MeanZeroingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void MeanZeroingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
ResizingDataProcessor<Dtype>::ResizingDataProcessor(
    const DataProcessorParameter& param)
    : DataProcessor<Dtype>(param) {
}

template<typename Dtype>
void ResizingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
DataProcessor<Dtype>* GetDataProcessor(const DataProcessorParameter& param) {
  const string& name = param.name();
  const DataProcessorParameter_DataProcessorType& type = param.type();
  switch (type) {
    case DataProcessorParameter_DataProcessorType_CROPPING: {
      return new CroppingDataProcessor<Dtype>(param);
    }
    case DataProcessorParameter_DataProcessorType_MIRRORING: {
      return new MirroringDataProcessor<Dtype>(param);
    }
    case DataProcessorParameter_DataProcessorType_MEAN_SUBTRACTION: {
      return new MeanSubtractionDataProcessor<Dtype>(param);
    }
    case DataProcessorParameter_DataProcessorType_SCALING: {
      return new ScalingDataProcessor<Dtype>(param);
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
