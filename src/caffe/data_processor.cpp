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
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param),
      crop_size_(this->processor_param_.cropping_param().crop_size()) {
}

/*
 * Adapted from the original implementation in DataLayerPrefetch from
 *   src/caffe/layers/data_layers.cpp authored by Yangqing Jia
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
            output_data[((n * channels + c) * crop_size_ + h) * crop_size_ + w] =
                data[((n * channels + c) * height + h + height_offset_) * width
                    + w + width_offset_];
          }  // for (int w = 0; w < crop_size_; ++w) {
        }  // for (int h = 0; h < crop_size_; ++h) {
      }  // for (int c = 0; c < channels; ++c) {
    }  // for (int n = 0; n < num; ++n) {
  }  // if (crop_size_) {
}

template<typename Dtype>
MirroringDataProcessor<Dtype>::MirroringDataProcessor(
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param),
      filler_(filler_param_),
      random_one_to_zero_(new Blob<Dtype>()),
      type_(this->processor_param_.mirroring_param().type()),
      random_sampling_ratio_(
          this->processor_param_.mirroring_param().random_sampling_ratio()) {
  CHECK_GE(random_sampling_ratio_, 0)<<
  "Random sampling ration must be no less than 0";
  CHECK_LE(random_sampling_ratio_, 1) <<
  "Random sampling ration must be no greater than 1";
}

/*
 * Adapted from the original implementation in DataLayerPrefetch from
 *   src/caffe/layers/data_layers.cpp authored by Yangqing Jia
 */
template<typename Dtype>
void MirroringDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
  const Dtype* data = input->cpu_data();
  int num = input->num();
  int channels = input->channels();
  int height = input->height();
  int width = input->width();
  int datum_dim = channels * height * width;
  output->Reshape(num, channels, height, width);
  Dtype* output_data = output->mutable_cpu_data();
  random_one_to_zero_->Reshape(num, 1, 1, 1);
  filler_.Fill(random_one_to_zero_);
  is_mirrored_.resize(num);
  for (int n = 0; n < num; ++n) {
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    if (random_one_to_zero_->data_at(n, 0, 0, 0) <= random_sampling_ratio_) {
      is_mirrored_[n] = true;
      switch (type_) {
      case MirroringParameter::UP_DOWN: {
        BLOB_DATUM_DIMS_LOOP_BEGIN(channels, height, width)
            output_data[((n * channels + c) * height + h) * width + w] =
                data[((n * channels + c) * height + height - 1 - h) * width +
                     w];
        BLOB_DATUM_DIMS_LOOP_END
        break;
      }
      case MirroringParameter::LEFT_RIGHT_AND_UP_DOWN: {
        BLOB_DATUM_DIMS_LOOP_BEGIN(channels, height, width)
            output_data[((n * channels + c) * height + h) * width + w] =
                data[((n * channels + c) * height + height - 1 - h) * width +
                     width - 1 - w];
        BLOB_DATUM_DIMS_LOOP_END
        break;
      }
      case MirroringParameter::LEFT_RIGHT: {
        BLOB_DATUM_DIMS_LOOP_BEGIN(channels, height, width)
            output_data[((n * channels + c) * height + h) * width + w] =
                data[((n * channels + c) * height + h) * width +
                     width - 1 - w];
        BLOB_DATUM_DIMS_LOOP_END
        break;
      }
      case MirroringParameter::NONE:
      default: {
        is_mirrored_[n] = false;
        memcpy(output_data + output->offset(n), data + input->offset(n),
               sizeof(Dtype) * datum_dim);
        break;
      }
      }  // switch (type_) {
    } else {
      is_mirrored_[n] = false;
      memcpy(output_data + output->offset(n), data + input->offset(n),
             sizeof(Dtype) * datum_dim);
    }  // if ((float) rand() / RAND_MAX < random_sampling_ratio_) {
  }  // for (int n = 0; n < num; ++n) {
}

template<typename Dtype>
MeanSubtractionDataProcessor<Dtype>::MeanSubtractionDataProcessor(
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param) {
  if (this->processor_param_.mean_subtraction_param().has_mean_file()) {
    mean_file_ = this->processor_param_.mean_subtraction_param().mean_file();
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << mean_file_;
    ReadProtoFromBinaryFile(mean_file_.c_str(), &blob_proto);
    mean_blob_->FromProto(blob_proto);
  }
}

template<typename Dtype>
void MeanSubtractionDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
  CHECK(mean_blob_) << "Mean blob is not set";
  const Dtype* mean_data = mean_blob_->cpu_data();
  const Dtype* data = input->cpu_data();
  int num = input->num();
  int channels = input->channels();
  int height = input->height();
  int width = input->width();
  output->Reshape(num, channels, height, width);
  Dtype* output_data = output->mutable_cpu_data();
  BLOB_ALL_DIMS_LOOP_BEGIN(num, channels, height, width)
  output_data[((n * channels + c) * height + h) * width + w] =
            data[((n * channels + c) * height + h) * width + w] -
            mean_data[((n * channels + c) * height + h) * width + w];
  BLOB_ALL_DIMS_LOOP_END
}

template<typename Dtype>
ScalingDataProcessor<Dtype>::ScalingDataProcessor(
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param) {
}

template<typename Dtype>
void ScalingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
RotationDataProcessor<Dtype>::RotationDataProcessor(
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param) {
}

template<typename Dtype>
void RotationDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
MeanZeroingDataProcessor<Dtype>::MeanZeroingDataProcessor(
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param) {
}

template<typename Dtype>
void MeanZeroingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
ResizingDataProcessor<Dtype>::ResizingDataProcessor(
    const DataProcessorParameter& processor_param)
    : DataProcessor<Dtype>(processor_param) {
}

template<typename Dtype>
void ResizingDataProcessor<Dtype>::Process(
    const shared_ptr<Blob<Dtype> >& input, shared_ptr<Blob<Dtype> > output) {
}

template<typename Dtype>
DataProcessor<Dtype>* GetDataProcessor(
    const DataProcessorParameter& processor_param) {
  const string& name = processor_param.name();
  const DataProcessorParameter_DataProcessorType& type =
      processor_param.type();
  switch (type) {
    case DataProcessorParameter_DataProcessorType_CROPPING: {
      return new CroppingDataProcessor<Dtype>(processor_param);
    }
    case DataProcessorParameter_DataProcessorType_MIRRORING: {
      return new MirroringDataProcessor<Dtype>(processor_param);
    }
    case DataProcessorParameter_DataProcessorType_MEAN_SUBTRACTION: {
      return new MeanSubtractionDataProcessor<Dtype>(processor_param);
    }
    case DataProcessorParameter_DataProcessorType_SCALING: {
      return new ScalingDataProcessor<Dtype>(processor_param);
    }
    case DataProcessorParameter_DataProcessorType_ROTATION: {
      return new RotationDataProcessor<Dtype>(processor_param);
    }
    case DataProcessorParameter_DataProcessorType_MEAN_ZEROING: {
      return new MeanZeroingDataProcessor<Dtype>(processor_param);
    }
    case DataProcessorParameter_DataProcessorType_RESIZING: {
      return new ResizingDataProcessor<Dtype>(processor_param);
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
DataProcessor<float>* GetDataProcessor(
    const DataProcessorParameter& processor_param);
template
DataProcessor<double>* GetDataProcessor(
    const DataProcessorParameter& processor_param);

INSTANTIATE_CLASS(MeanSubtractionDataProcessor);
INSTANTIATE_CLASS(ScalingDataProcessor);
INSTANTIATE_CLASS(MirroringDataProcessor);
INSTANTIATE_CLASS(RotationDataProcessor);
INSTANTIATE_CLASS(MeanZeroingDataProcessor);
INSTANTIATE_CLASS(ResizingDataProcessor);
INSTANTIATE_CLASS(CroppingDataProcessor);

}  // namespace caffe
