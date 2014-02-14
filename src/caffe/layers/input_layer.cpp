// Copyright 2014 Sergio Guadarrama

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/filler.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* InputLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  InputLayer<Dtype>* layer = reinterpret_cast<InputLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  const int cropsize = layer->layer_param_.cropsize();
  const bool mirror = layer->layer_param_.mirror();

  if (mirror && cropsize == 0) {
    LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->bottom_channels_;
  const int height = layer->bottom_height_;
  const int width = layer->bottom_width_;
  const int size = layer->bottom_size_;
  const Dtype* mean = layer->data_mean_.cpu_data();
  for (int itemid = 0; itemid < batchsize; ++itemid) {
    // get a blob
    CHECK(layer->iter_);
    CHECK(layer->iter_->Valid());
    datum.ParseFromString(layer->iter_->value().ToString());
    const string& data = datum.data();
    if (cropsize) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (Caffe::phase() == Caffe::TRAIN) {
        h_off = rand() % (height - cropsize);
        w_off = rand() % (width - cropsize);
      } else {
        h_off = (height - cropsize) / 2;
        w_off = (width - cropsize) / 2;
      }
      if (mirror && rand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < cropsize; ++h) {
            for (int w = 0; w < cropsize; ++w) {
              top_data[((itemid * channels + c) * cropsize + h) * cropsize
                       + cropsize - 1 - w] =
                  (static_cast<Dtype>(
                      (uint8_t)data[(c * height + h + h_off) * width
                                    + w + w_off])
                    - mean[(c * height + h + h_off) * width + w + w_off])
                  * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < cropsize; ++h) {
            for (int w = 0; w < cropsize; ++w) {
              top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
                  = (static_cast<Dtype>(
                      (uint8_t)data[(c * height + h + h_off) * width
                                    + w + w_off])
                     - mean[(c * height + h + h_off) * width + w + w_off])
                  * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          top_data[itemid * size + j] =
              (static_cast<Dtype>((uint8_t)data[j]) - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[itemid * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
    }

    top_label[itemid] = datum.label();
    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst();
    }
  }

  return (void*)NULL;
}

template <typename Dtype>
InputLayer<Dtype>::~InputLayer<Dtype>() {
  // Finally, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void InputLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Input Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Input Layer takes a single blob as output.";
  int cropsize = this->layer_param_.cropsize();
  if (cropsize > 0) {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), bottom.channels(), cropsize, cropsize);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), bottom.channels(), bottom.height(),
        bottom.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  bottom_channels_ = bottom.channels();
  bottom_height_ = bottom.height();
  bottom_width_ = bottom.width();
  bottom_size_ = bottom.channels() * bottom.height() * bottom.width();
  CHECK_GT(bottom_height_, cropsize);
  CHECK_GT(boottom_width_, cropsize);
  // check if we want to have mean
  if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), bottom_channels_);
    CHECK_EQ(data_mean_.height(), bottom_height_);
    CHECK_EQ(data_mean_.width(), boottom_width_);
  } else {
    // Intialize the data_mean with zeros
    data_mean_.Reshape(1, bottom_channels_, bottom_height_, boottom_width_);
    // Or if there is a bias_filler use it to initialize the data_mean
    if (this->layer_param_.has_bias_filler()) {
      shared_ptr<Filler<Dtype> > bias_filler(
        GetFiller<Dtype>(this->layer_param_.bias_filler()));
      bias_filler->Fill(&this->data_mean_);
    }
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void InputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void InputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype InputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype InputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(InputLayer);

}  // namespace caffe
