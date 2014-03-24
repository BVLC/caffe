// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::pair;

namespace caffe {

template <typename Dtype>
void* ImagesLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  ImagesLayer<Dtype>* layer =
      reinterpret_cast<ImagesLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  const int cropsize = layer->layer_param_.cropsize();
  const bool mirror = layer->layer_param_.mirror();
  const int new_height  = layer->layer_param_.new_height();
  const int new_width  = layer->layer_param_.new_height();

  if (mirror && cropsize == 0) {
    LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const int lines_size = layer->lines_.size();
  const Dtype* mean = layer->data_mean_.cpu_data();
  for (int itemid = 0; itemid < batchsize; ++itemid) {
    // get a blob
    CHECK_GT(lines_size, layer->lines_id_);
    if (!ReadImageToDatum(layer->lines_[layer->lines_id_].first,
          layer->lines_[layer->lines_id_].second,
          new_height, new_width, &datum)) {
      continue;
    }
    const string& data = datum.data();
    if (cropsize) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (Caffe::phase() == Caffe::TRAIN) {
        // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
        h_off = rand() % (height - cropsize);
        // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
        w_off = rand() % (width - cropsize);
      } else {
        h_off = (height - cropsize) / 2;
        w_off = (width - cropsize) / 2;
      }
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
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
      // Just copy the whole data
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
    layer->lines_id_++;
    if (layer->lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->lines_id_ = 0;
      if (layer->layer_param_.shuffle_images()) {
        std::random_shuffle(layer->lines_.begin(), layer->lines_.end());
      }
    }
  }

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
ImagesLayer<Dtype>::~ImagesLayer<Dtype>() {
  // Finally, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void ImagesLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Input Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Input Layer takes two blobs as output.";
  const int new_height = this->layer_param_.new_height();
  const int new_width = this->layer_param_.new_height();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) <<
  "Current implementation requires new_height and new_width to be set"
  "at the same time.";
  // Read the file with filenames and labels
  LOG(INFO) << "Opening file " << this->layer_param_.source();
  std::ifstream infile(this->layer_param_.source().c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.shuffle_images()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines_.begin(), lines_.end());
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.rand_skip()) {
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    unsigned int skip = rand() % this->layer_param_.rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enought points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                         new_height, new_width, &datum));
  // image
  int cropsize = this->layer_param_.cropsize();
  if (cropsize > 0) {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.batchsize(), datum.channels(), cropsize, cropsize));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum.channels(), datum.height(),
        datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.batchsize(), datum.channels(), datum.height(),
        datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), 1, 1, 1));
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, cropsize);
  CHECK_GT(datum_width_, cropsize);
  // check if we want to have mean
  if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, ImagesLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
Dtype ImagesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, ImagesLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImagesLayer);

}  // namespace caffe
