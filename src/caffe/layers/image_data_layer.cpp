#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ProcessImageDatum(
    const int channels, const int height, const int width, const int size,
    const int crop_size, const bool mirror, const Dtype* mean,
    const Dtype scale, const Datum datum, const int item_id, Dtype* top_data,
    Dtype* top_label) {
  const string& data = datum.data();
  if (crop_size > 0) {
    CHECK_GT(height, crop_size);
    CHECK_GT(width, crop_size);
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (Caffe::phase() == Caffe::TRAIN) {
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      h_off = caffe_rng_rand() % (height - crop_size);
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      w_off = caffe_rng_rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    if (mirror && caffe_rng_rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data[((item_id * channels + c) * crop_size + h) * crop_size
                + crop_size - 1 - w] = (static_cast<Dtype>((uint8_t) data[(c
                * height + h + h_off) * width + w + w_off])
                - mean[(c * height + h + h_off) * width + w + w_off]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data[((
                item_id * channels + c) * crop_size + h) * crop_size + w] =
                (static_cast<Dtype>((uint8_t) data[(c * height + h + h_off)
                                                   * width + w + w_off])
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
        top_data[item_id * size + j] = (static_cast<Dtype>((uint8_t) data[j])
            - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        top_data[item_id * size + j] = (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }  // if (crop_size > 0) {

  top_label[item_id] = datum.label();
}

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(prefetch_data_.count());
  Dtype* top_data = prefetch_data_.mutable_cpu_data();
  Dtype* top_label = prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const Dtype scale = image_data_param.scale();
  const int batch_size = image_data_param.batch_size();
  const int crop_size = image_data_param.crop_size();
  const bool mirror = image_data_param.mirror();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = datum_channels_;
  const int height = datum_height_;
  const int width = datum_width_;
  const int size = datum_size_;
  const int lines_size = lines_.size();
  const Dtype* mean = data_mean_.cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    if (!ReadImageToDatum(lines_[lines_id_].first,
          lines_[lines_id_].second,
          new_height, new_width, &datum)) {
      continue;
    }
    ProcessImageDatum(channels, height, width, size, crop_size, mirror, mean,
                      scale, datum, item_id, top_data, top_label);

    top_label[item_id] = datum.label();
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
}

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  if (this->layer_param_.image_data_param().has_source()) {
    // Finally, join the thread
    JoinPrefetchThread();
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  is_datum_set_up_ = false;
  top_ = top;
  Layer<Dtype>::SetUp(bottom, top);
  const int new_height  = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  (*top)[1]->Reshape(this->layer_param_.image_data_param().batch_size(),
                     1, 1, 1);
  if (this->layer_param_.image_data_param().has_source()) {
    // Read the file with filenames and labels
    const string& source = this->layer_param_.image_data_param().source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    string filename;
    int label;
    while (infile >> filename >> label) {
      lines_.push_back(std::make_pair(filename, label));
    }
    if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_data_param().rand_skip()) {
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      unsigned int skip = caffe_rng_rand() %
          this->layer_param_.image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(lines_.size(), skip) << "Not enought points to skip";
      lines_id_ = skip;
    }
    // Read a data point, and use it to initialize the top blob.
    Datum datum;
    CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                           new_height, new_width, &datum));
    // image
    const int crop_size = this->layer_param_.image_data_param().crop_size();
    SetUpWithDatum(crop_size, datum, top);
    if (this->layer_param_.image_data_param().has_source()) {
      DLOG(INFO) << "Initializing prefetch";
      CreatePrefetchThread();
      DLOG(INFO) << "Prefetch initialized.";
    }
  }  // if (this->layer_param_.image_data_param().has_source()) {
}

template<typename Dtype>
void ImageDataLayer<Dtype>::SetUpWithDatum(
    const int crop_size, const Datum datum, vector<Blob<Dtype>*>* top) {
  // datum size
  datum_channels_ = datum.channels();
  CHECK_GT(datum_channels_, 0);
  datum_height_ = datum.height();
  CHECK_GT(datum_height_, 0);
  datum_width_ = datum.width();
  CHECK_GT(datum_width_, 0);
  datum_size_ = datum.channels() * datum.height() * datum.width();

  if (crop_size > 0) {
    CHECK_GT(datum_height_, crop_size);
    CHECK_GT(datum_width_, crop_size);
    (*top)[0]->Reshape(this->layer_param_.image_data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    prefetch_data_.Reshape(this->layer_param_.image_data_param().batch_size(),
                           datum.channels(), crop_size, crop_size);
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.image_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    prefetch_data_.Reshape(
        this->layer_param_.image_data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
  }
  prefetch_label_.Reshape(this->layer_param_.image_data_param().batch_size(),
                          1, 1, 1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
  << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
  << (*top)[0]->width();

  // check if we want to have mean
  if (this->layer_param_.image_data_param().has_mean_file()) {
    BlobProto blob_proto;
    string mean_file = this->layer_param_.image_data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
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
  prefetch_data_.mutable_cpu_data();
  prefetch_label_.mutable_cpu_data();
  data_mean_.cpu_data();

  is_datum_set_up_ = true;
}

template <typename Dtype>
void ImageDataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand =
      this->layer_param_.image_data_param().shuffle() ||
      this->layer_param_.image_data_param().crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!StartInternalThread()) << "Pthread execution failed";
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void ImageDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed";
}

template <typename Dtype>
unsigned int ImageDataLayer<Dtype>::PrefetchRand() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::AddImagesAndLabels(const vector<cv::Mat>& images,
                                               const vector<int>& labels) {
  size_t num_images = images.size();
  CHECK_GT(num_images, 0) << "There is no image to add";
  int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_LE(num_images, batch_size)<<
      "The number of added images " << images.size() <<
      " must be no greater than the batch size " << batch_size;
  CHECK_LE(num_images, labels.size()) <<
      "The number of images " << images.size() <<
      " must be no greater than the number of labels " << labels.size();

  const int crop_size = this->layer_param_.image_data_param().crop_size();
  const bool mirror = this->layer_param_.image_data_param().mirror();
  if (mirror && crop_size == 0) {
    LOG(FATAL)<< "Current implementation requires mirror and crop size to be "
        << "set at the same time.";
  }
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width = this->layer_param_.image_data_param().new_height();

  // TODO: create a thread-safe buffer with Intel TBB concurrent container
  //   and process the images in multiple threads with boost::thread
  Datum datum;
  int item_id = 0;
  OpenCVImageToDatum(images[item_id], labels[item_id], new_height, new_width,
                     &datum);
  if (!is_datum_set_up_) {
    SetUpWithDatum(crop_size, datum, top_);
  }
  // datum scales
  const int channels = this->datum_channels_;
  const int height = this->datum_height_;
  const int width = this->datum_width_;
  const int size = this->datum_size_;
  const Dtype* mean = this->data_mean_.cpu_data();
  const Dtype scale = this->layer_param_.image_data_param().scale();
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ProcessImageDatum<Dtype>(channels, height, width, size, crop_size, mirror,
                           mean, scale, datum, item_id, top_data, top_label);
  int image_id;
  for (item_id = 1; item_id < batch_size; ++item_id) {
    image_id = item_id % num_images;
    OpenCVImageToDatum(images[image_id], labels[image_id], new_height,
                       new_width, &datum);
    ProcessImageDatum<Dtype>(channels, height, width, size, crop_size, mirror,
                             mean, scale, datum, item_id, top_data, top_label);
  }
}

template <typename Dtype>
Dtype ImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  if (this->layer_param_.image_data_param().has_source()) {
    // First, join the thread
    JoinPrefetchThread();
  }
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
             (*top)[1]->mutable_cpu_data());
  // Start a new prefetch thread
  if (this->layer_param_.image_data_param().has_source()) {
    CreatePrefetchThread();
  }
  return Dtype(0.);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
