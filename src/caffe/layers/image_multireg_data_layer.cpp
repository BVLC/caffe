#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayerMultiRegression<Dtype>::~ImageDataLayerMultiRegression<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayerMultiRegression<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
#if 0
  string filename;
  int label;
  while (infile >> filename >> label) {
    LOG(INFO) << filename;
    lines_.push_back(std::make_pair(filename, label));
  }
#else
  size_t label_dim = 0;
  for(string line; getline(infile, line);) {
    stringstream line_s(line);
    string filename;

    line_s >> filename;
    vector<float> labels;
    float label;
    while(line_s >> label) labels.push_back(label);
    if(label_dim == 0) label_dim = labels.size();

    lines_.push_back(std::make_pair(filename, labels)); 
  }
#endif

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
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  DatumMultiRegression datum;
  CHECK(ReadImageToDatumMultiRegression(lines_[lines_id_].first, lines_[lines_id_].second,
                                         new_height, new_width, &datum));

  // for(int debug_i = 0; debug_i < label_dim; debug_i++) {
  //   LOG(INFO) << "XXX datum.mutable_label()[debug_i] = " << datum.label(debug_i);
  //   LOG(INFO) << "YYY lines_[lines_id_].second = " << lines_[lines_id_].second[debug_i];
  // }

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  LOG(INFO) << "crop_size = " << crop_size;
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
#if 0
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
#else
  (*top)[1]->Reshape(batch_size, label_dim, 1, 1);
  this->prefetch_label_.Reshape(batch_size, label_dim, 1, 1);
#endif  

  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template <typename Dtype>
void ImageDataLayerMultiRegression<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayerMultiRegression<Dtype>::InternalThreadEntry() {
  DatumMultiRegression datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    if (!ReadImageToDatumMultiRegression(lines_[lines_id_].first,
          lines_[lines_id_].second,
          new_height, new_width, &datum)) {
      continue;
    }

    // Apply transformations (mirror, crop...) to the data
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    //top_label[item_id] = datum.label();
    size_t label_dim = datum.label().size();
    for(int ldim = 0; ldim < label_dim; ldim++) {
      top_label[item_id * label_dim + ldim] = datum.label(ldim);
    }

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

INSTANTIATE_CLASS(ImageDataLayerMultiRegression);

}  // namespace caffe
