// Copyright 2014 BVLC and contributors.

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/objdetect/rect.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


template <typename Dtype>
void* SpatialPyramidPoolingDataLayerPrefetch(void* layer_pointer) {
  SpatialPyramidPoolingDataLayer<Dtype>* layer =
      reinterpret_cast<SpatialPyramidPoolingDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_diff = layer->prefetch_data_->mutable_cpu_diff();
  // the top diff have to be able to contain the
  // channels, height and width of the bottom blob
  CHECK_GE(layer->prefetch_data_->count() / layer->prefetch_data_->num(), 3);
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const float positive_sampling_ratio =
      layer->layer_param_.spatial_pyramid_pooling_data_param(
          ).positive_sampling_ratio();
  const size_t batch_size = layer->layer_param_.window_data_param().batch_size();

  // zero out batch
  caffe_set(layer->prefetch_data_->count(), Dtype(0), top_data);
  const size_t num_positive_samples = layer->positive_regions_.size();
  const size_t num_negative_samples = layer->negative_regions_.size();
  const size_t num_samples = num_positive_samples + num_negative_samples;
  size_t region_id;
  size_t sample_id;
  int class_label;
  Rect sample_region;
  size_t feature_height;
  size_t feature_width;
  float height_scaling_ratio;
  float width_scaling_ratio;
  for (size_t i = 0; i < batch_size; ++i) {
    // sample a window
    const unsigned int rand_index = layer->PrefetchRand();
    Blob<Dtype> sample_feature;
    if (rand_index % num_samples / static_cast<float>(num_samples) <=
        positive_sampling_ratio) {
      region_id = rand_index % num_positive_samples;
      sample_region = layer->positive_regions_[region_id];
      sample_id = layer->positive_regions_sample_id_[region_id];
      class_label = layer->positive_regions_class_label_[region_id];
    } else {
      region_id = rand_index % num_negative_samples;
      sample_region = layer->negative_regions_[region_id];
      sample_id = layer->negative_regions_sample_id_[region_id];
      class_label = layer->negative_regions_class_label_[region_id];
    }
    feature_height = layer->sample_features_[sample_id]->height();
    feature_width = layer->sample_features_[sample_id]->width();
    height_scaling_ratio = static_cast<float>(feature_height) /
        layer->sample_heights_[sample_id];
    width_scaling_ratio = static_cast<float>(feature_width) /
        layer->sample_widths_[sample_id];
    Rect feature_region(sample_region.x1() * width_scaling_ratio,
                        sample_region.y1() * height_scaling_ratio,
                        sample_region.x2() * width_scaling_ratio,
                        sample_region.y2() * height_scaling_ratio);
    sample_feature.CopyFromRegion(*(layer->sample_features_[sample_id]),
                                  feature_region, false, true);
    top_diff[layer->prefetch_data_->offset(i)] = sample_feature.channels();
    top_diff[layer->prefetch_data_->offset(i, 1)] = sample_feature.height();
    top_diff[layer->prefetch_data_->offset(i, 2)] = sample_feature.width();
    caffe_copy(sample_feature.count(), sample_feature.cpu_data(),
               top_data + layer->prefetch_data_->offset(i));
    top_label[i] = class_label;
  }

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
void SpatialPyramidPoolingDataLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);

  LOG(INFO) << "SpatialPyramidPooling data layer:" << std::endl
      << "  Positive (object) overlap threshold: "
      << this->layer_param_.spatial_pyramid_pooling_data_param(
          ).positive_threshold() << std::endl
      << "  Negative samples (non-object) overlap threshold: "
      << this->layer_param_.spatial_pyramid_pooling_data_param(
          ).negative_threshold() << std::endl
      << "  positive sampling ratio: "
      << this->layer_param_.spatial_pyramid_pooling_data_param(
          ).positive_sampling_ratio();
  const float positive_threshold =
      this->layer_param_.spatial_pyramid_pooling_data_param(
          ).positive_threshold();
  CHECK_GT(positive_threshold, 0);
  CHECK_LE(positive_threshold, 1);
  const float negative_threshold =
      this->layer_param_.spatial_pyramid_pooling_data_param(
          ).negative_threshold();
  CHECK_GT(negative_threshold, 0);
  CHECK_LT(negative_threshold, positive_threshold);

  // regions_file format
  // repeated:
  //    # sample_key
  //    height
  //    width
  //    num_regions
  //    class_index x1 y1 x2 y2
  //    ...  // for num_regions rows
  std::ifstream infile(this->layer_param_.spatial_pyramid_pooling_data_param(
      ).regions_source().c_str());
  CHECK(infile.good()) << "Failed to open regions file "
      << this->layer_param_.spatial_pyramid_pooling_data_param().regions_source()
      << std::endl;

  string hashtag;
  string sample_key;
  if (!(infile >> hashtag >> sample_key)) {
    LOG(FATAL) << "Regions file is empty";
  }
  size_t height;
  size_t width;
  size_t num_regions = 0;
  size_t class_label;
  size_t x1;
  size_t y1;
  size_t x2;
  size_t y2;
  vector<Rect> regions;
  vector<int> class_labels;
  vector<vector<Rect> > all_regions;
  vector<vector<int> > all_class_labels;
  boost::unordered_map<string, size_t> sample_keys;
  do {
    CHECK(sample_keys.find(sample_key) == sample_keys.end());
    sample_keys[sample_key] = sample_keys.size();
    CHECK_EQ(hashtag, "#") << "invalid regions file format";
    infile >> height >> width >> num_regions;
    sample_heights_.push_back(height);
    sample_widths_.push_back(width);
    regions.clear();
    class_labels.clear();
    for (int i = 0; i < num_regions; ++i) {
      infile >> class_label >> x1 >> y1 >> x2 >> y2;
      class_labels.push_back(class_label);
      Rect region(x1, y1, x2, y2);
      regions.push_back(region);
    }
    all_regions.push_back(regions);
    all_class_labels.push_back(class_labels);
  } while (infile >> hashtag >> sample_key);

  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " <<
      this->layer_param_.spatial_pyramid_pooling_data_param(
          ).features_source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.spatial_pyramid_pooling_data_param(
          ).features_source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.spatial_pyramid_pooling_data_param(
          ).features_source()
      << std::endl << status.ToString();
  shared_ptr<leveldb::DB> db(db_temp);
  shared_ptr<leveldb::Iterator> iter(db->NewIterator(leveldb::ReadOptions()));
  iter->SeekToFirst();
  Datum datum;
  size_t size;
  while (iter->Valid()) {
    if (sample_keys.find(iter->key().ToString()) != sample_keys.end()) {
      datum.ParseFromString(iter->value().ToString());
      size = datum.float_data_size();
      shared_ptr<Blob<Dtype> > data(new Blob<Dtype>(
          1, datum.channels(), datum.height(), datum.width()));
      Dtype* ptr = data->mutable_cpu_data();
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = datum.float_data(i);
      }
      sample_features_[sample_keys[iter->key().ToString()]] = data;
    }
  }
  CHECK_GT(sample_keys.size(), 0);
  for (size_t n = 0; n < all_regions.size(); ++n) {
    // Mining the positive samples
    for (size_t i = 0; i < all_regions[n].size(); ++i) {
      if (all_class_labels[n][i] >= 0) {
        positive_regions_.push_back(all_regions[n][i]);
        positive_regions_class_label_.push_back(all_class_labels[n][i]);
        positive_regions_sample_id_.push_back(n);
      }
    }
    if (positive_threshold < 1) {
      for (size_t i = 0; i < all_regions[n].size(); ++i) {
        if (all_class_labels[n][i] > 0) {
          for (size_t j = i + 1; j < all_regions[n].size(); ++j) {
            if (all_class_labels[n][j] > 0) {
              if (all_regions[n][i].intersect(all_regions[n][j]).area() /
                  all_regions[n][i].area() > positive_threshold) {
                positive_regions_.push_back(all_regions[n][j]);
                positive_regions_class_label_.push_back(all_class_labels[n][j]);
                positive_regions_sample_id_.push_back(n);
              }
            }  // if (all_class_labels[n][j] > 0) {
          }  // for (size_t j = i + 1; j < all_regions[n].size(); ++j) {
        }  // if (all_class_labels[n][i] > 0) {
      }  // for (size_t i = 0; i < all_regions[n].size(); ++i) {
    }  // if (positive_threshold < 1) {

    // Mining the negative samples
    for (size_t i = 0; i < all_regions[n].size(); ++i) {
      if (all_class_labels[n][i] > 0) {
        for (size_t j = i + 1; j < all_regions[n].size(); ++j) {
          if (all_class_labels[n][j] < 0) {
            float overlapping_ratio = all_regions[n][i].intersect(
                all_regions[n][j]).area() / all_regions[n][i].area();
            if (overlapping_ratio > 0 &&
                overlapping_ratio < negative_threshold) {
              negative_regions_.push_back(all_regions[n][j]);
              negative_regions_class_label_.push_back(all_class_labels[n][j]);
              negative_regions_sample_id_.push_back(n);
            }
          }  // if (all_class_labels[n][j] < 0) {
        }  // for (size_t j = i + 1; j < all_regions[n].size(); ++j) {
      }  // if (all_class_labels[n][i] > 0) {
    }  // for (size_t i = 0; i < all_regions[n].size(); ++i) {
  }  // for (size_t n = 0; n < all_regions.size(); ++n) {

  LOG(INFO) << "Number of samples: " << sample_features_.size();
  DLOG(INFO) << "Spatial pyramid pooling data loaded.";
}

template <typename Dtype>
void SpatialPyramidPoolingDataLayer<Dtype>::CreatePrefetchThread() {
  // Create Caffe::RNG for the thread
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, SpatialPyramidPoolingDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void SpatialPyramidPoolingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int SpatialPyramidPoolingDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype SpatialPyramidPoolingDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
             (*top)[1]->mutable_cpu_data());
  return Dtype(0.);
}

INSTANTIATE_CLASS(SpatialPyramidPoolingDataLayer);

}  // namespace caffe
