#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
MapDataLayer<Dtype>::~MapDataLayer<Dtype>(){
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Initialize DB
  leveldb::DB* db_temp;
  leveldb::Options options = GetLevelDBOptions();
  options.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << this->layer_param_.data_param().source() << std::endl
                     << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first" << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }

  // Read a data point and use it to initialize the top blob.
  BlobProtoVector maps;
  maps.ParseFromString(iter_->value().ToString());
  CHECK(maps.blobs_size() == 2) << "MapDataLayer accepts BlobProtoVector with"
                                << " 2 BlobProtos: data and label.";
  BlobProto dataMap = maps.blobs(0);
  BlobProto labelMap = maps.blobs(1);

  // do not support mirror and crop for the moment
  int crop_size = this->layer_param_.transform_param().crop_size();
  bool mirror = this->layer_param_.transform_param().mirror();
  CHECK(crop_size == 0) << "MapDataLayer does not support cropping.";
  CHECK(!mirror) << "MapDataLayer does not support mirroring";

  // reshape data map
  (*top)[0]->Reshape(
      this->layer_param_.data_param().batch_size(), dataMap.channels(),
      dataMap.height(), dataMap.width());
  this->prefetch_data_.Reshape(this->layer_param_.data_param().batch_size(),
      dataMap.channels(), dataMap.height(), dataMap.width());
  // reshape label map
  (*top)[1]->Reshape(
      this->layer_param_.data_param().batch_size(), labelMap.channels(),
      labelMap.height(), labelMap.width());
  this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
      labelMap.channels(), labelMap.height(), labelMap.width());
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // data map size
  this->datum_channels_ = dataMap.channels();
  this->datum_height_ = dataMap.height();
  this->datum_width_ = dataMap.width();
  this->datum_size_ = dataMap.channels() * dataMap.height() * dataMap.width();
  int label_size = labelMap.channels() * labelMap.height() * labelMap.width();
  label_mean_ = new Dtype[label_size]();
}

Datum BlobProto2Datum(const BlobProto& blob){
  Datum datum;
  datum.set_channels(blob.channels());
  datum.set_height(blob.height());
  datum.set_width(blob.width());
  datum.mutable_float_data()->CopyFrom(blob.data());
  return datum;
}

template<typename Dtype>
void MapDataLayer<Dtype>::InternalThreadEntry() {
  BlobProtoVector maps;
  Datum dataMap, labelMap;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(iter_);
    CHECK(iter_->Valid());
    maps.ParseFromString(iter_->value().ToString());
    // Data transformer only accepts Datum
    dataMap = BlobProto2Datum(maps.blobs(0));
    labelMap = BlobProto2Datum(maps.blobs(1));

    // Apply data and label transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, dataMap, this->mean_, top_data);
    this->data_transformer_.Transform(item_id, labelMap, this->label_mean_, top_label);

    iter_->Next();
    if (!iter_->Valid()) {
      DLOG(INFO) << "Restarting data prefetching from start.";
      iter_->SeekToFirst();
    }
  }
}

INSTANTIATE_CLASS(MapDataLayer);

} // caffe