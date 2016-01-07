#ifndef CAFFE_KEY_VALUE_DATA_LAYER_HPP_
#define CAFFE_KEY_VALUE_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Provides data to net from a database, data is provided in order
 *        specified in a csv file
 * 
 * The layer is specified by a database to read from, a csv file seperated with
 * the ';' character and the number of the column that should be read from.
 * Each column then contains a list of keys that should be read from the
 * database. The layer then outputs the values of one key after the other.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class KeyValueDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit KeyValueDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param), key_index_(0) {}
  virtual ~KeyValueDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void load_batch(Batch<Dtype>* batch);
  // This can be shared
  virtual inline bool ShareInParallel() const { return true; }
  virtual inline const char* type() const { return "KeyValueData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  vector<string> keys_;                     /// Keys to be fetched
  int key_index_;                           /// Index of the next key to fetch
  shared_ptr<db::DB> db_;                   /// Database to read from
  shared_ptr<db::Cursor> cursor_;           /// Cursor to database
};

}  // namespace caffe

#endif  // CAFFE_KEY_VALUE_DATA_LAYER_HPP_
