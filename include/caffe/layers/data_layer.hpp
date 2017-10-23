#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class DataLayer
    : public BasePrefetchingDataLayer<Dtype, MItype, MOtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual inline const char* type() const { return "Data"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 0; }
  virtual inline int_tp MinTopBlobs() const { return 1; }
  virtual inline int_tp MaxTopBlobs() const { return 2; }

 protected:
  void Next();
  bool Skip();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
