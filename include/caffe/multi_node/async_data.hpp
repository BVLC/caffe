

#ifndef MULTI_NODE_ASYNC_DATA_H
#define MULTI_NODE_ASYNC_DATA_H

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/data_layer.hpp"

#include "caffe/multi_node/async_reader.hpp"

namespace caffe {

template <typename Dtype>
class AsyncDataLayer : public BasePrefetchingDataLayer<Dtype> {

public:
  explicit AsyncDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param) {
      queue_index_ = AsyncReader::Instance()->RegisterLayer(param);
      full_ = AsyncReader::Instance()->GetFull(queue_index_);
      free_ = AsyncReader::Instance()->GetFree(queue_index_);
    }
  
  
  // disable sharing data layer in different solvers
  #ifndef TEST_ACCURACY
  virtual inline bool ShareInParallel() const { return false; }
  #endif

  virtual ~AsyncDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AsyncData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }


protected:
  virtual void load_batch(Batch<Dtype>* batch);

protected:
  int queue_index_; 
  BlockingQueue<Datum*> *full_;
  BlockingQueue<Datum*> *free_;
};


} // end caffe


#endif


