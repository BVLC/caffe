#ifndef CAFFE_SPARSE_DATA_LAYER_HPP_
#define CAFFE_SPARSE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/sparse_blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Provides sparse data to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class SparseDataLayer : public BasePrefetchingSparseDataLayer<Dtype> {
 public:
  explicit SparseDataLayer(const LayerParameter& param);
  virtual ~SparseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			      const vector<Blob<Dtype>*>& top);
  /*void SparseDataLayerSetUp(const vector<SparseBlob<Dtype>*>& bottom,
    const vector<SparseBlob<Dtype>*>& top);*/

  //TODO
  virtual inline bool ShareInParallel() const { return false; }
 
  // Data layers have no bottoms, so reshaping is trivial.
  /*virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}*/

  virtual inline const char* type() const { return "SparseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(SparseBatch<Dtype>* batch);
  /*virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread(); // ?
  virtual void InternalThreadEntry(); // ?*/

  /*shared_ptr<SparseBlob<Dtype> > prefetch_data_; // ?
  shared_ptr<SparseBlob<Dtype> > prefetch_data_copy_; // ?
  shared_ptr<Blob<Dtype> > prefetch_label_; // ?
  shared_ptr<Blob<Dtype> > prefetch_label_copy_; // ?*/
/*shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;*/
  DataReader<SparseDatum> reader_;
};

}

#endif
