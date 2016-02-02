#ifndef CAFFE_REMOTE_DATA_LAYER_HPP_
#define CAFFE_REMOTE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

#define HDF5_DATA_DATASET_NAME "data"
#define HDF5_DATA_LABEL_NAME "label"

namespace caffe {
/**
 * @brief Provides data to the Net from remote endpoint.
 *
 * TODO(dox): thorough documentation.
 */
template <typename Dtype>
class RemoteDataLayer : public BaseDataLayer<Dtype> {
  void prepare(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 public:
  explicit RemoteDataLayer(const LayerParameter& param);

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    prepare(bottom, top);
  }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RemoteData"; }

  struct RemoteDataQueue;
 private:
  shared_ptr<RemoteDataQueue> queue;
  Blob<Dtype>* transform_blob;
};

}  // namespace caffe

#endif  // CAFFE_REMOTE_DATA_LAYER_HPP_
