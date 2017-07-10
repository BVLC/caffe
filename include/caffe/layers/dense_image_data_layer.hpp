#ifndef CAFFE_DENSE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_DENSE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <random>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


template <typename Dtype>
class DenseImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DenseImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DenseImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
  Blob<Dtype> transformed_label_;
  std::mt19937 rg_;
  std::uniform_int_distribution<int> rd_;  
};


}  // namespace caffe

#endif  
