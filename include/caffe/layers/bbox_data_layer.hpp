#ifndef CAFFE_BBOX_DATA_LAYER_HPP
#define CAFFE_BBOX_DATA_LAYER_HPP

#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data and bounding box details to the Net from image files and txt files
 */
typedef struct {
    int xmin, xmax;
    int ymin, ymax;
    int class_idx;
} single_object;

template <typename Dtype>
class BboxDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BboxDataLayer(const LayerParameter &param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~BboxDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BboxData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
  void infer_bbox_shape(const string& filename,
  const std::vector<single_object>& bbox_);
};

}  // namespace caffe

#endif  // CAFFE_BBOX_DATA_LAYER_HPP
