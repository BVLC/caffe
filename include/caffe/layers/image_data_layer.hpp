#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

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
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param),
        ignore_label_(param.image_data_param().ignore_label()) {}
  virtual ~ImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

  // For the multi-label case, NUM_LABEL_LISTS of labels are maintained.
  // (by default, these are separated by ';')
  static const int NUM_LABEL_LISTS = 2;
  static const Dtype USE_LABEL = 1;

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, vector<vector<int> > > > lines_;
  int lines_id_;

  int num_labels_per_line_;

  // The label assigned to labels which should be ignored.
  Dtype ignore_label_;

  // For mulit-label problems, each of the labels in a particular list are
  // assigned a particular value.
  Dtype LABEL_VALUES[NUM_LABEL_LISTS];
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
