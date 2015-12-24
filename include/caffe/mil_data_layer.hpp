#ifndef MIL_DATA_LAYER_HPP_
#define MIL_DATA_LAYER_HPP_

#include "hdf5.h"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {
/**
 * @brief Uses a text file which specifies the image names, and a hdf5 file for the labels.
 *        Note that each image can have multiple positive labels.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MILDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MILDataLayer(const LayerParameter& param);
  virtual ~MILDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;

  virtual const char* type() const override;

  virtual int ExactNumBottomBlobs() const override;

  virtual inline int ExactNumTopBlobs() const override;

protected:
  virtual void load_batch(Batch<Dtype>* batch) override;
  virtual unsigned int PrefetchRand();
  int num_images_;
  unsigned int counter_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, std::string>> image_database_;
  hid_t label_file_id_;
  
  vector<float> mean_value_;
  Blob<Dtype> label_blob_;
};

}  // namespace caffe

#endif  MIL_DATA_LAYER_HPP_
