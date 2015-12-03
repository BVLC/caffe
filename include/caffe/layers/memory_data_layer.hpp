#ifndef CAFFE_MEMORY_DATA_LAYER_HPP_
#define CAFFE_MEMORY_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from memory.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MemoryDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit MemoryDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MemoryData"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 0; }
  virtual inline int_tp ExactNumTopBlobs() const { return 2; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
#ifdef USE_OPENCV
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector,
      const vector<int_tp>& labels);
#endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, Dtype* label, int_tp n);
  void set_batch_size(int_tp new_size);

  vector<int_tp> shape() { return shape_; }
  vector<int_tp> label_shape() { return label_shape_; }
  int_tp batch_size() { return shape_[0]; }
  int_tp channels() { return shape_[1]; }
  int_tp height() { return shape_[2]; }
  int_tp width() { return shape_[3]; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  vector<int_tp> shape_;
  vector<int_tp> label_shape_;
  int_tp size_;

  Dtype* data_;
  Dtype* labels_;
  int_tp n_;
  uint_tp pos_;
  Blob<Dtype> added_data_;
  Blob<Dtype> added_label_;
  bool has_new_data_;
};

}  // namespace caffe

#endif  // CAFFE_MEMORY_DATA_LAYER_HPP_
