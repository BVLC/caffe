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
template <typename Dtype> class MemoryDataLayer : public BaseDataLayer<Dtype> {
public:
  explicit MemoryDataLayer(const LayerParameter &param)
      : BaseDataLayer<Dtype>(param) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "MemoryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

#ifdef USE_OPENCV
  virtual std::shared_ptr<Blob<Dtype>> FromMatVector(const vector<cv::Mat> &mat_vector);
#endif // USE_OPENCV

  int channels() { return channels_; }
  int height() { return height_; }
  int width() { return width_; }

protected:
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) {}
  void Forward_const_gpu(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top) const override {}
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) {}
  void Forward_const_cpu(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top) const override {}

  int channels_, height_, width_, size_;
};

} // namespace caffe

#endif // CAFFE_MEMORY_DATA_LAYER_HPP_
