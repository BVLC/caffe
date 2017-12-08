#ifndef CAFFE_INPUT_LAYER_HPP_
#define CAFFE_INPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net by assigning tops directly.
 *
 * This data layer is a container that merely holds the data assigned to it;
 * forward, backward, and reshape are all no-ops.
 */
template <typename Dtype>
class InputLayer : public Layer<Dtype> {
 public:
  explicit InputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override {}

  virtual inline const char* type() const { return "Input"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

  const std::vector<int> &heights() { return heights_; }
  const std::vector<int> &widths() { return widths_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Forward_const_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override {}
  virtual void Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override {}
 private:
  std::vector<int> heights_;
  std::vector<int> widths_;

};

}  // namespace caffe

#endif  // CAFFE_INPUT_LAYER_HPP_
