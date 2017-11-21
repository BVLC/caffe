#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;
  virtual inline const char* type() const { return "Split"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

};

}  // namespace caffe

#endif  // CAFFE_SPLIT_LAYER_HPP_
