#ifndef CAFFE_RANDVEC_LAYER_HPP_
#define CAFFE_RANDVEC_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
namespace caffe {

template <typename Dtype>
class RandVecLayer : public Layer<Dtype> {
 public:
  explicit RandVecLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "RandVec"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual Dtype GetRandom(const Dtype lower, const Dtype upper);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}


  shared_ptr<Caffe::RNG> data_rng_;
  int batch_size_;
  int dim_;
  Dtype lower_;
  Dtype upper_;
  int iter_idx_;
  int height_;
  int width_;
};

}  // namespace caffe

#endif  // CAFFE_RANDVEC_LAYER_HPP_