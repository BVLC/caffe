#ifndef CAFFE_CYCLIC_SLICE_LAYER_HPP_
#define CAFFE_CYCLIC_SLICE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// cyclic slicing increase the number of batches by 4 folds
// and rotate the feature map clockwise 90 degree on each rotation
// the feature map has to be equal in height and width
// because this usually operate on the datalayer, so no backprop is done

template <typename Dtype>
class CyclicSliceLayer : public Layer<Dtype> {
 public:
  explicit CyclicSliceLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "CyclicSlice"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

}  // namespace caffe
#endif  // CAFFE_CYCLIC_SLICE_LAYER_HPP_
