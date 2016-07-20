#ifndef CAFFE_CYCLIC_ROLL_LAYER_HPP_
#define CAFFE_CYCLIC_ROLL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// cyclic rolling increase the number of features by 4 folds
// and reallign each of the rotational batch to the feature space
// the feature map has to be equal in height and width
// because this usually operate on the datalayer, so no backprop is done
// stack the rotated features from the bottom batch in the same group to
// the top channel dimension in a continous manner,
// rotated featrures are continuous in the top channel dimension

template <typename Dtype>
class CyclicRollLayer : public Layer<Dtype> {
 public:
  explicit CyclicRollLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "CyclicRoll"; }
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
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe
#endif  // CAFFE_CYCLIC_ROLL_LAYER_HPP_
