#ifndef CAFFE_MERGECROP_LAYER_HPP_
#define CAFFE_MERGECROP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Merges and crops feature maps for U-Net architectures.
 */
template<typename Dtype>
class MergeCropLayer : public Layer<Dtype> {
 public:
  explicit MergeCropLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline int_tp ExactNumBottomBlobs() const {
    return 2;
  }

  virtual inline const char* type() const {
    return "MergeCrop";
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

 private:
  vector<int_tp> forward_;
  vector<int_tp> backward_;
  Blob<int_tp> shape_a_;
  Blob<int_tp> shape_b_;
  MergeCropParameter_MergeOp op_;
};

}  // namespace caffe

#endif  // CAFFE_MERGECROP_LAYER_HPP_
