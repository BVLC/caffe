#ifndef CAFFE_LABELSEQUENCE_ACCURACY_LAYER_HPP_
#define CAFFE_LABELSEQUENCE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for
 *        label sequence learning task with ctc loss
 */
template <typename Dtype>
class LabelsequenceAccuracyLayer : public Layer<Dtype> {
 public:
  explicit LabelsequenceAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LabelsequenceAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void GetLabelseqs(const vector<int>& label_seq_with_blank,
      vector<int>& label_seq);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { CAFFE_NOT_IMPLEMENTED; }
    }
  }

  int blank_label_;
  vector<int> pred_label_seq_;   // prediction
  vector<int> gt_label_seq_;     // ground truth
};

}  // namespace caffe

#endif  // CAFFE_LABELSEQUENCE_ACCURACY_LAYER_HPP_
