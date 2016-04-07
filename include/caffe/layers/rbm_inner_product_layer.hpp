#ifndef CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Fully connected RBM layer, with both hidden and visible biases
 *
 * The layer takes exactly one bottom blob. During learning (forward_is_update
 * is set to true) the blob just specifies input data for forward pass.
 * During sampling (forward_is_update is set to false) when the backward sample
 * is created the bottom blob's diff gets set to the backward propagated
 * probabilites, and the blob's data gets samples from these probabilites.
 *
 * The top contains then one to three blobs describing the hidden state of the
 * RBM followed by an arbitrary number of blobs with different error values.
 *
 * Note that even though only the _cpu() functions are implemented, all the heavy
 * lifting is happening in the connection_layer_, activation_layer_, and the
 * like. So if these functions have a _gpu() implementation the RBM will run on
 * the GPU.
 */

template <typename Dtype>
class RBMInnerProductLayer : public Layer<Dtype> {
 public:
  explicit RBMInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RBMInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  /**
   * Performing a forward pass either performs an unsupervised update if
   * forward_is_update is set to true in the prototxt or just a forward pass if
   * it is set  to false. During a forward pass the data is first passed through
   * the connection layer and saved to the first top, then through the sqashing
   * layer and saved to second top (if the top is there) and then through the
   * sampling layer and saved to thrird top (if the top is there).
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  /// Layer which is used to fist process the input on forward pass
  shared_ptr<Layer<Dtype> > connection_layer_;
  /// Layer used to squash the hidden units
  shared_ptr<Layer<Dtype> > hidden_activation_layer_;
  /// Layer used to squash the visible units
  shared_ptr<Layer<Dtype> > visible_activation_layer_;
  /// Layer used to sample the hidden activations.
  shared_ptr<Layer<Dtype> > hidden_sampling_layer_;
  /// Layer used to sample the visible activations
  shared_ptr<Layer<Dtype> > visible_sampling_layer_;
};

}  // namespace caffe

#endif  // CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
