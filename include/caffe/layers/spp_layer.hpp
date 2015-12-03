#ifndef CAFFE_SPP_LAYER_HPP_
#define CAFFE_SPP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Does spatial pyramid pooling on the input image
 *        by taking the max, average, etc. within regions
 *        so that the result vector of different sized
 *        images are of the same size.
 */
template <typename Dtype>
class SPPLayer : public Layer<Dtype> {
 public:
  explicit SPPLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SPP"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // calculates the kernel and stride dimensions for the pooling layer,
  // returns a correctly configured LayerParameter for a PoolingLayer
  virtual LayerParameter GetPoolingParam(const int_tp pyramid_level,
                                         const int_tp bottom_h,
                                         const int_tp bottom_w,
                                         const SPPParameter spp_param);

  int_tp pyramid_height_;
  int_tp bottom_h_, bottom_w_;
  int_tp num_;
  int_tp channels_;
  int_tp kernel_h_, kernel_w_;
  int_tp pad_h_, pad_w_;
  bool reshaped_first_time_;

  /// the int_tpernal Split layer that feeds the pooling layers
  shared_ptr<SplitLayer<Dtype> > split_layer_;
  /// top vector holder used in call to the underlying SplitLayer::Forward
  vector<Blob<Dtype>*> split_top_vec_;
  /// bottom vector holder used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob<Dtype>*>*> pooling_bottom_vecs_;
  /// the int_tpernal Pooling layers of different kernel sizes
  vector<shared_ptr<PoolingLayer<Dtype> > > pooling_layers_;
  /// top vector holders used in call to the underlying PoolingLayer::Forward
  vector<vector<Blob<Dtype>*>*> pooling_top_vecs_;
  /// pooling_outputs stores the outputs of the PoolingLayers
  vector<Blob<Dtype>*> pooling_outputs_;
  /// the int_tpernal Flatten layers that the Pooling layers feed int_tpo
  vector<FlattenLayer<Dtype>*> flatten_layers_;
  /// top vector holders used in call to the underlying FlattenLayer::Forward
  vector<vector<Blob<Dtype>*>*> flatten_top_vecs_;
  /// flatten_outputs stores the outputs of the FlattenLayers
  vector<Blob<Dtype>*> flatten_outputs_;
  /// bottom vector holder used in call to the underlying ConcatLayer::Forward
  vector<Blob<Dtype>*> concat_bottom_vec_;
  /// the int_tpernal Concat layers that the Flatten layers feed int_tpo
  shared_ptr<ConcatLayer<Dtype> > concat_layer_;
};

}  // namespace caffe

#endif  // CAFFE_SPP_LAYER_HPP_
