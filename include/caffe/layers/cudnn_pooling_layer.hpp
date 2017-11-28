#ifndef CAFFE_CUDNN_POOLING_LAYER_HPP_
#define CAFFE_CUDNN_POOLING_LAYER_HPP_

#include <vector>
#include <boost/thread/tss.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of PoolingLayer.
 *        Fallback to PoolingLayer for CPU mode.
*/
template <typename Dtype>
class CuDNNPoolingLayer : public PoolingLayer<Dtype> {
 public:
  explicit CuDNNPoolingLayer(const LayerParameter& param)
      : PoolingLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;
  virtual ~CuDNNPoolingLayer();
  // Currently, cuDNN does not support the extra top blob.
  virtual inline int MinTopBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  mutable ::boost::thread_specific_ptr<cudnnHandle_t> handle_ptr_{
      [](cudnnHandle_t *handle) {
	  cudnnDestroy(*handle);
      }};

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t>
      bottom_desc_ptr_{[](cudnnTensorDescriptor_t *desc) {
          cudnnDestroyTensorDescriptor(*desc);
      }};

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t>
      top_desc_ptr_{[](cudnnTensorDescriptor_t *desc) {
          cudnnDestroyTensorDescriptor(*desc);
      }};

  mutable ::boost::thread_specific_ptr<cudnnPoolingDescriptor_t>
      pooling_desc_ptr_{[](cudnnPoolingDescriptor_t *desc) {
	  cudnnDestroyPoolingDescriptor(*desc);
      }};

};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_POOLING_LAYER_HPP_
