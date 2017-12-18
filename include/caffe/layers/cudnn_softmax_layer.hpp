#ifndef CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
#define CAFFE_CUDNN_SOFTMAX_LAYER_HPP_

#include <boost/thread/tss.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/**
 * @brief cuDNN implementation of SoftmaxLayer.
 *        Fallback to SoftmaxLayer for CPU mode.
 */
template <typename Dtype>
class CuDNNSoftmaxLayer : public SoftmaxLayer<Dtype> {
 public:
  explicit CuDNNSoftmaxLayer(const LayerParameter& param)
      : SoftmaxLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;
  virtual ~CuDNNSoftmaxLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;

  /*
  mutable ::boost::thread_specific_ptr<cudnnHandle_t> handle_ptr_{
      [](cudnnHandle_t *handle) {
	  cudnnDestroy(*handle);
      }};
      */

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t>
      bottom_desc_ptr_{[](cudnnTensorDescriptor_t *desc) {
          cudnnDestroyTensorDescriptor(*desc);
      }};

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t>
      top_desc_ptr_{[](cudnnTensorDescriptor_t *desc) {
          cudnnDestroyTensorDescriptor(*desc);
      }};
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
