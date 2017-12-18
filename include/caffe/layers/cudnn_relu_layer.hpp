#ifndef CAFFE_CUDNN_RELU_LAYER_HPP_
#define CAFFE_CUDNN_RELU_LAYER_HPP_

#include <boost/thread/tss.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/relu_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/**
 * @brief CuDNN acceleration of ReLULayer.
 */
template <typename Dtype> class CuDNNReLULayer : public ReLULayer<Dtype> {
public:
  explicit CuDNNReLULayer(const LayerParameter &param)
      : ReLULayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);
  void Reshape_const(const vector<Blob<Dtype> *> &bottom,
                     const vector<Blob<Dtype> *> &top) const override;
  virtual ~CuDNNReLULayer() = default;

protected:
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  void Forward_const_gpu(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top) const override;

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t>
      bottom_desc_ptr_{[](cudnnTensorDescriptor_t *desc) {
        cudnnDestroyTensorDescriptor(*desc);
      }};

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t> top_desc_ptr_{
      [](cudnnTensorDescriptor_t *desc) {
        cudnnDestroyTensorDescriptor(*desc);
      }};
  mutable ::boost::thread_specific_ptr<cudnnActivationDescriptor_t>
      activ_desc_ptr_{[](cudnnActivationDescriptor_t *desc) {
        cudnnDestroyActivationDescriptor(*desc);
      }};
};
#endif

} // namespace caffe

#endif // CAFFE_CUDNN_RELU_LAYER_HPP_
