#ifndef CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
#define CAFFE_CUDNN_SOFTMAX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"

#ifdef USE_CUDNN  // cuDNN acceleration library.
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {

#ifdef USE_CUDNN
/**
 * @brief cuDNN implementation of SoftmaxLayer.
 *        Fallback to SoftmaxLayer for CPU mode.
 */
template<typename Dtype, typename MItype, typename MOtype>
class CuDNNSoftmaxLayer : public SoftmaxLayer<Dtype, MItype, MOtype> {
 public:
  explicit CuDNNSoftmaxLayer(const LayerParameter& param)
      : SoftmaxLayer<Dtype, MItype, MOtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual ~CuDNNSoftmaxLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_SOFTMAX_LAYER_HPP_
