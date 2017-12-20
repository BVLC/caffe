#ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
#define CAFFE_CUDNN_CONV_LAYER_HPP_

#include <boost/thread/tss.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of ConvolutionLayer.
 *        Fallback to ConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates convolution through forward kernels for filtering and bias
 * plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
 *
 * The CUDNN engine does not have memory overhead for matrix buffers. For many
 * input and filter regimes the CUDNN engine is faster than the CAFFE engine,
 * but for fully-convolutional models and large inputs the CAFFE engine can be
 * faster as long as it fits in memory.
 */
template <typename Dtype>
class CuDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
public:
  explicit CuDNNConvolutionLayer(const LayerParameter &param)
      : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);
  virtual void Reshape_const(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top) const override;
  virtual ~CuDNNConvolutionLayer();

protected:
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void
  Forward_const_gpu(const vector<Blob<Dtype> *> &bottom,
                    const vector<Blob<Dtype> *> &top) const override;


  mutable ::boost::thread_specific_ptr<vector<cudnnTensorDescriptor_t>>
      bottom_descs_ptr_{[](vector<cudnnTensorDescriptor_t> *descs) {
        for (int i = 0; i < descs->size(); i++) {
          cudnnDestroyTensorDescriptor((*descs)[i]);
        }
      }};

  mutable ::boost::thread_specific_ptr<vector<cudnnTensorDescriptor_t>>
      top_descs_ptr_{[](vector<cudnnTensorDescriptor_t> *descs) {
        for (int i = 0; i < descs->size(); i++) {
          cudnnDestroyTensorDescriptor((*descs)[i]);
        }
      }};

  mutable ::boost::thread_specific_ptr<cudnnTensorDescriptor_t> bias_desc_ptr_{
      [](cudnnTensorDescriptor_t *desc) {
        cudnnDestroyTensorDescriptor(*desc);
      }};

  mutable ::boost::thread_specific_ptr<cudnnFilterDescriptor_t>
      filter_desc_ptr_{[](cudnnFilterDescriptor_t *desc) {
        cudnnDestroyFilterDescriptor(*desc);
      }};

  mutable ::boost::thread_specific_ptr<vector<cudnnConvolutionDescriptor_t>>
      conv_descs_ptr_{[](vector<cudnnConvolutionDescriptor_t> *descs) {
        for (int i = 0; i < descs->size(); i++) {
          cudnnDestroyConvolutionDescriptor((*descs)[i]);
        }
      }};

  int bias_offset_;

  mutable ::boost::thread_specific_ptr<Blob<int>> workspaceData;
};
#endif

} // namespace caffe

#endif // CAFFE_CUDNN_CONV_LAYER_HPP_
