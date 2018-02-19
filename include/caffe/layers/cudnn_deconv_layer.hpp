#ifndef CAFFE_CUDNN_DECONV_LAYER_HPP_
#define CAFFE_CUDNN_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/deconv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
 * @brief cuDNN implementation of DeConvolutionLayer.
 *        Fallback to DeConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates deconvolution through forward kernels for filtering and
 * bias plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
*/
template <typename Dtype>
class CuDNNDeconvolutionLayer : public DeconvolutionLayer<Dtype> {
 public:
  explicit CuDNNDeconvolutionLayer(const LayerParameter& param)
    : DeconvolutionLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNDeconvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_DECONV_LAYER_HPP_
