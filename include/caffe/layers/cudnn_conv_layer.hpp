#ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
#define CAFFE_CUDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"
#ifndef CPU_ONLY
#include "caffe/util/gpu_memory.hpp"
#endif

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
  // In iteration 0, use a small amount of memory in order to leave
  // most of memory for allocating layer blobs.
  // NOLINT_NEXT_LINE(build/storage_class)
  const static size_t INITIAL_WORKSPACE_SIZE;
  // Use 95% of available memory.
  // Using all of memory may result in failure of workspace.reserve.
  // NOLINT_NEXT_LINE(build/storage_class)
  const static float MAX_WORKSPACE_RATIO;
  // We update it on second Fwd/Bwd pass and we allocate it *once*
  // when we start third pass. We might recompute it later if demand grows
  // and/or we suddenly need to get extra memory for other needs.
  static size_t& workspace_size(int device);
  static vector<size_t> WORKSPACE_SIZES;
  // This is the workspace used by all Convolution layers one after another.
  // We carry it global to prevent unnecessary allocations/deallocations
  // because they hurt performance.
  static GPUMemory::MultiWorkspace WORKSPACE;

 public:
  explicit CuDNNConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param), handles_setup_(false),
        use_algo_seeker_(true), use_modest_workspace_(true) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNConvolutionLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;

  int bottom_offset_, top_offset_, bias_offset_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;

 private:
  bool use_algo_seeker_;
  bool use_modest_workspace_;
#if CUDNN_VERSION_MIN(5, 0, 0)
  void FindExConvAlgo(const vector<Blob<Dtype>*>& bottom,
                      const vector<Blob<Dtype>*>& top);
#endif
  void GetConvAlgo(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top,
                   const size_t workspace_bytes);

  size_t ComputeFindExWorkspaceSize();

  vector<cudnnTensorDescriptor_t>      cached_bottom_descs_;
  vector<cudnnConvolutionDescriptor_t> cached_conv_descs_;
  bool IsBottomDescChanged(const vector<Blob<Dtype>*>& bottom);
  bool IsConvDescChanged(const vector<Blob<Dtype>*>& bottom);

  bool use_reshape_;
  bool initialized_cached_descs_;

  void UpdateWorkspaceDemand(int size);

  // This is current *demand*: it might be not yet allocated.
};

template<typename Dtype>
vector<size_t> CuDNNConvolutionLayer<Dtype>::WORKSPACE_SIZES;

template<typename Dtype>
const size_t CuDNNConvolutionLayer<Dtype>::INITIAL_WORKSPACE_SIZE =
    4*1024*1024;

template<typename Dtype>
GPUMemory::MultiWorkspace CuDNNConvolutionLayer<Dtype>::WORKSPACE;

template<typename Dtype>
const float CuDNNConvolutionLayer<Dtype>::MAX_WORKSPACE_RATIO = 0.95F;

#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
