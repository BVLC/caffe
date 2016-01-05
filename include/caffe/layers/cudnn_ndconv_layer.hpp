#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {


template <typename Dtype>
class CudnnNdConvolutionLayer : public Layer<Dtype> {
 public:
  explicit CudnnNdConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CudnnNdConvolutionLayer();

  virtual inline const char* type() const { return "NdConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape();

  vector<int> kernel_shape_;
  vector<int> stride_shape_;
  int num_;
  int channels_;
  vector<int> pad_shape_;
  vector<int> input_shape_;
  int group_;
  int num_output_;
  vector<int> output_shape_;
  bool bias_term_;

  int conv_out_spatial_dim_;
  int kernel_dim_;
  int output_offset_;

  Blob<Dtype> bias_multiplier_;

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;
  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, weight_offset_, bias_offset_;
  size_t workspaceSizeInBytes;
  void *workspace;
};

}  // namespace caffe

