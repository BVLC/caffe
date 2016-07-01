#ifndef HEADER_LOCAL_LAYER
#define HEADER_LOCAL_LAYER

#include <vector>

// #include "caffe/blob.hpp"
// #include "caffe/layer.hpp"
// #include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class LocalLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit LocalLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Local"; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();


  int kernel_size_;
  // int stride_;
  // int num_; // commented because already in base_conv_layer.hpp
  // int channels_; // commented because already in base_conv_layer.hpp
  // int pad_;
  int height_, width_;
  int height_out_, width_out_;
  // int num_output_; // commented because already in base_conv_layer.hpp
  // bool bias_term_;

  int M_;
  int K_;
  int N_;

  // const int dilation_;

  Blob<Dtype> col_buffer_;
};

}  // namespace caffe

#endif
