#ifndef CAFFE_LIBDNN_CONV_LAYER_HPP_
#define CAFFE_LIBDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

#include "caffe/greentea/libdnn.hpp"

namespace caffe {
#ifdef USE_GREENTEA

template <typename Dtype>
class LibDNNConvolutionLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit LibDNNConvolutionLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~LibDNNConvolutionLayer();

  virtual void Tune(Dtype* top_data, Dtype* top_diff,
                    Dtype* bottom_data, Dtype* bottom_diff,
                    int_tp batch_size);

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


 private:
  shared_ptr<LibDNNConv<Dtype> > libdnn_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_LIBDNN_CONV_LAYER_HPP_
