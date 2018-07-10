#ifdef USE_LIBDNN
#ifndef CAFFE_LIBDNN_CONV_LAYER_HPP_
#define CAFFE_LIBDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/libdnn/libdnn_conv.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class LibDNNConvolutionLayer :
    public BaseConvolutionLayer<Dtype, MItype, MOtype> {
 public:
  explicit LibDNNConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype, MItype, MOtype>(param) {
    this->deconvolution_ = false;
  }
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);
  virtual ~LibDNNConvolutionLayer();

  virtual void Tune(vptr<Dtype> top_data, vptr<Dtype> top_diff,
                    vptr<Dtype> bottom_data, vptr<Dtype> bottom_diff,
                    int_tp batch_size);

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
                           const vector<Blob<MOtype>*>& top) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

 private:
  shared_ptr<LibDNNConv<MItype, MOtype> > libdnn_;
};

}  // namespace caffe

#endif  // CAFFE_LIBDNN_CONV_LAYER_HPP_
#endif  // USE_LIBDNN
