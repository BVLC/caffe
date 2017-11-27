#ifdef USE_LIBDNN
#ifndef CAFFE_LIBDNN_DECONV_LAYER_HPP_
#define CAFFE_LIBDNN_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/libdnn/libdnn.hpp"
#include "caffe/layers/deconv_layer.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class LibDNNDeconvolutionLayer
    : public DeconvolutionLayer<Dtype, MItype, MOtype> {
 public:
  explicit LibDNNDeconvolutionLayer(const LayerParameter& param)
      : DeconvolutionLayer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual ~LibDNNDeconvolutionLayer();
  virtual void Tune(vptr<MOtype> top_data, vptr<MOtype> top_diff,
                    vptr<MItype> bottom_data, vptr<MItype> bottom_diff,
                    int_tp batch_size);

 protected:
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);


 private:
  shared_ptr<LibDNNDeconv<Dtype, MItype, MOtype> > libdnn_;
};

}  // namespace caffe

#endif  // CAFFE_LIBDNN_DECONV_LAYER_HPP_
#endif  // USE_LIBDNN
