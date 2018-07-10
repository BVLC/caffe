#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief An interface for layers that take one blob as input (@f$ X @f$)
 *        and produce one equally-sized blob as output (@f$ Y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 */
template<typename Dtype, typename MItype, typename MOtype>
class NeuronLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
