#ifndef CAFFE_AFFINITY_LAYER_HPP_
#define CAFFE_AFFINITY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Computes a one edge per dimension 2D affinity graph
 * for a given segmentation/label map
 */
template<typename Dtype, typename MItype, typename MOtype>
class AffinityLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit AffinityLayer(const LayerParameter& param)
    : Layer<Dtype, MItype, MOtype>(param) {
  }

  virtual inline const char* type() const {
    return "Affinity";
  }

 protected:
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);

  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
                           const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom);

 private:
  vector< shared_ptr< Blob<Dtype> > > min_index_;
  vector<int_tp> offsets_;
};

}  // namespace caffe

#endif  // CAFFE_AFFINITY_LAYER_HPP_
