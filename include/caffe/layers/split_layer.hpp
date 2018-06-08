#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        int_tpo multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template<typename Dtype, typename MItype, typename MOtype>
class SplitLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "Split"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  int_tp count_;
};

}  // namespace caffe

#endif  // CAFFE_SPLIT_LAYER_HPP_
