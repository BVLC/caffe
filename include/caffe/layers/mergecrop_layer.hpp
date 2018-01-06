#ifndef CAFFE_MERGECROP_LAYER_HPP_
#define CAFFE_MERGECROP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Merges and crops feature maps for U-Net architectures.
 */
template<typename Dtype, typename MItype, typename MOtype>
class MergeCropLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit MergeCropLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {
  }

  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);

  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  virtual inline int_tp ExactNumBottomBlobs() const {
    return 2;
  }

  virtual inline const char* type() const {
    return "MergeCrop";
  }

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

  void GenerateProgram();

 private:
  vector<bool> forward_;
  vector<bool> backward_;
  Blob<int_tp> shape_a_;
  Blob<int_tp> shape_b_;
  MergeCropParameter_MergeOp op_;
};

}  // namespace caffe

#endif  // CAFFE_MERGECROP_LAYER_HPP_
