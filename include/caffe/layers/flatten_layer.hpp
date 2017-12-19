#ifndef CAFFE_FLATTEN_LAYER_HPP_
#define CAFFE_FLATTEN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Reshapes the input Blob into flat vectors.
 *
 * Note: because this layer does not change the input values -- merely the
 * dimensions -- it can simply copy the input. The copy happens "virtually"
 * (thus taking effectively 0 real time) by setting, in Forward, the data
 * pointer of the top Blob to that of the bottom Blob (see Blob::ShareData),
 * and in Backward, the diff pointer of the bottom Blob to that of the top Blob
 * (see Blob::ShareDiff).
 */
template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
  explicit FlattenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const;

  virtual inline const char* type() const { return "Flatten"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times CHW \times 1 \times 1) @f$
   *      the outputs -- i.e., the (virtually) copied, flattened inputs
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_const_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;
  void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override {
    Forward_cpu(bottom,top);
  }
  void Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override {
    Forward_const_cpu(bottom,top);
  }
};

}  // namespace caffe

#endif  // CAFFE_FLATTEN_LAYER_HPP_
