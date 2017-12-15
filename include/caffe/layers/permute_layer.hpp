#ifndef CAFFE_PERMUTE_LAYER_HPP_
#define CAFFE_PERMUTE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permute the input blob by changing the memory order of the data.
 *
 * TODO(weiliu89): thorough documentation for Forward, Backward, and proto
 * params.
 */

// The main function which does the permute.
template <typename Dtype>
void Permute(const int count, Dtype *bottom_data, const bool forward,
             const int *permute_order, const int *old_steps,
             const int *new_steps, const int num_axes, Dtype *top_data);

template <typename Dtype> class PermuteLayer : public Layer<Dtype> {
public:
  explicit PermuteLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) {}
  void Reshape_const(const vector<Blob<Dtype> *> &bottom,
                     const vector<Blob<Dtype> *> &top) const {}

  virtual inline const char *type() const { return "Permute"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  void Forward_const_cpu(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top) const override;
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  void Forward_const_gpu(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top) const override;

  int num_axes_;
  bool need_permute_;

  // Use Blob because it is convenient to be accessible in .cu file.
  Blob<int> permute_order_;
};

} // namespace caffe

#endif // CAFFE_PERMUTE_LAYER_HPP_
