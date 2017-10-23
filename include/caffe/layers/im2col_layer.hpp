#ifndef CAFFE_IM2COL_LAYER_HPP_
#define CAFFE_IM2COL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief a helper for image operations that rearranges image regions int_tpo
 *        column vectors.  Used by ConvolutionLayer to perform convolution
 *        by matrix multiplication.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template<typename Dtype, typename MItype, typename MOtype>
class Im2colLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "Im2col"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

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

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int_tp> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int_tp> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int_tp> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int_tp> dilation_;

  int_tp num_spatial_axes_;
  int_tp bottom_dim_;
  int_tp top_dim_;

  int_tp channel_axis_;
  int_tp num_;
  int_tp channels_;

  bool force_nd_im2col_;
};

}  // namespace caffe

#endif  // CAFFE_IM2COL_LAYER_HPP_
