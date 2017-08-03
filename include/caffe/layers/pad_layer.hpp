#ifndef CAFFE_PAD_LAYER_HPP_
#define CAFFE_PAD_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes a Blob and pad it, to the shape specified by the pad
 * parameter, across all dimensions after the specified axis. This
 * means add extra pixels to the outside. The values of these pixels
 * are set, depending on the value of the padtype parameter. To
 * describe the types, imagine padding along one dimension, that I'll
 * refer to as a row, with all other dimensions held fixed. The index
 * for the first pixel in the row in the unpadded image is 0. The
 * pixel at location i is p[i], and negative i means a pixel in the
 * padding. The types are:
 * 
 *   'zero': Set the padding to zero. This is what Convolution layers
 *      effectively do with their padding.
 *   'constant': Copy the border value into the padding. For positive
 *      i p[-i] is set to p[0].
 *   'even': Reflect interior pixel values into the padding. For
 *      positive index i p[-i] is set to p[i].
 *   'odd': Reflect both spatially and in value.  For positive index i
 *      p[-i] is set to p[0] - (p[i]-p[0]) = 2*p[0] - p[i]. This give
 *      the padding zero second difference across the boundary, so that
 *      it avoids creating ridges or valleys at the borders.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template <typename Dtype>
class PadLayer : public Layer<Dtype> {
 public:
  explicit PadLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pad"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  PadParameter_PadType PAD_TYPE_;
  unsigned int PAD_;
  int NUM_;
  int CHANNEL_;
  int HEIGHT_IN_;
  int WIDTH_IN_;
  int HEIGHT_OUT_;
  int WIDTH_OUT_;

 private:
  // Rec// ursive copy function.
  // void pad_copy(const vector<Blob<Dtype>*>& bottom,
  //              const vector<Blob<Dtype>*>& top,
  //              const int* pads,
  //              vector<int> indices,
  //              int cur_dim,
  //              const Dtype* src_data,
  //              Dtype* dest_data,
  //              bool is_forward);

  // Recursive copy function: this is similar to pad_copy() but loops over all
  // but the last two dimensions to allow for ND padding while still relying on
  // a CUDA kernel for the innermost two dimensions for performance reasons.  An
  // alterantive implementation could rely on the kernel more by passing
  // pads, but this is problematic because of its variable length.
  // Since in the standard (N,C,W,H) case N,C are usually not padded a speedup
  // could be achieved by not looping the application of the copy_kernel around
  // these dimensions.
  // void pad_copy_gpu(const vector<Blob<Dtype>*>& bottom,
  //               const vector<Blob<Dtype>*>& top,
  //               const vector<int>& pads,
  //               vector<int> indices,
  //               int cur_dim,
  //               const Dtype* src_data,
  //               Dtype* dest_data,
  //               bool is_forward);
};
}  // namespace caffe

#endif  // CAFFE_PAD_LAYER_HPP_
