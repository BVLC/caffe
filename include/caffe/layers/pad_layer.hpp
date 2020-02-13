#ifndef CAFFE_PAD_LAYER_HPP_
#define CAFFE_PAD_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes a Blob and expands it by an amount specified by the
 * pad parameter, across the last two dimensions. This means add extra
 * pixels to the outside. These pixels are set to values that depend
 * on the value of the padtype parameter. To describe the types,
 * imagine padding along one dimension, that I'll refer to as a row,
 * with all other dimensions held fixed. The following describes the
 * padding at the beginning of a row; the padding at the end is
 * analogous. The index for the first pixel in the row in the unpadded
 * image is 0. The pixel at location i is p[i], and negative i means a
 * pixel in the padding. The types are:
 *
 *   'zero': Set the padding to zero. This is what Convolution layers
 *      effectively do with their padding.
 *   'replicate': Copy the border value into the padding. For positive
 *      i p[-i] is set to p[0].
 *   'reflect': Reflect interior pixel values into the padding. For
 *      index i >= 0. p[-i] is set to p[i-1].
 *   'reflect_101': Reflect interior pixel values into the
 *      padding. For index i >= 0. p[-i] is set to p[i].
 *   'odd': (Not yet implemented.) Reflect both spatially and in
 *      value.  For positive index i p[-i] is set to p[0] -
 *      (p[i-1]-p[0]) = 2*p[0] - p[i-1]. This give the padding zero
 *      second difference across the boundary, so that it avoids
 *      creating ridges or valleys at the borders.
 *   'odd_101': (Not yet implemented.) Reflect both spatially and in
 *      value.  For positive index i p[-i] is set to p[0] -
 *      (p[i]-p[0]) = 2*p[0] - p[i]. This give the padding zero second
 *      difference across the boundary, so that it avoids creating
 *      ridges or valleys at the borders.
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
};
}  // namespace caffe

#endif  // CAFFE_PAD_LAYER_HPP_
