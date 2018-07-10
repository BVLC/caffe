#ifndef CAFFE_CROP_LAYER_HPP_
#define CAFFE_CROP_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes a Blob and crop it, to the shape specified by the second input
 *  Blob, across all dimensions after the specified axis.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

template<typename Dtype, typename MItype, typename MOtype>
class CropLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit CropLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "Crop"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 2; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  Blob<int_tp> offsets;
  Blob<int_tp> src_strides_;
  Blob<int_tp> dst_strides_;

 private:
  // Recursive copy function.
  void crop_copy(const vector<Blob<MItype>*>& bottom,
               const vector<Blob<MOtype>*>& top,
               const int_tp* offsets,
               vector<int_tp> indices,
               int_tp cur_dim,
               const Dtype* src_data,
               Dtype* dest_data,
               bool is_forward);

  // Recursive copy function: this is similar to crop_copy() but loops over all
  // but the last two dimensions to allow for ND cropping while still relying on
  // a CUDA kernel for the innermost two dimensions for performance reasons.  An
  // alterantive implementation could rely on the kernel more by passing
  // offsets, but this is problematic because of its variable length.
  // Since in the standard (n,c,W,H) case n,c are usually not cropped a speedup
  // could be achieved by not looping the application of the copy_kernel around
  // these dimensions.
  void crop_copy_gpu(const vector<Blob<MItype>*>& bottom,
                const vector<Blob<MOtype>*>& top,
                const vector<int_tp>& offsets,
                vector<int_tp> indices,
                int_tp cur_dim,
                const Dtype* src_data,
                Dtype* dest_data,
                bool is_forward);

  void GenerateProgram();
};
}  // namespace caffe

#endif  // CAFFE_CROP_LAYER_HPP_
