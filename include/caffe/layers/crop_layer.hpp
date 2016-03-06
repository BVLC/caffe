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

template <typename Dtype>
class CropLayer : public Layer<Dtype> {
 public:
  explicit CropLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Crop"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 2; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<int_tp> offsets;

 private:
  void crop_copy(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top,
               const vector<int_tp>& offsets,
               vector<int_tp> indices,
               int_tp cur_dim,
               const Dtype* src_data,
               Dtype* dest_data,
               bool is_forward);

  void crop_copy_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top,
                const vector<int_tp>& offsets,
                vector<int_tp> indices,
                int_tp cur_dim,
                const Dtype* src_data,
                Dtype* dest_data,
                bool is_forward);
};
}  // namespace caffe

#endif  // CAFFE_CROP_LAYER_HPP_
