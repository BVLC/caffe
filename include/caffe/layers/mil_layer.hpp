#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {
template <typename Dtype>
class MILLayer : public Layer<Dtype> {
 public:
    explicit MILLayer(const LayerParameter& param);
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual const char* type() const;
    virtual int ExactNumBottomBlobs() const;
    virtual int ExactNumTopBlobs() const;

 protected:
    int channels_, height_, width_, num_images_;
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};
}  // namespace caffe
