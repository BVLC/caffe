#pragma once

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe
{
  template <typename Dtype>
  class MILLayer : public Layer<Dtype> {
  public:
    explicit MILLayer(const LayerParameter& param);
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
    virtual const char* type() const override;
    virtual int ExactNumBottomBlobs() const override;
    virtual int ExactNumTopBlobs() const override;

  protected:
    int channels_, height_, width_, num_images_;
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) override;
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) override;
  };
}  // namespace caffe