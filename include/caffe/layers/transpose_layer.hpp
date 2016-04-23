#ifndef CAFFE_TRANSPOSE_LAYER_HPP_
#define CAFFE_TRANSPOSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  template <typename Dtype>
  class TransposeLayer : public Layer<Dtype> {
   public:
    explicit TransposeLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Transpose"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom);

   private:
    TransposeParameter transpose_param_;
    vector<int> permute(const vector<int>& vec);
    Blob<int> bottom_counts_;
    Blob<int> top_counts_;
    Blob<int> forward_map_;
    Blob<int> backward_map_;
    Blob<int> buf_;
  };

}  // namespace caffe

#endif  // CAFFE_TRANSPOSE_LAYER_HPP_
