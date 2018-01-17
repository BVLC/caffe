#ifndef CAFFE_CONTINUATION_INDICATOR_LAYER_HPP_
#define CAFFE_CONTINUATION_INDICATOR_LAYER_HPP_
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    template <typename Dtype>
    class ContinuationIndicatorLayer: public Layer<Dtype> {
        public:
        explicit ContinuationIndicatorLayer(const LayerParameter& param)
              : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const { return "ContinuationIndicator";}
        virtual inline int ExactNumBottomBlobs() const { return 0;}
        virtual inline int ExactNumTopBlobs() const { return 1;}

        protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom) {}
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& bottom);
        int mini_batch_;
        int time_step_;
    };
}

#endif // end file
