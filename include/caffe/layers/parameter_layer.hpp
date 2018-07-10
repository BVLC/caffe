#ifndef CAFFE_PARAMETER_LAYER_HPP_
#define CAFFE_PARAMETER_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class ParameterLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit ParameterLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(1);
      this->blobs_[0].reset(new Blob<Dtype>());
      this->blobs_[0]->Reshape(this->layer_param_.parameter_param().shape());
    }
    top[0]->Reshape(this->layer_param_.parameter_param().shape());

    this->InitializeQuantizers(bottom, top);
  }
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) { }
  virtual inline const char* type() const { return "Parameter"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 0; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
    top[0]->ShareData(*(this->blobs_[0]));
    top[0]->ShareDiff(*(this->blobs_[0]));
  }
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom)
  { }
};

}  // namespace caffe

#endif
