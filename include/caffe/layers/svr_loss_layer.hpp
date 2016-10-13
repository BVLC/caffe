#ifndef CAFFE_SVR_LOSS_LAYER_HPP_
#define CAFFE_SVR_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief This Layer computes the Support Vector Regression (SVR)(L1)
 *        loss. It does not include epsilon insensitivity i.e. 
 *        epsilon is assumed to be zero in this implementation.
 *        L = \frac{1}{N} \sum\limits_{n=1}^N \left| \left| \hat{f}_n - y_n
 *        \right| \right|_1 @f$ for real-valued regression tasks. 
 *        Takes the output of the network beneath it (bottom_blob[0]) and 
 *        the ground truth scores (bottom_blob[1]) as inputs and outputs the 
 *        loss        
 */
template <typename Dtype>
class SVRLossLayer : public LossLayer<Dtype> {
 public:
  explicit SVRLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SVRLoss"; }  

 protected: 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);    

  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_SVR_LOSS_LAYER_HPP_
