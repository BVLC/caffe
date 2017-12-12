#ifndef CAFFE_AXPY_LAYER_HPP_
#define CAFFE_AXPY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief For reduce memory and time both on training and testing, we combine
 *        channel-wise scale operation and element-wise addition operation 
 *        into a single layer called "axpy".
 *       
 */
template <typename Dtype>
class AxpyLayer: public Layer<Dtype> {
 public:
  explicit AxpyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Reshape_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;

  virtual inline const char* type() const { return "Axpy"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
/**
 * @param Formulation:
 *            F = a * X + Y
 *	  Shape info:
 *            a:  N x C          --> bottom[0]      
 *            X:  N x C x H x W  --> bottom[1]       
 *            Y:  N x C x H x W  --> bottom[2]     
 *            F:  N x C x H x W  --> top[0]
 */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_const_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const override;

  Blob<Dtype> spatial_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_AXPY_LAYER_HPP_
