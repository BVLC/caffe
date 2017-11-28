/*!
 * \file mtcnn_bbox_layer.hpp
 *
 * \brief
 * \author cyy
 * \date 2017-11-28
 */
#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MTCNNBBoxLayer : public Layer<Dtype> {
 public:
  explicit MTCNNBBoxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
	threshold_ = this->layer_param_.threshold_param().threshold();
	DCHECK(threshold_ >= 0.);
	DCHECK(threshold_ <= 1.);
      }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { }
  virtual void Reshape_const(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top) const override { }

  virtual inline const char* type() const { return "MTCNNBBoxLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu_const(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const;

  /*
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      */

  static constexpr int stride_ {2};
  static  constexpr int cellsize_{12};
  Dtype threshold_;
};

}  // namespace caffe

