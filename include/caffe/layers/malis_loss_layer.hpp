#ifndef CAFFE_MALIS_LOSS_LAYER_HPP_
#define CAFFE_MALIS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class MalisLossLayer : public LossLayer<Dtype> {
 public:
  explicit MalisLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MalisLoss"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return -1; }
  virtual inline int_tp MinBottomBlobs() const { return 3; }
  virtual inline int_tp MaxBottomBlobs() const { return 4; }
  virtual inline int_tp ExactNumTopBlobs() const { return -1; }
  virtual inline int_tp MinTopBlobs() const { return 1; }
  virtual inline int_tp MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  void Malis(const Dtype* conn_data, const int_tp conn_num_dims,
             const int_tp* conn_dims,
             const int_tp* nhood_data, const int_tp* nhood_dims,
             const Dtype* seg_data,
             const bool pos, Dtype* dloss_data, Dtype* loss_out,
             Dtype *classerr_out, Dtype *rand_index_out);

  int_tp nedges_;
  int_tp conn_num_dims_;
  std::vector<int_tp> conn_dims_;
  std::vector<int_tp> nhood_data_;
  std::vector<int_tp> nhood_dims_;

  Blob<Dtype> affinity_pos_;
  Blob<Dtype> affinity_neg_;
  Blob<Dtype> dloss_pos_;
  Blob<Dtype> dloss_neg_;
};

}  // namespace caffe

#endif  // CAFFE_MALIS_LOSS_LAYER_HPP_
