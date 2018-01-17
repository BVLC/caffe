#ifndef CAFFE_CTC_LOSS_LAYER_HPP_
#define CAFFE_CTC_LOSS_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/3rdparty/ctc/ctc.h"

namespace caffe {

#define CHECK_CTC_STATUS(ret_status) do{ctcStatus_t status = ret_status; \
                                        CHECK_EQ(status, CTC_STATUS_SUCCESS) \
                                        << std::string(ctcGetStatusString(status));} while(0);
/**
 * @brief Computes the ctc loss
 */
template <typename Dtype>
class CtcLossLayer : public LossLayer<Dtype> {
 public:
  explicit CtcLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CtcLoss"; }

 protected:
  virtual void FlattenLabels(const Blob<Dtype>* label_blob);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<int> label_lengths_;
  vector<int> flat_labels_;
  vector<int> input_lengths_;
  int total_label_length_;
  int alphabet_size_;
  int blank_label_;
};
}  // namespace caffe
#endif  // CAFFE_CTC_LOSS_LAYER_HPP_
