#ifndef CAFFE_BERNOULLI_SAMPLE_LAYER_HPP_
#define CAFFE_BERNOULLI_SAMPLE_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Given a blob of probabilites, create bernoulli samples
 */

template <typename Dtype>
class BernoulliSampleLayer : public Layer<Dtype> {
 public:
  explicit BernoulliSampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "BernoulliSample"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @brief Create random bernoulli samples.
   * @param bottom vector with single pointer to a blob. The blob should have
   *        values between zero and 1.
   * @param top vector with single pointer to a blob. The top will be the same
   *            shape as the bottom, and will have zero or one values. The
   *            probability of an entry being one is equal to the entry at that
   *            position in bottom.
   */
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
  shared_ptr<SyncedMemory> rng_data_;
};

}  // namespace caffe

#endif  // CAFFE_BERNOULLI_SAMPLE_LAYER_HPP_
