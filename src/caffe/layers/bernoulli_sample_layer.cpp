#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/bernoulli_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  rng_data_.reset(new SyncedMemory(0));
}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  if (rng_data_->size() < bottom[0]->count() * sizeof(Dtype)) {
    rng_data_.reset(new SyncedMemory(bottom[0]->count() * sizeof(Dtype)));
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int random_number;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    caffe_rng_bernoulli(1, bottom_data[i], &random_number);
    top_data[i] = random_number;
  }
}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      LOG(FATAL) << "BernoulliSampleLayer::Backward_cpu is not yet implemented";
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BernoulliSampleLayer);
#endif

INSTANTIATE_CLASS(BernoulliSampleLayer);
REGISTER_LAYER_CLASS(BernoulliSample);

}  // namespace caffe
