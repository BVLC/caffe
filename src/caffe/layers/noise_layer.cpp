#include <algorithm>
#include <string>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
const std::string NoiseLayer<Dtype>::GAUSSIAN = "gaussian";
template<typename Dtype>
const std::string NoiseLayer<Dtype>::UNIFORM = "uniform";

template <typename Dtype>
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top) {
  CHECK(this->layer_param().has_noise_param()) <<
      "No noise parameter specified for NoiseLayer.";
  CHECK(this->NoiseParam().has_filler_param()) <<
      "No filler specified for NoiseLayer noise_param.";
  NoiseParameter noise_param = this->layer_param().noise_param();
  CHECK(NoiseType() == GAUSSIAN || NoiseType() == UNIFORM) <<
      "NoiseLayer only supports normally- or uniformly-distributed noise.";
}

template <typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  this->inplace_ = bottom[0] == top[0];
  if (Inplace()) {
    // For in-place noising.
    inplace_noise_.ReshapeLike(*bottom[0]);
  } else {
    top[0]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int count = top[0]->count();

  CHECK(top[0]->count());
  // If we are noising in-place, then put the generated noise in the layer's
  // buffer. Otherwise, put it directly into the top blob.
  Dtype* noise_buffer = Inplace() ? inplace_noise_.mutable_cpu_data() :
                                    top_data;
  if (NoiseType() == GAUSSIAN) {
    caffe_rng_gaussian<Dtype>(count,
        Dtype(FillerParam().mean()),
        Dtype(FillerParam().std()),
        noise_buffer);
  } else if (NoiseType() == UNIFORM) {
    caffe_rng_uniform(count,
        Dtype(FillerParam().min()),
        Dtype(FillerParam().max()),
        noise_buffer);
  } else {
    LOG(FATAL) << "Unexpected noise type in NoiseLayer.";
  }
  // Add the noise to the bottom blob to produce the top blob.
  caffe_add(count, bottom_data, noise_buffer, top_data);
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (bottom.empty()) {
    return;
  }
  // Only copy the top diffs to bottom if we are not noising in-place.
  if (propagate_down[0] && !Inplace()) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(NoiseLayer);
#endif

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);

}  // namespace caffe
