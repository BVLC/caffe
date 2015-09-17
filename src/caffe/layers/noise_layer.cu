#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  CHECK(top[0]->count());
  // If we are noising in-place, then put the generated noise in the layer's
  // buffer. Otherwise, put it directly into the top blob.
  Dtype* noise_buffer = Inplace() ? inplace_noise_.mutable_gpu_data() :
                                    top_data;
  if (NoiseType() == GAUSSIAN) {
    caffe_gpu_rng_gaussian<Dtype>(count,
        Dtype(FillerParam().mean()),
        Dtype(FillerParam().std()),
        noise_buffer);
  } else if (NoiseType() == UNIFORM) {
    caffe_gpu_rng_uniform(count,
        Dtype(FillerParam().min()),
        Dtype(FillerParam().max()),
        noise_buffer);
  } else {
    LOG(FATAL) << "Unexpected noise type in NoiseLayer.";
  }
  caffe_gpu_add(count, bottom_data, noise_buffer, top_data);
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);

}  // namespace caffe
