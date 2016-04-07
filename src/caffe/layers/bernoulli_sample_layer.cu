#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/bernoulli_sample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CompareKernel(const int n, const Dtype* input_data,
                              const Dtype* random, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, n) {
    output_data[index] = Dtype(input_data[index] > random[index] ? 1 : 0);
  }
}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* uniform_sample = static_cast<Dtype*>(rng_data_->mutable_gpu_data());
  const int count = bottom[0]->count();
  caffe_gpu_rng_uniform<Dtype>(count, 0., 1., uniform_sample);

  // Transform this uniform sample to zeros and ones
  // NOLINT_NEXT_LINE(whitespace/operators)
  CompareKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, bottom_data, uniform_sample, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void BernoulliSampleLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(BernoulliSampleLayer);

}  // namespace caffe
