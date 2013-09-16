#include "caffeine/layer.hpp"
#include "caffeine/vision_layers.hpp"
#include <algorithm>

using std::max;

namespace caffeine {

template <typename Dtype>
void DropoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  // Set up the cache for random number generation
  rand_mat_.reset(new Blob<float>(bottom.num(), bottom.channels(),
      bottom.height(), bottom.width());
  filler_.reset(new UniformFiller<float>(FillerParameter()));
};

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // First, create the random matrix
  filler_->Fill(rand_mat_.get()); 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* rand_vals = rand_mat_->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  float threshold = layer_param_->dropout_ratio();
  float scale = layer_param_->dropo
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = rand_mat_ > ;
  }
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] >= 0);
    }
  }
  return Dtype(0);
}

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = max(in[index], Dtype(0.));
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int blocks = (count + CAFFEINE_CUDA_NUM_THREADS - 1) /
      CAFFEINE_CUDA_NUM_THREADS;
  DropoutForward<<<blocks, CAFFEINE_CUDA_NUM_THREADS>>>(count, bottom_data,
      top_data);
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] * (in_data[index] >= 0);
  }
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    const int blocks = (count + CAFFEINE_CUDA_NUM_THREADS - 1) /
        CAFFEINE_CUDA_NUM_THREADS;
    DropoutBackward<<<blocks, CAFFEINE_CUDA_NUM_THREADS>>>(count, top_diff,
        bottom_data, bottom_diff);
  }
  return Dtype(0);
}

template class DropoutLayer<float>;
template class DropoutLayer<double>;


}  // namespace caffeine
