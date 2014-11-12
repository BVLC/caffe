#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SwitchForward(const int n, const Dtype* select, 
    const Dtype* in0, const Dtype* in1, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = select[index] ? in1[index] : in0[index];
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* select = bottom[0]->gpu_data();
  const Dtype* in0 = bottom[1]->gpu_data();
  const Dtype* in1 = bottom[2]->gpu_data();
  const int num_elem = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SwitchForward<Dtype><<<CAFFE_GET_BLOCKS(num_elem), CAFFE_CUDA_NUM_THREADS>>>(
      num_elem, select, in0, in1, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // don't need to implement yet
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
