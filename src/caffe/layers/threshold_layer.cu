#include <vector>

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#include "caffe/layers/threshold_layer.hpp"
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> pod/caffe-merge
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> device-abstraction
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> pod/post-rebase-error-fix
=======
#include "caffe/neuron_layers.hpp"
>>>>>>> pod-caffe-pod.hpp-merge

namespace caffe {

template <typename Dtype>
__global__ void ThresholdForward(const int n, const Dtype threshold,
    const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}

template <typename Dtype>
void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, threshold_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);


}  // namespace caffe
