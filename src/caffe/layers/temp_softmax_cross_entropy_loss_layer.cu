#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TempSoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* input_data = mvn_in_output_.gpu_data();
    const Dtype* target = mvn_target_output_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, input_data, bottom_diff);
    caffe_gpu_axpby(count, Dtype(-1)*temperature, target, temperature,
            bottom_diff);
    const int N = bottom[0]->channels();
    caffe_gpu_scal(count, Dtype(1) / (N * temperature * temperature),
                        bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(TempSoftmaxCrossEntropyLossLayer);


}  // namespace caffe
