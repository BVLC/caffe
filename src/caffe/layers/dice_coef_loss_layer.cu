#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  
  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[0]->gpu_data(), 
                tmp_.mutable_gpu_data());
  
  caffe_gpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.gpu_data(), 
                    multiplier_.gpu_data(), Dtype(1.), result_tmp_.mutable_gpu_data());
  caffe_gpu_mul(count, bottom[1]->gpu_data(), bottom[1]->gpu_data(), 
                tmp_.mutable_gpu_data());
  caffe_gpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(1.), tmp_.gpu_data(), 
                    multiplier_.gpu_data(), Dtype(1.), result_tmp_.mutable_gpu_data());
  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
                tmp_.mutable_gpu_data());
  caffe_gpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(2.), tmp_.gpu_data(), 
                    multiplier_.gpu_data(), Dtype(1.), result_.mutable_gpu_data());
  caffe_gpu_div(bottom[0]->num(), result_.gpu_data(), result_tmp_.gpu_data(), 
                result_.mutable_gpu_data());
  Dtype loss;
  caffe_gpu_asum(bottom[0]->num(), result_.gpu_data(), &loss);
  loss /= bottom[0]->num();
  loss = Dtype(1.) - loss;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void DiceCoefLossBackward(const int n, const Dtype* x,
    const Dtype* y, Dtype* out_diff, const Dtype sign,
    const Dtype* loss, const Dtype* denominator, const int dim) {
  CUDA_KERNEL_LOOP(index, n) {
    const int i = index / dim;
    const Dtype alpha = Dtype(-2.) / denominator[i] * sign;
    const Dtype beta = Dtype(2.) * loss[i] / denominator[i] * sign;
    out_diff[index] = alpha * x[index] + beta * y[index];
  }
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      int count = bottom[i]->count();
      const Dtype sign = Dtype(1.0)*top[0]->cpu_diff()[0];
      const int index = (i == 0) ? 1 : 0;
      DiceCoefLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[index]->gpu_data(), bottom[i]->gpu_data(), bottom[i]->mutable_gpu_diff(), 
        sign, result_.gpu_data(), result_tmp_.gpu_data(), bottom[i]->count(1));
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DiceCoefLossLayer);

}  // namespace caffe
