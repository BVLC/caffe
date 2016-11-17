#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/quantization_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ForwardGPU(const int nthreads, const Dtype* bottom_data, Dtype threshold, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if((threshold > 0) && ((abs(bottom_data[index]) - 1) > threshold)){
          loss_data[index] = (abs(bottom_data[index]) - 1) - log(Dtype(2));
      }
      else{
          loss_data[index] = log(cosh(abs(bottom_data[index]) - 1));
      }
  }
}

template <typename Dtype>
void QuantizationLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();

  //calculate quantization loss
  int nthreads = outer_num_ * inner_num_;
  ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, threshold_, loss_.mutable_gpu_data());

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_.gpu_data(), &loss);
  loss /= nthreads;

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void BackwardGPU(const int nthreads, const int num, const int dim, 
        const Dtype* bottom_data, const Dtype scale, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(bottom_data[index] >= 0){
        bottom_diff[index] = scale * tanh(abs(bottom_data[index]) - 1);
    }
    else{
        bottom_diff[index] = -scale * tanh(abs(bottom_data[index]) - 1);
    }
  }
}

template <typename Dtype>
void QuantizationLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    
    //calculate quantization diff
    int nthreads = inner_num_ * outer_num_;
    BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, outer_num_, inner_num_, bottom_data, 
                loss_weight_ / outer_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizationLossLayer);

}  // namespace caffe
