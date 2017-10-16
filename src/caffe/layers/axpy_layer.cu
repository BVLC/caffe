#include "caffe/layers/axpy_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AxpyForward(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* x_data, const Dtype* y_data,
    Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, count) {
    out_data[index] = scale_data[index / spatial_dim] * x_data[index]
        + y_data[index];
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* scale_data = bottom[0]->gpu_data();
  const Dtype* x_data = bottom[1]->gpu_data();
  const Dtype* y_data = bottom[2]->gpu_data();
  Dtype* out_data = top[0]->mutable_gpu_data();
  const int count = bottom[1]->count();
  AxpyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[1]->count(2), scale_data, x_data, y_data, out_data);  
}


INSTANTIATE_LAYER_GPU_FUNCS(AxpyLayer);

}  // namespace caffe
