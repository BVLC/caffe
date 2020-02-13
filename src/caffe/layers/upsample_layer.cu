#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

__device__ int translate_idx_inv(
    int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

template <typename Dtype>
__global__ void upscale(const Dtype *input, Dtype *output,
        int no_elements, int scale_factor, int d1, int d2, int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}

template <typename Dtype>
__global__ void downscale(Dtype *gradInput_data, const Dtype *gradOutput_data,
                          int no_elements, int scale_factor, int d1, int d2,
                          int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  if (ii >= no_elements) return;
  for (int i = 0; i < scale_factor; i++) {
    for (int j = 0; j < scale_factor; j++) {
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput_data[ii] += gradOutput_data[ipidx];
    }
  }
}



template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int d1, d2, d3;

  d1 = top[0]->shape(1);
  d2 = top[0]->shape(2);
  d3 = top[0]->shape(3);

  int no_elements = top[0]->count();

  upscale<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(no_elements), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->gpu_data(),
      top[0]->mutable_gpu_data(), no_elements, scale_, d1, d2, d3);
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int d1, d2, d3;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  d1 = bottom[0]->shape(1);
  d2 = bottom[0]->shape(2);
  d3 = bottom[0]->shape(3);
  int no_elements = bottom[0]->count();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  downscale<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(no_elements), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_diff, top[0]->gpu_diff(), no_elements, scale_, d1, d2, d3);
}

INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);

}  // namespace caffe

