#include <cfloat>
#include <vector>

#include "caffe/layers/cyclic_pool_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CyclicPoolAVEForward(const int n,
    const Dtype* bottom_data,
    const int batch_dim, Dtype* top_data) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    int inner_index = index%batch_dim;
    int batch_num = index/batch_dim;
    top_data[index] = (bottom_data[4*batch_num*batch_dim+inner_index]
      + bottom_data[(4*batch_num+1)*batch_dim+inner_index]
      + bottom_data[(4*batch_num+2)*batch_dim+inner_index]
      + bottom_data[(4*batch_num+3)*batch_dim+inner_index])/4;
  }
}

template <typename Dtype>
__global__ void CyclicPoolMAXForward(const int n,
    const Dtype* bottom_data,
    const int batch_dim, Dtype* top_data, int* maxidx_data) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    int inner_index = index%batch_dim;
    int batch_num = index/batch_dim;
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    int bottom_index;
    for (int i = 0; i < 4; ++i) {
      bottom_index = (4*batch_num+i)*batch_dim+inner_index;
      if (bottom_data[bottom_index] > maxval) {
        maxval = bottom_data[bottom_index];
        maxidx = bottom_index;
      }
    }
    top_data[index] = maxval;
    maxidx_data[index] = maxidx;
  }
}

template <typename Dtype>
__global__ void CyclicPoolAVEBackward(const int n,
    const Dtype* top_diff,
    const int batch_dim, Dtype* bottom_diff) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    int inner_index = index%batch_dim;
    int batch_num = index/batch_dim;
    bottom_diff[4*batch_num*batch_dim+inner_index] += top_diff[index]/4;
    bottom_diff[(4*batch_num+1)*batch_dim+inner_index] += top_diff[index]/4;
    bottom_diff[(4*batch_num+2)*batch_dim+inner_index] += top_diff[index]/4;
    bottom_diff[(4*batch_num+3)*batch_dim+inner_index] += top_diff[index]/4;
  }
}

template <typename Dtype>
__global__ void CyclicPoolMAXBackward(const int n,
    const Dtype* top_diff, const int* maxidx_data,
    Dtype* bottom_diff) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[maxidx_data[index]] += top_diff[index];
  }
}

template <typename Dtype>
void CyclicPoolLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  const int batch_dim = bottom[0]->count(1);
  const int top_count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* mask = max_idx_.mutable_gpu_data();

  // different pooling method
  switch (this->layer_param_.cyclic_pool_param().pool()) {
    case CyclicPoolParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    CyclicPoolAVEForward<Dtype> <<<CAFFE_GET_BLOCKS(top_count),
      CAFFE_CUDA_NUM_THREADS>>>(top_count,
      bottom_data, batch_dim, top_data);
    CUDA_POST_KERNEL_CHECK;
    break;
    case CyclicPoolParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    CyclicPoolMAXForward<Dtype> <<<CAFFE_GET_BLOCKS(top_count),
      CAFFE_CUDA_NUM_THREADS>>>(top_count,
      bottom_data, batch_dim, top_data, mask);
    CUDA_POST_KERNEL_CHECK;
    break;
    case CyclicPoolParameter_PoolMethod_RMS:
    CHECK(0) << "currently not supported";
    break;
    default:
    break;
  }
}

template <typename Dtype>
void CyclicPoolLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int batch_dim = top[0]->count(1);
  const int top_count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int* mask = max_idx_.gpu_data();
  // clear bottom diff
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // different pooling method
  switch (this->layer_param_.cyclic_pool_param().pool()) {
    case CyclicPoolParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    CyclicPoolAVEBackward<Dtype> <<<CAFFE_GET_BLOCKS(top_count),
      CAFFE_CUDA_NUM_THREADS>>>(top_count,
      top_diff, batch_dim, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    break;
    case CyclicPoolParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    CyclicPoolMAXBackward<Dtype> <<<CAFFE_GET_BLOCKS(top_count),
      CAFFE_CUDA_NUM_THREADS>>>(top_count,
      top_diff, mask, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    break;
    case CyclicPoolParameter_PoolMethod_RMS:
    CHECK(0) << "currently not supported";
    break;
    default:
    break;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CyclicPoolLayer);

}  // namespace caffe
