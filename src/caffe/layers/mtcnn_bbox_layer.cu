#include <algorithm>
#include <memory>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector>

#include "caffe/layers/mtcnn_bbox_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void filter_by_threshold(const Dtype *prob, const int prob_cnt,
                                    const Dtype threshold, int *out,
                                    int *out_size) {
  //     __shared__ int local_idx[CAFFE_CUDA_NUM_THREADS];

  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < prob_cnt) {
    if (prob[x] > threshold) {
      int old_size = atomicAdd(out_size, 1);
      out[old_size] = x;

      //  local_idx[threadIdx.x]=x;
    } else {
      // local_idx[threadIdx.x]=-1;
    }

    /*
    __syncthreads() ;

    if(threadIdx.x==0) {

    }
    */
  }
}

template <typename Dtype>
__global__ void generateBBox(const Dtype scale, const int height,
                             const int width, const int index_cnt,
                             const int *index_data, const Dtype *bbox_reg,
                             const Dtype *prob, const int stride,
                             const int cellsize, Dtype *out) {
  CUDA_KERNEL_LOOP(i, index_cnt) {
    int idx = index_data[i];
    int h = idx / width;
    int w = idx % width;
    auto out_ptr = out + i * 9;
    out_ptr[0] = static_cast<int>(1e-4 + ((stride * h + 1) / scale - 1));
    out_ptr[1] = static_cast<int>(1e-4 + ((stride * w + 1) / scale - 1));
    out_ptr[2] = static_cast<int>(1e-4 + ((stride * h + cellsize) / scale - 1));
    out_ptr[3] = static_cast<int>(1e-4 + ((stride * w + cellsize) / scale - 1));
    out_ptr[4] = prob[idx];
    out_ptr[5] = (bbox_reg[0 * width * height + idx]);
    out_ptr[6] = (bbox_reg[1 * width * height + idx]);
    out_ptr[7] = (bbox_reg[2 * width * height + idx]);
    out_ptr[8] = (bbox_reg[3 * width * height + idx]);
  }
}

template <typename Dtype>
void MTCNNBBoxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom, top);
}

template <typename Dtype>
void MTCNNBBoxLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) const {
  const auto bbox_reg = bottom[0]->gpu_data();
  const auto &shape = bottom[1]->shape();
  const auto prob = bottom[1]->gpu_data() + shape[2] * shape[3];
  const auto scale = bottom[2]->cpu_data()[0];

  std::unique_ptr<Blob<int>> indices_ptr;
  indices_ptr.reset(new Blob<int>(shape[2] * shape[3], 1, 1, 1));

  std::unique_ptr<Blob<int>> index_cnt_ptr;
  index_cnt_ptr.reset(new Blob<int>(1, 1, 1, 1));

  index_cnt_ptr->mutable_cpu_data()[0] = 0;

  filter_by_threshold<Dtype>
      <<<CAFFE_GET_BLOCKS(shape[2] * shape[3]), CAFFE_CUDA_NUM_THREADS>>>(
          prob, shape[2] * shape[3], threshold_,
          indices_ptr->mutable_gpu_data(), index_cnt_ptr->mutable_gpu_data());

  auto cnt = indices_ptr->mutable_cpu_data()[0];

  if (cnt == 0) {
    return;
  }

  top[0]->Reshape(1, 1, cnt, 9);

  generateBBox<Dtype><<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      scale, shape[2], shape[3], (int)cnt, indices_ptr->gpu_data(), bbox_reg,
      prob, stride_, cellsize_, top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(MTCNNBBoxLayer);

} // namespace caffe
