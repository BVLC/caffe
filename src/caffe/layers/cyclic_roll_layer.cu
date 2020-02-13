#include <vector>

#include "caffe/layers/cyclic_roll_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CyclicRollForward(const int n,
    const Dtype* bottom_data,
    const int bottom_batch_dim, const int channel_dim,
    const int bottom_channels, const int size,
    Dtype* top_data) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    int bottom_batch_id = index/bottom_batch_dim;
    int bottom_batch_rotid = bottom_batch_id%4;
    int tmp = index%bottom_batch_dim;
    int bottom_channel_id = tmp/channel_dim;
    tmp = tmp%channel_dim;
    int h = tmp/size;
    int w = tmp%size;
    int top_channels = bottom_channels*4;
    for (int top_batch_rotid = 0; top_batch_rotid < 4; ++top_batch_rotid) {
        int rotation = (bottom_batch_rotid-top_batch_rotid)%4;
        rotation = (rotation >= 0)?rotation:(4+rotation);
        int top_channel_id = 4*bottom_channel_id+ rotation;
        int top_batch_id = bottom_batch_id-bottom_batch_rotid+top_batch_rotid;
        switch (rotation) {
          case 0:
          top_data[((top_batch_id*top_channels+top_channel_id)*
            size+h)*size+w] = bottom_data[index]; break;
          case 1:
          top_data[((top_batch_id*top_channels+top_channel_id)*size+size-1-w)*
            size+h] = bottom_data[index]; break;
          case 2:
          top_data[((top_batch_id*top_channels+top_channel_id)*size+size-1-h)*
            size+size-1-w] = bottom_data[index]; break;
          case 3:
          top_data[((top_batch_id*top_channels+top_channel_id)*size+w)*
            size+size-1-h] = bottom_data[index]; break;
          default: break;
       }
    }
  }
}

template <typename Dtype>
__global__ void CyclicRollBackward(const int n,
    const Dtype* top_diff,
    const int bottom_batch_dim, const int channel_dim,
    const int bottom_channels, const int size,
    Dtype* bottom_diff) {
  // each kernel moves one
  CUDA_KERNEL_LOOP(index, n) {
    int bottom_batch_id = index/bottom_batch_dim;
    int bottom_batch_rotid = bottom_batch_id%4;
    int tmp = index%bottom_batch_dim;
    int bottom_channel_id = tmp/channel_dim;
    tmp = tmp%channel_dim;
    int h = tmp/size;
    int w = tmp%size;
    int top_channels = bottom_channels*4;
    for (int top_batch_rotid = 0; top_batch_rotid < 4; ++top_batch_rotid) {
        int rotation = (bottom_batch_rotid-top_batch_rotid)%4;
        rotation = (rotation >= 0)?rotation:(4+rotation);
        int top_channel_id = 4*bottom_channel_id+rotation;
        int top_batch_id = bottom_batch_id-bottom_batch_rotid+top_batch_rotid;
        switch (rotation) {
          case 0:
          bottom_diff[index] +=
          top_diff[((top_batch_id*top_channels+top_channel_id)*size+h)*size+w];
          break;
          case 1:
          bottom_diff[index] +=
          top_diff[((top_batch_id*top_channels+top_channel_id)*size+size-1-w)*
            size+h];
          break;
          case 2:
          bottom_diff[index] +=
          top_diff[((top_batch_id*top_channels+top_channel_id)*size+size-1-h)*
            size+size-1-w];
          break;
          case 3:
          bottom_diff[index] +=
          top_diff[((top_batch_id*top_channels+top_channel_id)*size+w)*
            size+size-1-h];
          break;
          default: break;
       }
    }
  }
}



template <typename Dtype>
void CyclicRollLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  const int bottom_channels = bottom[0]->shape(1);
  const int bottom_batch_dim = bottom[0]->count(1);
  const int channel_dim = bottom[0]->count(2);
  const int size = bottom[0]->shape(2);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int bottom_count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CyclicRollForward<Dtype> <<<CAFFE_GET_BLOCKS(bottom_count),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_count,
    bottom_data, bottom_batch_dim, channel_dim,
    bottom_channels, size, top_data);
}

template <typename Dtype>
void CyclicRollLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int num = bottom[0]->shape(0);
  const int bottom_channels = bottom[0]->shape(1);
  const int bottom_batch_dim = bottom[0]->count(1);
  const int channel_dim = bottom[0]->count(2);
  const int size = bottom[0]->shape(2);
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int bottom_count = bottom[0]->count();
  // clear bottom diff
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  CyclicRollBackward<Dtype> <<<CAFFE_GET_BLOCKS(bottom_count),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_count,
    top_diff, bottom_batch_dim, channel_dim,
    bottom_channels, size, bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(CyclicRollLayer);

}  // namespace caffe
