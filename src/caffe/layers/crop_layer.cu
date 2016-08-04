#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void copy_kernel(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dest_outer_stride, const int dest_inner_stride,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index / height * src_outer_stride
                  + index % height * src_inner_stride;
    int dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::crop_copy_gpu(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const vector<int>& offsets,
             vector<int> indices,
             int cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) {
  if (cur_dim + 2 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursivley
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      indices[cur_dim] = i;
      crop_copy_gpu(bottom, top, offsets, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last two dimensions, which are stored continously in memory
    // With (N,C,H,W)
    //      (0,1,2,3) cur_dim   -> H
    //                cur_dim+1 -> W
    const int lines = top[0]->shape(cur_dim);
    const int height = top[0]->shape(cur_dim);
    const int width = top[0]->shape(cur_dim+1);
    std::vector<int> ind_off(cur_dim+2, 0);
    for (int j = 0; j < cur_dim; ++j) {
        ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];
    ind_off[cur_dim+1] = offsets[cur_dim+1];
    // Compute copy strides
    const int src_outer_stride =
        bottom[0]->shape(cur_dim)*bottom[0]->shape(cur_dim+1);
    const int src_inner_stride = bottom[0]->shape(cur_dim+1);
    const int dest_outer_stride =
        top[0]->shape(cur_dim)*top[0]->shape(cur_dim+1);
    const int dest_inner_stride = top[0]->shape(cur_dim+1);

    if (is_forward) {
      const Dtype* bottom_data = bottom[0]->gpu_data() +
          bottom[0]->offset(ind_off);
      Dtype* top_data = top[0]->mutable_gpu_data() +
          top[0]->offset(indices);
      // NOLINT_NEXT_LINE(whitespace/operators)
      copy_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
          lines, height, width,
          src_outer_stride, src_inner_stride,
          dest_outer_stride, dest_inner_stride,
          bottom_data, top_data);

    } else {
      const Dtype* top_diff = top[0]->gpu_diff() +
          top[0]->offset(indices);
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() +
          bottom[0]->offset(ind_off);
      // NOLINT_NEXT_LINE(whitespace/operators)
      copy_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
          lines, height, width,
          dest_outer_stride, dest_inner_stride,
          src_outer_stride, src_inner_stride,
          top_diff, bottom_diff);
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  crop_copy_gpu(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    std::vector<int> indices(top[0]->num_axes(), 0);
    crop_copy_gpu(bottom, top, offsets, indices, 0, top_diff, bottom_diff,
                  false);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
