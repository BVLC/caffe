#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_const_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) const {
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int slice_axis=get_slice_axis(bottom);
  const int bottom_slice_axis = bottom[0]->shape(slice_axis);
  const bool kForward = true;
  const int num_slices = bottom[0]->count(0, slice_axis);
  const int slice_size = bottom[0]->count(slice_axis + 1);
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis);
    const int top_slice_size = top_slice_axis * slice_size;
    const int nthreads = top_slice_size * num_slices;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_slices, slice_size,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
    offset_slice_axis += top_slice_axis;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS_CONST(SliceLayer);

}  // namespace caffe
