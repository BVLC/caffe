#ifdef USE_OCL
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/concat_layer.hpp"

extern "C" const char _cl_concat_layer_start;
extern "C" const char _cl_concat_layer_end;

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ClState& state = Caffe::cl_state();
  state.submit_program("concat",  &_cl_concat_layer_start,
      &_cl_concat_layer_end);
  ClKernel kernel = state.get_kernel("ConcatForward");

  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    int idx = 0;
    kernel.set_arg(idx++, nthreads);
    kernel.set_arg(idx++, bottom_data);
    kernel.set_arg(idx++, num_concats_);
    kernel.set_arg(idx++, concat_input_size_);
    kernel.set_arg(idx++, top_concat_axis);
    kernel.set_arg(idx++, bottom_concat_axis);
    kernel.set_arg(idx++, offset_concat_axis);
    kernel.set_arg(idx++, top_data);
    kernel.enqueue(nthreads);
    offset_concat_axis += bottom_concat_axis;
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  ClState& state = Caffe::cl_state();
  state.submit_program("concat",  &_cl_concat_layer_start,
      &_cl_concat_layer_end);
  ClKernel kernel = state.get_kernel("ConcatBackward");

  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  for (int i = 0; i < bottom.size(); ++i) {
    if (!propagate_down[i]) { continue; }
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    cl_uint idx = 0;
    kernel.set_arg(idx++, nthreads);
    kernel.set_arg(idx++, top_diff);
    kernel.set_arg(idx++, num_concats_);
    kernel.set_arg(idx++, concat_input_size_);
    kernel.set_arg(idx++, top_concat_axis);
    kernel.set_arg(idx++, bottom_concat_axis);
    kernel.set_arg(idx++, offset_concat_axis);
    kernel.set_arg(idx++, bottom_diff);
    kernel.enqueue(nthreads);
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
#endif // USE_OCL
