#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void Slice(const int_tp nthreads, const Dtype* in_data,
                      const bool forward, const int_tp num_slices,
                      const int_tp slice_size, const int_tp bottom_slice_axis,
                      const int_tp top_slice_axis,
                      const int_tp offset_slice_axis,
                      Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int_tp total_slice_size = slice_size * top_slice_axis;
    const int_tp slice_num = index / total_slice_size;
    const int_tp slice_index = index % total_slice_size;
    const int_tp bottom_index = slice_index
        + (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
  int_tp offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int_tp bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int_tp i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int_tp top_slice_axis = top[i]->shape(slice_axis_);
    const int_tp top_slice_size = top_slice_axis * slice_size_;
    const int_tp nthreads = top_slice_size * num_slices_;

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS)(
          nthreads, bottom_data, kForward, num_slices_, slice_size_,
          bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      viennacl::ocl::kernel &oclk_slice = program.get_kernel(
          CL_KERNEL_SELECT("slice"));
      viennacl::ocl::enqueue(
          oclk_slice(nthreads, WrapHandle((cl_mem) bottom_data, &ctx),
                     kForward ? 1 : 0, num_slices_, slice_size_,
                     bottom_slice_axis, top_slice_axis, offset_slice_axis,
                     WrapHandle((cl_mem) top_data, &ctx)),
          ctx.get_queue());
#endif  // USE_GREENTEA
    }

    offset_slice_axis += top_slice_axis;
  }
}

template<typename Dtype>
void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int_tp offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const int_tp top_slice_axis = top[i]->shape(slice_axis_);
    const int_tp top_slice_size = top_slice_axis * slice_size_;
    const int_tp nthreads = top_slice_size * num_slices_;

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      CUDA_KERNEL(CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS)(
          nthreads, top_diff, kForward, num_slices_, slice_size_,
          bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      viennacl::ocl::kernel &oclk_slice = program.get_kernel(
          CL_KERNEL_SELECT("slice"));
      viennacl::ocl::enqueue(
          oclk_slice(nthreads, WrapHandle((cl_mem) top_diff, &ctx),
                     kForward ? 1 : 0, num_slices_, slice_size_,
                     bottom_slice_axis, top_slice_axis, offset_slice_axis,
                     WrapHandle((cl_mem) bottom_diff, &ctx)),
          ctx.get_queue());
#endif  // USE_GREENTEA
    }
    offset_slice_axis += top_slice_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SliceLayer);

}  // namespace caffe
