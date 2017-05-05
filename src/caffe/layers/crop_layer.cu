#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

#ifdef USE_CUDA
__device__ int_tp compute_uncropped_index(
    int_tp index,
    const int_tp ndims,
    const int_tp* src_strides,
    const int_tp* dst_strides,
    const int_tp* offsets) {
  int_tp dst_index = index;
  int_tp src_index = 0;
  for (int_tp i = 0; i < ndims; ++i) {
      int_tp coord = dst_index / dst_strides[i];
      dst_index -= coord * dst_strides[i];
      src_index += src_strides[i] * (coord + offsets[i]);
  }
  return src_index;
}

template <typename Dtype>
__global__ void crop_kernel_forward(const int_tp nthreads,
    const int_tp ndims,
    const int_tp* src_strides,
    const int_tp* dst_strides,
    const int_tp* offsets,
    const Dtype* src, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp src_index = compute_uncropped_index(
        index, ndims, src_strides, dst_strides, offsets);
    dst[index] = src[src_index];
  }
}

template <typename Dtype>
__global__ void crop_kernel_backward(const int_tp nthreads,
    const int_tp ndims,
    const int_tp* src_strides,
    const int_tp* dst_strides,
    const int_tp* offsets,
    Dtype* src, const Dtype* dst) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp src_index = compute_uncropped_index(
        index, ndims, src_strides, dst_strides, offsets);
    src[src_index] = dst[index];
  }
}
#endif  // USE_CUDA



template<typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int_tp n = top[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
  // NOLINT_NEXT_LINE(whitespace/operators)
  crop_kernel_forward CUDA_KERNEL(CAFFE_GET_BLOCKS(n),
                                  CAFFE_CUDA_NUM_THREADS)(n,
      bottom[0]->num_axes(),
      src_strides_.gpu_data(),
      dst_strides_.gpu_data(),
      offsets.gpu_data(),
      bottom_data, top_data);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_crop_forward = program.get_kernel(
        CL_KERNEL_SELECT("crop_forward"));

    viennacl::ocl::enqueue(
        oclk_crop_forward(n,
                         bottom[0]->num_axes(),
                         WrapHandle((cl_mem)(src_strides_.gpu_data()), &ctx),
                         WrapHandle((cl_mem)(dst_strides_.gpu_data()), &ctx),
                         WrapHandle((cl_mem)(offsets.gpu_data()), &ctx),
                         WrapHandle((cl_mem)(bottom_data), &ctx),
                         0,
                         WrapHandle((cl_mem)(top_data), &ctx),
                         0),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int_tp n = top[0]->count();
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
  if (propagate_down[0]) {
      caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
      // NOLINT_NEXT_LINE(whitespace/operators)
      crop_kernel_backward CUDA_KERNEL(CAFFE_GET_BLOCKS(n),
                                       CAFFE_CUDA_NUM_THREADS)(n,
          bottom[0]->num_axes(),
          src_strides_.gpu_data(),
          dst_strides_.gpu_data(),
          offsets.gpu_data(),
          bottom_diff, top_diff);
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (propagate_down[0]) {
      viennacl::ocl::context &ctx = viennacl::ocl::get_context(
          this->device_->id());
      viennacl::ocl::program &program = this->device_->program();

      greentea_gpu_set<Dtype>(this->device_->id(), bottom[0]->count(), 0,
                              (cl_mem)bottom_diff, 0);

      viennacl::ocl::kernel &oclk_crop_backward = program.get_kernel(
          CL_KERNEL_SELECT("crop_backward"));

      viennacl::ocl::enqueue(
          oclk_crop_backward(n,
                           bottom[0]->num_axes(),
                           WrapHandle((cl_mem)(src_strides_.gpu_data()), &ctx),
                           WrapHandle((cl_mem)(dst_strides_.gpu_data()), &ctx),
                           WrapHandle((cl_mem)(offsets.gpu_data()), &ctx),
                           WrapHandle((cl_mem)(bottom_diff), &ctx),
                           0,
                           WrapHandle((cl_mem)(top_diff), &ctx),
                           0),
          ctx.get_queue());
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
