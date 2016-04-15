#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

#ifdef USE_CUDA
// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template<typename Dtype>
__global__ void copy_kernel(const int_tp n, const int_tp height,
                            const int_tp width, const int_tp src_outer_stride,
                            const int_tp src_inner_stride,
                            const int_tp dest_outer_stride,
                            const int_tp dest_inner_stride, const Dtype* src,
                            Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int_tp src_start = index / height * src_outer_stride
        + index % height * src_inner_stride;
    int_tp dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int_tp i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}
#endif  // USE_CUDA

template <typename Dtype>
void CropLayer<Dtype>::crop_copy_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top,
                                     const vector<int_tp>& offsets,
                                     vector<int_tp> indices,
                                     int_tp cur_dim,
                                     const Dtype* src_data, Dtype* dest_data,
                                     bool is_forward) {
  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (cur_dim + 2 < top[0]->num_axes()) {
      // We are not yet at the final dimension, call copy recursivley
      for (int_tp i = 0; i < top[0]->shape(cur_dim); ++i) {
        indices[cur_dim] = i;
        crop_copy_gpu(bottom, top, offsets, indices, cur_dim + 1, src_data,
                      dest_data, is_forward);
      }
    } else {
      // We are at the last two dimensions,
      // which are stored continously in memory
      // With (N,C,H,W)
      //      (0,1,2,3) cur_dim   -> H
      //                cur_dim+1 -> W
      const int_tp lines = top[0]->shape(cur_dim);
      const int_tp height = top[0]->shape(cur_dim);
      const int_tp width = top[0]->shape(cur_dim + 1);
      std::vector<int_tp> ind_off(cur_dim + 2, 0);
      for (int_tp j = 0; j < cur_dim; ++j) {
        ind_off[j] = indices[j] + offsets[j];
      }
      ind_off[cur_dim] = offsets[cur_dim];
      ind_off[cur_dim + 1] = offsets[cur_dim + 1];
      // Compute copy strides
      const int_tp src_outer_stride = bottom[0]->shape(cur_dim)
          * bottom[0]->shape(cur_dim + 1);
      const int_tp src_inner_stride = bottom[0]->shape(cur_dim + 1);
      const int_tp dest_outer_stride = top[0]->shape(cur_dim)
          * top[0]->shape(cur_dim + 1);
      const int_tp dest_inner_stride = top[0]->shape(cur_dim + 1);

      if (is_forward) {
        const Dtype* bottom_data = bottom[0]->gpu_data()
            + bottom[0]->offset(ind_off);
        Dtype* top_data = top[0]->mutable_gpu_data() + top[0]->offset(indices);
        // NOLINT_NEXT_LINE(whitespace/operators)
        copy_kernel CUDA_KERNEL(CAFFE_GET_BLOCKS(lines),
                                CAFFE_CUDA_NUM_THREADS)(
            lines, height, width,
            src_outer_stride, src_inner_stride,
            dest_outer_stride, dest_inner_stride,
            bottom_data, top_data);

      } else {
        const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(indices);
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff()
            + bottom[0]->offset(ind_off);
        // NOLINT_NEXT_LINE(whitespace/operators)
        copy_kernel CUDA_KERNEL(CAFFE_GET_BLOCKS(lines),
                                CAFFE_CUDA_NUM_THREADS)(
            lines, height, width,
            dest_outer_stride, dest_inner_stride,
            src_outer_stride, src_inner_stride,
            top_diff, bottom_diff);
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();
    viennacl::ocl::kernel &oclk_copy_crop = program.get_kernel(
        CL_KERNEL_SELECT("crop_copy"));

    if (cur_dim + 2 < top[0]->num_axes()) {
      for (int_tp i = 0; i < top[0]->shape(cur_dim); ++i) {
        indices[cur_dim] = i;
        crop_copy_gpu(bottom, top, offsets, indices, cur_dim + 1, src_data,
                      dest_data, is_forward);
      }
    } else {
      const int_tp lines = top[0]->shape(cur_dim);
      const int_tp height = top[0]->shape(cur_dim);
      const int_tp width = top[0]->shape(cur_dim + 1);
      std::vector<int_tp> ind_off(cur_dim + 2, 0);
      for (int_tp j = 0; j < cur_dim; ++j) {
        ind_off[j] = indices[j] + offsets[j];
      }
      ind_off[cur_dim] = offsets[cur_dim];
      ind_off[cur_dim + 1] = offsets[cur_dim + 1];
      // Compute copy strides
      const int_tp src_outer_stride = bottom[0]->shape(cur_dim)
          * bottom[0]->shape(cur_dim + 1);
      const int_tp src_inner_stride = bottom[0]->shape(cur_dim + 1);
      const int_tp dest_outer_stride = top[0]->shape(cur_dim)
          * top[0]->shape(cur_dim + 1);
      const int_tp dest_inner_stride = top[0]->shape(cur_dim + 1);

      if (is_forward) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const int_tp bottom_off = bottom[0]->offset(ind_off);
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int_tp top_off = top[0]->offset(indices);
        viennacl::ocl::enqueue(
            oclk_copy_crop(lines, height, width, src_outer_stride,
                           src_inner_stride, dest_outer_stride,
                           dest_inner_stride,
                           WrapHandle((cl_mem) bottom_data, &ctx), bottom_off,
                           WrapHandle((cl_mem) top_data, &ctx), top_off),
            ctx.get_queue());
      } else {
        const Dtype* top_diff = top[0]->gpu_diff();
        const int_tp top_off = top[0]->offset(indices);
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int_tp bottom_off = bottom[0]->offset(ind_off);
        viennacl::ocl::enqueue(
            oclk_copy_crop(lines, height, width, dest_outer_stride,
                           dest_inner_stride, src_outer_stride,
                           src_inner_stride,
                           WrapHandle((cl_mem) top_diff, &ctx), top_off,
                           WrapHandle((cl_mem) bottom_diff, &ctx), bottom_off),
            ctx.get_queue());
      }
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  std::vector<int_tp> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  crop_copy_gpu(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

template<typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (propagate_down[0]) {
    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
#endif
    } else {
#ifdef USE_GREENTEA
      greentea_gpu_set(this->device_->id(), bottom[0]->count(),
                       static_cast<Dtype>(0), (cl_mem) bottom_diff, 0);
#endif
    }
    std::vector<int_tp> indices(top[0]->num_axes(), 0);
    crop_copy_gpu(bottom, top, offsets, indices, 0, top_diff, bottom_diff,
                  false);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
