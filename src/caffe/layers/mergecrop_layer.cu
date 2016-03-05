#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mergecrop_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void CopyForwardStack(const int_tp nthreads, const int_tp dims,
                                 const Dtype* bottom_a, const bool forward_a,
                                 const Dtype* bottom_b, const bool forward_b,
                                 Dtype* top, const int_tp num,
                                 const int_tp channels_a,
                                 const int_tp channels_b, const int_tp* shape_a,
                                 const int_tp* shape_b) {
  int_tp pad[6];      // NOLINT(runtime/arrays)
  int_tp tmp_idx[6];  // NOLINT(runtime/arrays)
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp batch_id = index / ((channels_a + channels_b) * size_a);
    int_tp bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int_tp channel_id = (index / size_a) % channels_a;
      int_tp aidx = batch_id * channels_a + channel_id;
      for (int_tp i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      top[index] = forward_a ? bottom_a[aidx] : 0;
    } else {
      int_tp channel_id = (index / size_a) % channels_b;
      int_tp bidx = (batch_id * channels_b + channel_id) * size_b;
      int_tp btemp = 1;
      for (int_tp i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      top[index] = forward_b ? bottom_b[bidx] : 0;
    }
  }
}

template<typename Dtype>
__global__ void CopyBackwardStack(const int_tp nthreads, const int_tp dims,
                                  Dtype* bottom_a, const bool backward_a,
                                  Dtype* bottom_b, const bool backward_b,
                                  const Dtype* top, const int_tp num,
                                  const int_tp channels_a,
                                  const int_tp channels_b,
                                  const int_tp* shape_a,
                                  const int_tp* shape_b) {
  int_tp pad[6];      // NOLINT(runtime/arrays)
  int_tp tmp_idx[6];  // NOLINT(runtime/arrays)
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp batch_id = index / ((channels_a + channels_b) * size_a);
    int_tp bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int_tp channel_id = (index / size_a) % channels_a;
      int_tp aidx = batch_id * channels_a + channel_id;
      for (int_tp i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      bottom_a[aidx] = backward_a ? top[index] : 0;
    } else {
      int_tp channel_id = (index / size_a) % channels_b;
      int_tp bidx = (batch_id * channels_b + channel_id) * size_b;
      int_tp btemp = 1;
      for (int_tp i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      bottom_b[bidx] = backward_b ? top[index] : 0;
    }
  }
}

template<typename Dtype>
__global__ void CopyForwardAdd(const int_tp nthreads, const int_tp dims,
                               const Dtype* bottom_a, const bool forward_a,
                               const Dtype* bottom_b, const bool forward_b,
                               Dtype* top, const int_tp num,
                               const int_tp channels, const int_tp* shape_a,
                               const int_tp* shape_b) {
  int_tp pad[6];      // NOLINT(runtime/arrays)
  int_tp tmp_idx[6];  // NOLINT(runtime/arrays)
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp batch_id = index / (channels * size_a);
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    top[index] = 0;
    int_tp channel_id = (index / size_a) % channels;
    int_tp aidx = batch_id * channels + channel_id;
    for (int_tp i = 0; i < dims; ++i) {
      aidx *= shape_a[i];
      aidx += tmp_idx[i];
    }
    top[index] = forward_a ? top[index] + bottom_a[aidx] : top[index];
    int_tp bidx = (batch_id * channels + channel_id) * size_b;
    int_tp btemp = 1;
    for (int_tp i = dims - 1; i >= 0; --i) {
      bidx += btemp * (tmp_idx[i] + pad[i]);
      btemp *= shape_b[i];
    }
    top[index] = forward_b ? top[index] + bottom_b[bidx] : top[index];
  }
}

template<typename Dtype>
__global__ void CopyBackwardAdd(const int_tp nthreads, const int_tp dims,
                                Dtype* bottom_a, const bool backward_a,
                                Dtype* bottom_b, const bool backward_b,
                                const Dtype* top, const int_tp num,
                                const int_tp channels, const int_tp* shape_a,
                                const int_tp* shape_b) {
  int_tp pad[6];      // NOLINT(runtime/arrays)
  int_tp tmp_idx[6];  // NOLINT(runtime/arrays)
  int_tp size_a = 1;
  int_tp size_b = 1;

  for (int_tp i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int_tp batch_id = index / (channels * size_a);
    int_tp counter = index;
    for (int_tp i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    int_tp channel_id = (index / size_a) % channels;
    int_tp aidx = batch_id * channels + channel_id;
    for (int_tp i = 0; i < dims; ++i) {
      aidx *= shape_a[i];
      aidx += tmp_idx[i];
    }
    bottom_a[aidx] = backward_a ? top[index] : 0;
    int_tp bidx = (batch_id * channels + channel_id) * size_b;
    int_tp btemp = 1;
    for (int_tp i = dims - 1; i >= 0; --i) {
      bidx += btemp * (tmp_idx[i] + pad[i]);
      btemp *= shape_b[i];
    }
    bottom_b[bidx] = backward_b ? top[index] : 0;
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void MergeCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  int_tp count = top[0]->count();

  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int_tp num = bottom[0]->shape(0);
  int_tp spatial_dims = bottom[0]->shape().size() - 2;

  // All channels of both inputs are copied
  int_tp channels_a = bottom[0]->shape(1);
  int_tp channels_b = bottom[1]->shape(1);

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    switch (op_) {
      case MergeCropParameter_MergeOp_STACK: {
        CopyForwardStack<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS) (
            count, spatial_dims, bottom_data_a,
            forward_[0], bottom_data_b,
            forward_[1], top_data, num, channels_a,
            channels_b, shape_a_.gpu_data(), shape_b_.gpu_data());
        CUDA_POST_KERNEL_CHECK;
      }
      break;
      case MergeCropParameter_MergeOp_ADD: {
        CopyForwardAdd<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS) (
            count, spatial_dims, bottom_data_a,
            forward_[0], bottom_data_b,
            forward_[1], top_data, num, channels_a,
            shape_a_.gpu_data(), shape_b_.gpu_data());
        CUDA_POST_KERNEL_CHECK;
      }
      break;
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    switch (op_) {
      case MergeCropParameter_MergeOp_STACK: {
        viennacl::ocl::kernel &oclk_copy_forward = program.get_kernel(
            CL_KERNEL_SELECT("merge_copy_forward_stack"));
        viennacl::ocl::enqueue(
            oclk_copy_forward(count, spatial_dims,
                              WrapHandle((cl_mem) bottom_data_a, &ctx),
                              forward_[0],
                              WrapHandle((cl_mem) bottom_data_b, &ctx),
                              forward_[1], WrapHandle((cl_mem) top_data, &ctx),
                              num, channels_a, channels_b,
                              WrapHandle((cl_mem) (shape_a_.gpu_data()), &ctx),
                              WrapHandle((cl_mem) (shape_b_.gpu_data()), &ctx)),
            ctx.get_queue());
      }
      break;
      case MergeCropParameter_MergeOp_ADD: {
        viennacl::ocl::kernel &oclk_copy_forward = program.get_kernel(
            CL_KERNEL_SELECT("merge_copy_forward_add"));
        viennacl::ocl::enqueue(
            oclk_copy_forward(count, spatial_dims,
                              WrapHandle((cl_mem) bottom_data_a, &ctx),
                              forward_[0],
                              WrapHandle((cl_mem) bottom_data_b, &ctx),
                              forward_[1], WrapHandle((cl_mem) top_data, &ctx),
                              num, channels_a,
                              WrapHandle((cl_mem) (shape_a_.gpu_data()), &ctx),
                              WrapHandle((cl_mem) (shape_b_.gpu_data()), &ctx)),
            ctx.get_queue());
      }
      break;
    }
    ctx.get_queue().finish();
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void MergeCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  int_tp count = top[0]->count();

  Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_b = bottom[1]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  int_tp num = bottom[0]->shape(0);
  int_tp spatial_dims = bottom[0]->shape().size() - 2;

  // All channels of both inputs are copied
  int_tp channels_a = bottom[0]->shape(1);
  int_tp channels_b = bottom[1]->shape(1);

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    switch (op_) {
      case MergeCropParameter_MergeOp_STACK: {
        CopyBackwardStack<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS) (
            count, spatial_dims, bottom_diff_a, backward_[0],
            bottom_diff_b, backward_[1], top_diff, num,
            channels_a, channels_b, shape_a_.gpu_data(), shape_b_.gpu_data());
        CUDA_POST_KERNEL_CHECK;
      }
      break;
      case MergeCropParameter_MergeOp_ADD: {
        CopyBackwardAdd<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS) (
            count, spatial_dims, bottom_diff_a, backward_[0],
            bottom_diff_b, backward_[1], top_diff, num,
            channels_a, shape_a_.gpu_data(), shape_b_.gpu_data());
        CUDA_POST_KERNEL_CHECK;
      }
      break;
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    switch (op_) {
      case MergeCropParameter_MergeOp_STACK: {
        viennacl::ocl::kernel &oclk_copy_backward = program.get_kernel(
            CL_KERNEL_SELECT("merge_copy_backward_stack"));
        viennacl::ocl::enqueue(
            oclk_copy_backward(
                count, spatial_dims, WrapHandle((cl_mem) bottom_diff_a, &ctx),
                backward_[0], WrapHandle((cl_mem) bottom_diff_b, &ctx),
                backward_[1], WrapHandle((cl_mem) top_diff, &ctx), num,
                channels_a, channels_b,
                WrapHandle((cl_mem) (shape_a_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (shape_b_.gpu_data()), &ctx)),
            ctx.get_queue());
      }
      break;
      case MergeCropParameter_MergeOp_ADD: {
        viennacl::ocl::kernel &oclk_copy_backward = program.get_kernel(
            CL_KERNEL_SELECT("merge_copy_backward_add"));
        viennacl::ocl::enqueue(
            oclk_copy_backward(
                count, spatial_dims, WrapHandle((cl_mem) bottom_diff_a, &ctx),
                backward_[0], WrapHandle((cl_mem) bottom_diff_b, &ctx),
                backward_[1], WrapHandle((cl_mem) top_diff, &ctx), num,
                channels_a, WrapHandle((cl_mem) (shape_a_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (shape_b_.gpu_data()), &ctx)),
            ctx.get_queue());
      }
      break;
    }

    ctx.get_queue().finish();
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MergeCropLayer);

}  // namespace caffe
