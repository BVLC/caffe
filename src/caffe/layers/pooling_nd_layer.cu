#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif  // USE_GREENTEA

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void MaxPoolNDForward(const int n, const int num_axes,
                                 const Dtype* bottom_data,
                                 const int channels, const int* size,
                                 const int* pooled_size, const int* kernel_size,
                                 const int* ext_kernel_size, const int* stride,
                                 const int* kstride, const int* pad,
                                 Dtype* top_data, int* mask, Dtype* top_mask) {
  int d_idx[6];  // NOLINT(runtime/arrays)
  int d_start[6];  // NOLINT(runtime/arrays)
  int d_end[6];  // NOLINT(runtime/arrays)
  int d_iter[6];  // NOLINT(runtime/arrays)
  int i;

  CUDA_KERNEL_LOOP(index, n) {
    int offset = 1;
    int num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = index % pooled_size[i];
      d_start[i] = d_idx[i] * stride[i] - pad[i];
      d_end[i] = min(d_start[i] + ext_kernel_size[i], size[i]);
      d_start[i] = max(d_start[i], 0);
      num /= pooled_size[i];
      offset *= size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] >= d_end[i]) {
        top_data[index] = -FLT_MAX;
        if (mask) {
          mask[index] = -1;
        } else {
          top_mask[index] = -1;
        }
        return;
      }
    }
    int chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    int final_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      int size_prod = 1;
      for (i = num_axes - 1; i >= 0; --i) {
        final_offset += d_iter[i] * size_prod;
        size_prod *= size[i];
      }

      if (bottom_data[final_offset] > maxval) {
        maxidx = final_offset;
        maxval = bottom_data[maxidx];
      }

      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        if (d_iter[i] >= d_end[i] - kstride[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += kstride[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);

    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void PoolingNDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX:
        if (use_top_mask) {
          top_mask = top[1]->mutable_gpu_data();
        } else {
          mask = max_idx_.mutable_gpu_data();
        }
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxPoolNDForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                            CAFFE_CUDA_NUM_THREADS)(
            count, num_spatial_axes_, bottom_data,
            channels_, size_.gpu_data(), pooled_size_.gpu_data(),
            kernel_shape_.gpu_data(), ext_kernel_shape_.gpu_data(),
            stride_.gpu_data(), kstride_.gpu_data(), pad_.gpu_data(),
            top_data, mask, top_mask);
        break;
      default: {
        LOG(FATAL)<< "Unknown pooling method.";
      }
    }
    CUDA_POST_KERNEL_CHECK;

#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_->id());

    switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX: {
        if (use_top_mask) {
          top_mask = top[1]->mutable_gpu_data();
        } else {
          mask = max_idx_.mutable_gpu_data();
        }
        viennacl::ocl::kernel &oclk_max_pool_forward = program.get_kernel(
            CL_KERNEL_SELECT("max_pool_forward_nd"));
        viennacl::ocl::enqueue(
            oclk_max_pool_forward(count, num_spatial_axes_,
                WrapHandle((cl_mem)bottom_data, &ctx),
                channels_,
                WrapHandle((cl_mem)(size_.gpu_data()), &ctx),
                WrapHandle((cl_mem)(pooled_size_.gpu_data()), &ctx),
                WrapHandle((cl_mem)(kernel_shape_.gpu_data()), &ctx),
                WrapHandle((cl_mem)(ext_kernel_shape_.gpu_data()), &ctx),
                WrapHandle((cl_mem)(stride_.gpu_data()), &ctx),
                WrapHandle((cl_mem)(kstride_.gpu_data()), &ctx),
                WrapHandle((cl_mem)(pad_.gpu_data()), &ctx),
                WrapHandle((cl_mem)top_data, &ctx),
                mask == NULL ? 0 : 1,
                WrapHandle((cl_mem)mask, &ctx),
                WrapHandle((cl_mem)top_mask, &ctx)),
            ctx.get_queue());
      }
      break;
      default: {
        LOG(FATAL)<< "Unknown pooling method.";
      }
    }
#endif  // USE_GREENTEA
  }
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void MaxPoolNDBackward(const int n, const int num_axes,
                                  const Dtype* top_diff, const int* mask,
                                  const Dtype* top_mask,
                                  const int channels, const int* size,
                                  const int* pooled_size,
                                  const int* kernel_size,
                                  const int* ext_kernel_size, const int* stride,
                                  const int* kstride, const int* pad,
                                  Dtype* bottom_diff) {
  int d_idx[6];  // NOLINT(runtime/arrays)
  int d_start[6];  // NOLINT(runtime/arrays)
  int d_end[6];  // NOLINT(runtime/arrays)
  int d_iter[6];  // NOLINT(runtime/arrays)
  int i;

  CUDA_KERNEL_LOOP(index, n) {
    // find out the local index
    // find out the local offset
    int offset = 1;
    int num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = num % size[i];
      d_start[i] = (d_idx[i] < ext_kernel_size[i]) ?
          d_idx[i] % kstride[i] : (d_idx[i] - ext_kernel_size[i]) + 1;
      d_end[i] = (d_idx[i] >= pooled_size[i]) ?
          (pooled_size[i] - 1) - (pooled_size[i] - 1 - d_start[i]) %
          kstride[i] : d_idx[i];
      num /= size[i];
      offset *= pooled_size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] > d_end[i]) {
        bottom_diff[index] = 0;
        return;
      }
    }
    int chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype gradient = 0;
    int final_offset = 0;
    int im_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      im_offset = 0;
      int size_prod = 1;
      int pooled_size_prod = 1;
      for (i = num_axes - 1; i >= 0; --i) {
        final_offset += d_iter[i] * pooled_size_prod;
        im_offset += d_idx[i] * size_prod;
        size_prod *= size[i];
        pooled_size_prod *= pooled_size[i];
      }

      if (mask) {
        if (mask[final_offset] == im_offset) {
          gradient += top_diff[final_offset];
        }
      } else {
        if (top_mask[final_offset] == im_offset) {
          gradient += top_diff[final_offset];
        }
      }

      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        if (d_iter[i] > d_end[i] - kstride[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += kstride[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);
    bottom_diff[index] = gradient;
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void PoolingNDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    caffe_gpu_set(count, Dtype(0.), bottom_diff);
    switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX:
        if (use_top_mask) {
          top_mask = top[1]->gpu_data();
        } else {
          mask = max_idx_.gpu_data();
        }
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxPoolNDBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                             CAFFE_CUDA_NUM_THREADS)(
            count, num_spatial_axes_, top_diff, mask, top_mask,
            channels_, size_.gpu_data(), pooled_size_.gpu_data(),
            kernel_shape_.gpu_data(), ext_kernel_shape_.gpu_data(),
            stride_.gpu_data(), kstride_.gpu_data(), pad_.gpu_data(),
            bottom_diff);
        break;
      default:
        LOG(FATAL)<<
        "Unknown or unsupported pooling method in Backward_gpu().";
      }
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_->id());

    greentea_gpu_set(this->device_context_->id(), count, Dtype(0.),
                     (cl_mem) bottom_diff, 0);

    switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX: {
        if (use_top_mask) {
          top_mask = top[1]->gpu_data();
        } else {
          mask = max_idx_.gpu_data();
        }
        viennacl::ocl::kernel &oclk_max_pool_backward = program.get_kernel(
            CL_KERNEL_SELECT("max_pool_backward_nd"));
        viennacl::ocl::enqueue(
            oclk_max_pool_backward(
                count, num_spatial_axes_, WrapHandle((cl_mem) top_diff, &ctx),
                mask == NULL ? 0 : 1,
                WrapHandle((cl_mem) mask, &ctx),
                WrapHandle((cl_mem) top_mask, &ctx), channels_,
                WrapHandle((cl_mem) (size_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (pooled_size_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (kernel_shape_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (ext_kernel_shape_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (stride_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (kstride_.gpu_data()), &ctx),
                WrapHandle((cl_mem) (pad_.gpu_data()), &ctx),
                WrapHandle((cl_mem) bottom_diff, &ctx)),
            ctx.get_queue());
      }
        break;
      default:
        LOG(FATAL)<<
        "Unknown or unsupported pooling method in Backward_gpu().";
      }
#endif  // USE_GREENTEA
    }
  }

INSTANTIATE_LAYER_GPU_FUNCS(PoolingNDLayer);

}  // namespace caffe
