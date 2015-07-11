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
__global__ void MaxPoolNDForward(const int nthreads, const Dtype* bottom_data,
                               const int num, const int channels,
                               const int height, const int width,
                               const int pooled_height, const int pooled_width,
                               const int kernel_h, const int kernel_w,
                               const int ext_kernel_h, const int ext_kernel_w,
                               const int stride_h, const int stride_w,
                               const int kstride_h, const int kstride_w,
                               const int pad_h, const int pad_w,
                               Dtype* top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + ext_kernel_h, height);
    int wend = min(wstart + ext_kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
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

  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;

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
            count, bottom_data, bottom[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_, kernel_h_,
            kernel_w_, ext_kernel_h, ext_kernel_w,
            stride_h_, stride_w_, kstride_h_, kstride_w_,
            pad_h_, pad_w_, top_data,
            mask, top_mask);
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
            oclk_max_pool_forward(count,
                         WrapHandle((cl_mem) bottom_data, &ctx),
                bottom[0]->num(), channels_, height_, width_,
                pooled_height_, pooled_width_, kernel_h_,
                kernel_w_, ext_kernel_h, ext_kernel_w,
                stride_h_, stride_w_, kstride_h_, kstride_w_,
                pad_h_, pad_w_,
                WrapHandle((cl_mem) top_data, &ctx),
                mask == NULL ? 0 : 1,
                WrapHandle((cl_mem) mask, &ctx),
                WrapHandle((cl_mem) top_mask, &ctx)),
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
__global__ void MaxPoolNDBackward(const int nthreads, const Dtype* top_diff,
                                const int* mask, const Dtype* top_mask,
                                const int num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                const int kernel_h, const int kernel_w,
                                const int ext_kernel_h, const int ext_kernel_w,
                                const int stride_h, const int stride_w,
                                const int kstride_h, const int kstride_w,
                                const int pad_h, const int pad_w,
                                Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int pooled_height_1 = pooled_height - 1;
    int pooled_width_1 = pooled_width - 1;
    int phstart = (h < ext_kernel_h) ? h % kstride_h : (h - ext_kernel_h) + 1;
    int phend =
        (h >= pooled_height) ?
            pooled_height_1 - (pooled_height_1 - phstart) % kstride_h : h;
    int pwstart = (w < ext_kernel_w) ? w % kstride_w : (w - ext_kernel_w) + 1;
    int pwend =
        (w >= pooled_width) ?
            pooled_width_1 - (pooled_width_1 - pwstart) % kstride_w : w;

    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    if (mask) {
      mask += offset;
      for (int ph = phstart; ph <= phend; ph += kstride_h) {
        for (int pw = pwstart; pw <= pwend; pw += kstride_w) {
          if (mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    } else {
      mask += offset;
      for (int ph = phstart; ph <= phend; ph += kstride_h) {
        for (int pw = pwstart; pw <= pwend; pw += kstride_w) {
          if (top_mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
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

  int ext_kernel_h = (kernel_h_ - 1) * kstride_h_ + 1;
  int ext_kernel_w = (kernel_w_ - 1) * kstride_w_ + 1;

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
            count, top_diff, mask, top_mask, top[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_,
            kernel_h_, kernel_w_, ext_kernel_h, ext_kernel_w,
            stride_h_, stride_w_, kstride_h_, kstride_w_,
            pad_h_, pad_w_,
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
            oclk_max_pool_backward(count, WrapHandle((cl_mem) top_diff, &ctx),
                                   mask == NULL ? 0 : 1,
                                   WrapHandle((cl_mem) mask, &ctx),
                                   WrapHandle((cl_mem) top_mask, &ctx),
                                   top[0]->num(), channels_, height_, width_,
                                   pooled_height_, pooled_width_, kernel_h_,
                                   kernel_w_, ext_kernel_h, ext_kernel_w,
                                   stride_h_, stride_w_, kstride_h_, kstride_w_,
                                   pad_h_, pad_w_,
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
