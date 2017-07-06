#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void LRNFillScale(const int_tp nthreads, const Dtype* const in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width,
                             const int_tp size, const Dtype alpha_over_size,
                             const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int_tp head = 0;
    const int_tp pre_pad = (size - 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      CrossChannelForward_gpu(bottom, top);
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      WithinChannelForward(bottom, top);
      break;
    default:
      LOG(FATAL)<< "Unknown normalization region.";
    }
  }

// TODO: check if it would be faster to just put it into the previous kernel.
#ifdef USE_CUDA
template<typename Dtype>
__global__ void LRNComputeOutput(const int_tp nthreads, const Dtype* const in,
                                 const Dtype* const scale,
                                 const Dtype negative_beta, Dtype* const out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_fuse_pooling_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    bool use_fuse) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  CHECK(IsFusedWithPoolMax() && this->device_->backend() == BACKEND_OpenCL);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
      this->device_->id());
  viennacl::ocl::program &program = this->device_->program();

  if (use_fuse) {
    viennacl::ocl::kernel &oclk_lrn_fill = program.get_kernel(
        CL_KERNEL_SELECT("lrn_fuse_pool_max"));
    #define TILE_W 16
    #define TILE_H 8
    size_t simd_size = TILE_W;
    cl_uint argIdx = 0;
    const int_tp tile_pooled_block_h = (TILE_H - pool_h_) / pool_stride_h_ + 1;
    const int_tp tile_pooled_block_w = (TILE_W - pool_w_) / pool_stride_w_ + 1;
    const int tiled_width = (width_ + tile_pooled_block_w * pool_stride_w_ - 1)
                            / (tile_pooled_block_w * pool_stride_w_);
    const int tiled_height =
      (height_ + tile_pooled_block_h * pool_stride_h_ - 1)
       / (tile_pooled_block_h * pool_stride_h_);
    int_tp n_threads = num_ * tiled_width * tiled_height;
    size_t global_work_size_[2] = {(size_t)n_threads, simd_size};
    size_t local_work_size[2] = {1, simd_size};
    oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
    oclk_lrn_fill.arg(argIdx++, channels_);
    oclk_lrn_fill.arg(argIdx++, height_);
    oclk_lrn_fill.arg(argIdx++, width_);
    oclk_lrn_fill.arg(argIdx++, tiled_height);
    oclk_lrn_fill.arg(argIdx++, tiled_width);
    oclk_lrn_fill.arg(argIdx++, size_);
    oclk_lrn_fill.arg(argIdx++, alpha_ / size_);
    oclk_lrn_fill.arg(argIdx++, fixup_arg_type(k_));
    oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) top_data, &ctx));
    oclk_lrn_fill.arg(argIdx++, fixup_arg_type(-beta_));
    oclk_lrn_fill.arg(argIdx++, pool_h_);
    oclk_lrn_fill.arg(argIdx++, pool_w_);
    oclk_lrn_fill.arg(argIdx++, pool_stride_h_);
    oclk_lrn_fill.arg(argIdx++, pool_stride_w_);
    oclk_lrn_fill.arg(argIdx++, pooled_height_);
    oclk_lrn_fill.arg(argIdx++, pooled_width_);
    oclk_lrn_fill.arg(argIdx++, tile_pooled_block_h);
    oclk_lrn_fill.arg(argIdx++, tile_pooled_block_w);
    OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     oclk_lrn_fill.handle().get(), 2, NULL,
                                     global_work_size_, local_work_size, 0,
                                     NULL, NULL));
  } else {
    Dtype* top_lrn_data = lrn_top_blob_.mutable_gpu_data();
    // Do LRN firstly.
    cl_uint argIdx = 0;
    int_tp n_threads = num_ * height_ * width_;
    size_t global_work_size_[1] = {(size_t)n_threads};
    viennacl::ocl::kernel &oclk_lrn_fill = program.get_kernel(
        CL_KERNEL_SELECT("lrn_full_no_scale"));
    oclk_lrn_fill.arg(argIdx++, n_threads);
    oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
    oclk_lrn_fill.arg(argIdx++, num_);
    oclk_lrn_fill.arg(argIdx++, channels_);
    oclk_lrn_fill.arg(argIdx++, height_);
    oclk_lrn_fill.arg(argIdx++, width_);
    oclk_lrn_fill.arg(argIdx++, size_);
    oclk_lrn_fill.arg(argIdx++, alpha_ / size_);
    oclk_lrn_fill.arg(argIdx++, fixup_arg_type(k_));
    oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) top_lrn_data, &ctx));
    oclk_lrn_fill.arg(argIdx++, fixup_arg_type(-beta_));
    OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                   oclk_lrn_fill.handle().get(), 1, NULL,
                                   global_work_size_, NULL, 0, NULL,
                                   NULL));
    // Do pooling.
    viennacl::ocl::kernel &oclk_max_pool_forward = program.get_kernel(
      CL_KERNEL_SELECT("max_pool_forward_no_mask"));

    int_tp count = pooled_width_ * pooled_height_ * channels_ * num_;
    viennacl::ocl::enqueue(
      oclk_max_pool_forward(count,
          WrapHandle((cl_mem) top_lrn_data, &ctx),
          num_, channels_, height_, width_,
          pooled_height_, pooled_width_, pool_h_,
          pool_w_, pool_stride_h_, pool_stride_w_, 0, 0,
          WrapHandle((cl_mem) top_data, &ctx)),
      ctx.get_queue());
  }
}

template<typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // We will launch one kernel for each pixel location, and have the kernel
    // go through all the channels.
    int_tp n_threads = num_ * height_ * width_;
    // NOLINT_NEXT_LINE(whitespace/operators)
    LRNFillScale CUDA_KERNEL(CAFFE_GET_BLOCKS(n_threads),
                             CAFFE_CUDA_NUM_THREADS)(
        n_threads, bottom_data, num_, channels_, height_,
        width_, size_,
        alpha_ / size_, k_, scale_data);
    CUDA_POST_KERNEL_CHECK;
    n_threads = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    LRNComputeOutput CUDA_KERNEL(CAFFE_GET_BLOCKS(n_threads),
                                 CAFFE_CUDA_NUM_THREADS)(
        n_threads, bottom_data, scale_data, -beta_, top_data);
    CUDA_POST_KERNEL_CHECK;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();


    if (this->phase_ == caffe::TRAIN) {
      cl_uint argIdx = 0;
      int_tp n_threads = num_ * height_ * width_;
      size_t global_work_size_[1] = {(size_t)n_threads};
      viennacl::ocl::kernel &oclk_lrn_fill = program.get_kernel(
          CL_KERNEL_SELECT("lrn_full"));

      oclk_lrn_fill.arg(argIdx++, n_threads);
      oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
      oclk_lrn_fill.arg(argIdx++, num_);
      oclk_lrn_fill.arg(argIdx++, channels_);
      oclk_lrn_fill.arg(argIdx++, height_);
      oclk_lrn_fill.arg(argIdx++, width_);
      oclk_lrn_fill.arg(argIdx++, size_);
      oclk_lrn_fill.arg(argIdx++, alpha_ / size_);
      oclk_lrn_fill.arg(argIdx++, fixup_arg_type(k_));
      oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) scale_data, &ctx));
      oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) top_data, &ctx));
      oclk_lrn_fill.arg(argIdx++, fixup_arg_type(-beta_));

      OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     oclk_lrn_fill.handle().get(), 1, NULL,
                                     global_work_size_, NULL, 0, NULL,
                                     NULL));
    } else {
      if (!IsFused()) {
        cl_uint argIdx = 0;
        int_tp n_threads = num_ * height_ * width_;
        size_t global_work_size_[1] = {(size_t)n_threads};
        viennacl::ocl::kernel &oclk_lrn_fill = program.get_kernel(
            CL_KERNEL_SELECT("lrn_full_no_scale"));
        oclk_lrn_fill.arg(argIdx++, n_threads);
        oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
        oclk_lrn_fill.arg(argIdx++, num_);
        oclk_lrn_fill.arg(argIdx++, channels_);
        oclk_lrn_fill.arg(argIdx++, height_);
        oclk_lrn_fill.arg(argIdx++, width_);
        oclk_lrn_fill.arg(argIdx++, size_);
        oclk_lrn_fill.arg(argIdx++, alpha_ / size_);
        oclk_lrn_fill.arg(argIdx++, fixup_arg_type(k_));
        oclk_lrn_fill.arg(argIdx++, WrapHandle((cl_mem) top_data, &ctx));
        oclk_lrn_fill.arg(argIdx++, fixup_arg_type(-beta_));
        OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_lrn_fill.handle().get(), 1, NULL,
                                       global_work_size_, NULL, 0, NULL,
                                       NULL));
      } else if (IsFusedWithPoolMax()) {
        // We can't make sure the fused kernel be the faster for all platforms.
        // have to apply a simple tuning here.
        if (this->device_->CheckCapability("cl_intel_subgroups")) {
          if (fuse_tuned_) {
            CrossChannelForward_fuse_pooling_gpu(bottom, top, tuned_use_fuse_);
          } else {
            float elapsedTime[2];
            bool use_fuse[2] = {true, false};
            // warm up.
            CrossChannelForward_fuse_pooling_gpu(bottom, top, true);
            CrossChannelForward_fuse_pooling_gpu(bottom, top, false);
            for (int i = 0; i < 2; i++) {
              Timer timer;
              timer.initted();
              timer.Start();
              int loop_cnt = 2;
              for (int j = 0; j < loop_cnt; j++) {
                CrossChannelForward_fuse_pooling_gpu(bottom, top, use_fuse[i]);
              }
              timer.Stop();
              elapsedTime[i] = timer.MilliSeconds() / loop_cnt;
            }
            tuned_use_fuse_ = elapsedTime[0] < elapsedTime[1];
            fuse_tuned_ = true;
          }
        } else {
          CrossChannelForward_fuse_pooling_gpu(bottom, top, false);
        }
      }
    }
#endif  // USE_GREENTEA
  }
}
#ifdef HAS_HALF_SUPPORT
template void LRNLayer<half>::CrossChannelForward_gpu(
    const vector<Blob<half>*>& bottom, const vector<Blob<half>*>& top);
template void LRNLayer<half>::CrossChannelForward_fuse_pooling_gpu(
    const vector<Blob<half>*>& bottom, const vector<Blob<half>*>& top, bool);
#endif
template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

template void LRNLayer<float>::CrossChannelForward_fuse_pooling_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top, bool);
template void LRNLayer<double>::CrossChannelForward_fuse_pooling_gpu(
    const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top, bool);

template<typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
      CrossChannelBackward_gpu(top, propagate_down, bottom);
      break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
      WithinChannelBackward(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL)<< "Unknown normalization region.";
    }
  }

#ifdef USE_CUDA
template<typename Dtype>
__global__ void LRNComputeDiff(const int_tp nthreads,
                               const Dtype* const bottom_data,
                               const Dtype* const top_data,
                               const Dtype* const scale,
                               const Dtype* const top_diff, const int_tp num,
                               const int_tp channels, const int_tp height,
                               const int_tp width, const int_tp size,
                               const Dtype negative_beta,
                               const Dtype cache_ratio,
                               Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    int_tp head = 0;
    const int_tp pre_pad = size - (size + 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step]
          / scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step]
          / scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step]
            * top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)
          * step] * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step]
            * top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)
          * step] * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int_tp n_threads = num_ * height_ * width_;

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    // NOLINT_NEXT_LINE(whitespace/operators)
    LRNComputeDiff CUDA_KERNEL(CAFFE_GET_BLOCKS(n_threads),
                               CAFFE_CUDA_NUM_THREADS)(
        n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
        scale_.gpu_data(), top[0]->gpu_diff(), num_,
        channels_, height_, width_,
        size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
        bottom[0]->mutable_gpu_diff());
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    viennacl::ocl::kernel &oclk_lrn = program.get_kernel(
        CL_KERNEL_SELECT("lrn_compute_diff"));
    viennacl::ocl::enqueue(
        oclk_lrn(n_threads, WrapHandle((cl_mem) (bottom[0]->gpu_data()), &ctx),
                 WrapHandle((cl_mem) (top[0]->gpu_data()), &ctx),
                 WrapHandle((cl_mem) (scale_.gpu_data()), &ctx),
                 WrapHandle((cl_mem) (top[0]->gpu_diff()), &ctx), num_,
                 channels_, height_, width_, size_, fixup_arg_type(-beta_),
                 fixup_arg_type(Dtype(2. * alpha_ * beta_ / size_)),
                 WrapHandle((cl_mem) (bottom[0]->mutable_gpu_diff()), &ctx)),
        ctx.get_queue());
#endif  // USE_GREENTEA
  }
}
#ifdef HAS_HALF_SUPPORT
template void LRNLayer<half>::CrossChannelBackward_gpu(
    const vector<Blob<half>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<half>*>& bottom);
#endif
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);

INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
