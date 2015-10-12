#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_CUDA
template<typename Dtype>
__global__ void CopyForward(const int nthreads, const int dims,
                            const Dtype* bottom_a, const bool forward_a,
                            const Dtype* bottom_b, const bool forward_b,
                            Dtype* top, const int num, const int channels_a,
                            const int channels_b, const int* shape_a,
                            const int* shape_b) {
  int pad[6];  // NOLINT(runtime/arrays)
  int tmp_idx[6];  // NOLINT(runtime/arrays)
  int size_a = 1;
  int size_b = 1;

  for (int i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int batch_id = index / ((channels_a + channels_b) * size_a);
    int bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int counter = index;
    for (int i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int channel_id = (index / size_a) % channels_a;
      int aidx = batch_id * channels_a + channel_id;
      for (int i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      top[index] = forward_a ? bottom_a[aidx] : 0;
    } else {
      int channel_id = (index / size_a) % channels_b;
      int bidx = (batch_id * channels_b + channel_id) * size_b;
      int btemp = 1;
      for (int i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      top[index] = forward_b ? bottom_b[bidx] : 0;
    }
  }
}

template<typename Dtype>
__global__ void CopyBackward(const int nthreads, const int dims,
                             Dtype* bottom_a, const bool backward_a,
                             Dtype* bottom_b, const bool backward_b,
                             const Dtype* top, const int num,
                             const int channels_a, const int channels_b,
                             const int* shape_a, const int* shape_b) {
  int pad[6];  // NOLINT(runtime/arrays)
  int tmp_idx[6];  // NOLINT(runtime/arrays)
  int size_a = 1;
  int size_b = 1;

  for (int i = 0; i < dims; ++i) {
    pad[i] = (shape_b[i] - shape_a[i]) / 2;
    size_a *= shape_a[i];
    size_b *= shape_b[i];
  }

  CUDA_KERNEL_LOOP(index, nthreads) {
    int batch_id = index / ((channels_a + channels_b) * size_a);
    int bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)
        / (channels_a * size_a)) % 2;
    int counter = index;
    for (int i = dims - 1; i >= 0; --i) {
      tmp_idx[i] = counter % shape_a[i];
      counter /= shape_a[i];
    }

    if (bottom_id == 0) {
      int channel_id = (index / size_a) % channels_a;
      int aidx = batch_id * channels_a + channel_id;
      for (int i = 0; i < dims; ++i) {
        aidx *= shape_a[i];
        aidx += tmp_idx[i];
      }
      bottom_a[aidx] = backward_a ? top[index] : 0;
    } else {
      int channel_id = (index / size_a) % channels_b;
      int bidx = (batch_id * channels_b + channel_id) * size_b;
      int btemp = 1;
      for (int i = dims - 1; i >= 0; --i) {
        bidx += btemp * (tmp_idx[i] + pad[i]);
        btemp *= shape_b[i];
      }
      bottom_b[bidx] = backward_b ? top[index] : 0;
    }
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void MergeCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  int count = top[0]->count();

  const Dtype* bottom_data_a = bottom[0]->gpu_data();
  const Dtype* bottom_data_b = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int num = bottom[0]->num();
  int spatial_dims = bottom[0]->shape().size() - 2;

  // All channels of both inputs are copied
  int channels_a = bottom[0]->channels();
  int channels_b = bottom[1]->channels();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    CopyForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS) (
        count, spatial_dims, bottom_data_a,
        forward_[0], bottom_data_b,
        forward_[1], top_data, num, channels_a,
        channels_b, shape_a_.gpu_data(), shape_b_.gpu_data());
    CUDA_POST_KERNEL_CHECK
    ;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_->id());

    viennacl::ocl::kernel &oclk_copy_forward = program.get_kernel(
        CL_KERNEL_SELECT("merge_copy_forward"));
    viennacl::ocl::enqueue(
        oclk_copy_forward(count, spatial_dims,
                          WrapHandle((cl_mem) bottom_data_a, &ctx), forward_[0],
                          WrapHandle((cl_mem) bottom_data_b, &ctx), forward_[1],
                          WrapHandle((cl_mem) top_data, &ctx),
                          num, channels_a, channels_b,
                          WrapHandle((cl_mem) (shape_a_.gpu_data()), &ctx),
                          WrapHandle((cl_mem) (shape_b_.gpu_data()), &ctx)),
        ctx.get_queue());
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

  int count = top[0]->count();

  Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_b = bottom[1]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  int num = bottom[0]->num();
  int spatial_dims = bottom[0]->shape().size() - 2;

  // All channels of both inputs are copied
  int channels_a = bottom[0]->channels();
  int channels_b = bottom[1]->channels();

  // Width and height of the smaller input, which should be input 0
  int height_a = bottom[0]->height();
  int width_a = bottom[0]->width();

  int height_b = bottom[1]->height();
  int width_b = bottom[1]->width();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    CopyBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS) (
        count, spatial_dims, bottom_diff_a, backward_[0],
        bottom_diff_b, backward_[1], top_diff, num,
        channels_a, channels_b, shape_a_.gpu_data(), shape_b_.gpu_data());
    CUDA_POST_KERNEL_CHECK
    ;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_->id());

    viennacl::ocl::kernel &oclk_copy_backward = program.get_kernel(
        CL_KERNEL_SELECT("merge_copy_backward"));
    viennacl::ocl::enqueue(
        oclk_copy_backward(count, spatial_dims,
                           WrapHandle((cl_mem) bottom_diff_a, &ctx), backward_[0],
                           WrapHandle((cl_mem) bottom_diff_b, &ctx), backward_[1],
                           WrapHandle((cl_mem) top_diff, &ctx),
                           num, channels_a, channels_b,
                           WrapHandle((cl_mem) (shape_a_.gpu_data()), &ctx),
                           WrapHandle((cl_mem) (shape_b_.gpu_data()), &ctx)),
        ctx.get_queue());
    ctx.get_queue().finish();

#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MergeCropLayer);

}  // namespace caffe
