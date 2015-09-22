#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void Im2colLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num_kernels = channels_ * top[0]->count(channel_axis_ + 1);

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int n = 0; n < num_; ++n) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        im2col_gpu(bottom_data + n * bottom_dim_, channels_,
                   bottom[0]->shape(channel_axis_ + 1),
                   bottom[0]->shape(channel_axis_ + 2),
                   kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                   top_data + n * top_dim_);
      } else {
        im2col_nd_gpu(bottom_data + n * bottom_dim_, num_spatial_axes_,
                      num_kernels, bottom[0]->gpu_shape() + channel_axis_,
                      top[0]->gpu_shape() + channel_axis_,
                      kernel_shape_.gpu_data(), pad_.gpu_data(),
                      stride_.gpu_data(), top_data + n * top_dim_);
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_->id());

    for (int n = 0; n < num_; ++n) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        greentea_im2col_gpu(&program, &ctx, (cl_mem) bottom_data,
                            n * bottom_dim_, channels_,
                            bottom[0]->shape(channel_axis_ + 1),
                            bottom[0]->shape(channel_axis_ + 2),
                            kernel_shape_.cpu_data()[0],
                            kernel_shape_.cpu_data()[1], pad_.cpu_data()[0],
                            pad_.cpu_data()[1], stride_.cpu_data()[0],
                            stride_.cpu_data()[1], (cl_mem) top_data,
                            n * top_dim_);
      } else {
        greentea_im2col_nd_gpu(&program, &ctx, (cl_mem) bottom_data,
                               n * bottom_dim_, num_spatial_axes_, num_kernels,
                               bottom[0]->gpu_shape() + channel_axis_,
                               top[0]->gpu_shape() + channel_axis_,
                               kernel_shape_.gpu_data(), pad_.gpu_data(),
                               stride_.gpu_data(), (cl_mem) top_data,
                               n * top_dim_);
      }
    }

    for (int n = 0; n < bottom[0]->num(); ++n) {
      greentea_im2col_gpu<Dtype>(&program, &ctx, (cl_mem) bottom_data,
                                 bottom[0]->offset(n), channels_, height_,
                                 width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                                 stride_h_, stride_w_, (cl_mem) top_data,
                                 top[0]->offset(n));
    }
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void Im2colLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    for (int n = 0; n < num_; ++n) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        col2im_gpu(top_diff + n * top_dim_, channels_,
                   bottom[0]->shape(channel_axis_ + 1),
                   bottom[0]->shape(channel_axis_ + 2),
                   kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                   bottom_diff + n * bottom_dim_);
      } else {
        col2im_nd_gpu(top_diff + n * top_dim_, num_spatial_axes_, bottom_dim_,
                      bottom[0]->gpu_shape() + channel_axis_,
                      top[0]->gpu_shape() + channel_axis_,
                      kernel_shape_.gpu_data(), pad_.gpu_data(),
                      stride_.gpu_data(), bottom_diff + n * bottom_dim_);
      }
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_context_->id());
    viennacl::ocl::program &program = Caffe::Get().GetDeviceProgram(
        this->device_context_->id());

    for (int n = 0; n < top[0]->num(); ++n) {
      if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
        greentea_col2im_gpu<Dtype>(&program, &ctx, (cl_mem) top_diff,
                                   n * top_dim_, channels_,
                                   bottom[0]->shape(channel_axis_ + 1),
                                   bottom[0]->shape(channel_axis_ + 2),
                                   kernel_shape_.cpu_data()[0],
                                   kernel_shape_.cpu_data()[1],
                                   pad_.cpu_data()[0], pad_.cpu_data()[1],
                                   stride_.cpu_data()[0], stride_.cpu_data()[1],
                                   (cl_mem) bottom_diff, n * bottom_dim_);
      } else {
        greentea_col2im_nd_gpu(&program, &ctx, (cl_mem) top_diff, n * top_dim_,
                               num_spatial_axes_, channel_axis_, bottom_dim_,
                               (cl_mem)(bottom[0]->gpu_shape()),
                               (cl_mem)(top[0]->gpu_shape()),
                               (cl_mem)(kernel_shape_.gpu_data()),
                               (cl_mem)(pad_.gpu_data()),
                               stride_.gpu_data(), (cl_mem) bottom_diff,
                               n * bottom_dim_);
      }
    }
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Im2colLayer);

}  // namespace caffe
