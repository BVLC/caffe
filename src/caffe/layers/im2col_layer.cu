#include <vector>

#include "caffe/layers/im2col_layer.hpp"
#include "caffe/util/im2col.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void Im2colLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                       const vector<Blob<MItype>*>& bottom,
                                       const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp num_kernels = channels_ * top[0]->count(channel_axis_ + 1);

  for (int_tp n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      this->device_->template im2col<Dtype>(
          bottom_data + n * bottom_dim_, channels_,
          bottom[0]->shape(channel_axis_ + 1),
          bottom[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          top_data + n * top_dim_);
    } else {
      this->device_->template im2col_nd<Dtype>(
          bottom_data + n * bottom_dim_, num_spatial_axes_,
          num_kernels, bottom[0]->gpu_shape() + channel_axis_,
          top[0]->gpu_shape() + channel_axis_,
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), top_data + n * top_dim_);
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void Im2colLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                       const vector<Blob<MOtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<MItype>*>& bottom) {
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();

  for (int n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      this->device_->template col2im<Dtype>(
                     top_diff + n * top_dim_, channels_,
                     bottom[0]->shape(channel_axis_ + 1),
                     bottom[0]->shape(channel_axis_ + 2),
                     kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                     pad_.cpu_data()[0], pad_.cpu_data()[1],
                     stride_.cpu_data()[0], stride_.cpu_data()[1],
                     dilation_.cpu_data()[0], dilation_.cpu_data()[1],
                     bottom_diff + n * bottom_dim_);
    } else {
      this->device_->template col2im_nd<Dtype>(
                     top_diff + n * top_dim_, num_spatial_axes_, bottom_dim_,
                     bottom[0]->gpu_shape() + channel_axis_,
                     top[0]->gpu_shape() + channel_axis_,
                     kernel_shape_.gpu_data(), pad_.gpu_data(),
                     stride_.gpu_data(), dilation_.gpu_data(),
                     bottom_diff + n * bottom_dim_);
    }
  }
}


INSTANTIATE_CLASST_FUNC_3T_GUARDED(Im2colLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(Im2colLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(Im2colLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(Im2colLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(Im2colLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(Im2colLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
