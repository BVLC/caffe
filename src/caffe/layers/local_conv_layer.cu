#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void crop_loc_patch_kernel(int count, const Dtype *src, int src_w,
                                      int src_h, int src_c, int crop_width,
                                      int crop_height, int w_off, int h_off,
                                      Dtype *local_patch_data) {
  //	int index = blockIdx.x * blockDim.x + threadIdx.x;
  CUDA_KERNEL_LOOP(index, count) {
    int spatial_dim = crop_width * crop_height;
    int channel = index / spatial_dim;
    int offset = index % spatial_dim;
    int height_out = offset / crop_width;
    int width_out = offset % crop_width;

    local_patch_data[(channel * crop_height + height_out) * crop_width +
                     width_out] =
        src[(channel * src_h + (height_out + h_off)) * src_w + width_out +
            w_off];
  }
}
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::crop_loc_patch_gpu(
    const Dtype *src, int src_w, int src_h, int src_c, int crop_width,
    int crop_height, int w_off, int h_off, Dtype *local_patch_data) const {
  // We are going to launch channels * crop_width * crop_height kernels, each
  // kernel responsible for  croping one element
  int num_kernels = src_c * crop_width * crop_height;
  crop_loc_patch_kernel<Dtype>
      <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels, src, src_w, src_h, src_c, crop_width, crop_height, w_off,
          h_off, local_patch_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void
realign_loc_conv_result_kernel2(int count, const Dtype *local_conv_data,
                                int loc_num_h, int loc_num_w, int loc_out_h,
                                int loc_out_w, int num_out, int dst_h,
                                int dst_w, Dtype *dst_data) {
  int loc_spatial_dim = loc_out_h * loc_out_w;
  int dst_spatial_dim = dst_h * dst_w;
  int loc_out_step = loc_spatial_dim * num_out;
  CUDA_KERNEL_LOOP(index, count) {
    int loc_count = index / loc_out_step;
    int loc_out_offset = index % loc_out_step;
    int loc_idx_h = loc_count / loc_num_w;
    int loc_idx_w = loc_count % loc_num_w;
    int c = loc_out_offset / loc_spatial_dim;
    int loc_offset = loc_out_offset % loc_spatial_dim;
    int loc_h = loc_offset / loc_out_w;
    int loc_w = loc_offset % loc_out_w;

    int dst_idx = c * dst_spatial_dim +
                  (loc_idx_h * loc_out_h + loc_h) * dst_w +
                  loc_idx_w * loc_out_w + loc_w;
    dst_data[dst_idx] = local_conv_data[index];
  }
}
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::realign_loc_conv_result_gpu(
    const Dtype *local_conv_data, Dtype *dst_data) const {
  // We are going to launch num_output * height_out * width_out kernels, each
  // kernel responsible for  copying  one local conv result per local region
  // int num_kernels = this->num_output_ * this->output_shape_[0] *
  // this->output_shape_[1]; //for realign_loc_conv_result_kernel()

  auto output_shape = compute_output_shape();
  int top_height = output_shape[0] * this->local_region_num_h_;
  int top_width = output_shape[1] * this->local_region_num_w_;
  int num_kernels = this->num_output_ * output_shape[0] *
                    output_shape[1] *
                    this->L_; // To get bigger size of Block
  realign_loc_conv_result_kernel2<Dtype>
      <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels, local_conv_data, this->local_region_num_h_,
          this->local_region_num_w_, output_shape[0],
          output_shape[1], this->num_output_,top_height,
	  top_width, dst_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  Forward_const_gpu(bottom,top);
}

template <typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_const_gpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) const {
  Dtype *loc_bottom_data = loc_bottom_buffer_ptr_->mutable_gpu_data();
  Dtype *loc_top_data = loc_top_buffer_ptr_->mutable_gpu_data();
  const Dtype *weight = this->blobs_[0]->gpu_data();

  const int *idx_to_off_data = this->loc_idx_to_offset_ptr_->cpu_data();

  int loc_h = this->conv_input_shape_ptr_->cpu_data()[1];
  int loc_w = this->conv_input_shape_ptr_->cpu_data()[2];
  int num=bottom[0]->count(0, this->channel_axis_);
  for (int i = 0; i < bottom.size(); i++) {
    const Dtype *bottom_data = bottom[i]->gpu_data();
    int bottom_w = bottom[i]->width();
    int bottom_h = bottom[i]->height();
    int bottom_c = bottom[i]->channels();
    Dtype *top_data = top[i]->mutable_gpu_data();

    for (int n = 0; n < num; n++) {
      const Dtype *single_bottom_data = bottom_data + bottom[i]->offset(n);
      for (int lh = 0; lh < local_region_num_h_; lh++) {
        for (int lw = 0; lw < local_region_num_w_; lw++) {
          int loc_num = lh * local_region_num_w_ + lw;
          const Dtype *loc_weight = weight + this->blobs_[0]->offset(loc_num);
          Dtype *loc_bottom =
              loc_bottom_data + loc_bottom_buffer_ptr_->offset(loc_num);
          Dtype *loc_top = loc_top_data + loc_top_buffer_ptr_->offset(loc_num);
          crop_loc_patch_gpu(
              single_bottom_data, bottom_w, bottom_h, bottom_c, loc_w, loc_h,
              idx_to_off_data[loc_idx_to_offset_ptr_->offset(lh, lw, 1, 0)],
              idx_to_off_data[loc_idx_to_offset_ptr_->offset(lh, lw, 0, 0)], loc_bottom);
          this->forward_gpu_gemm(loc_bottom, loc_weight, loc_top);
          if (this->bias_term_) {
            const Dtype *bias =
                this->blobs_[1]->gpu_data() + this->blobs_[1]->offset(loc_num);
            this->forward_gpu_bias(loc_top, bias);
          }
        }
      }
      realign_loc_conv_result_gpu(loc_top_data, top_data + top[i]->offset(n));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_CONST(LocalConvolutionLayer);

} // namespace caffe
