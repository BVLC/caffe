#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/local_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
__global__ void crop_loc_patch_kernel(int count, const Dtype *src, int src_w, int src_h, int src_c, int crop_width, int crop_height, int w_off, int h_off, Dtype *local_patch_data)
{
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
	CUDA_KERNEL_LOOP(index, count){
		int spatial_dim = crop_width * crop_height;
		int channel = index / spatial_dim;
		int offset = index % spatial_dim;
		int height_out = offset / crop_width;
		int width_out = offset % crop_width;

		local_patch_data[(channel * crop_height + height_out) * crop_width + width_out] = 
			src[(channel * src_h + (height_out + h_off)) * src_w + width_out + w_off];
	}
}
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::crop_loc_patch_gpu(const Dtype *src
		, int src_w, int src_h, int src_c
		, int crop_width, int crop_height
		, int w_off, int h_off
		, Dtype *local_patch_data)
{
	//We are going to launch channels * crop_width * crop_height kernels, each kernel responsible for 
	//croping one element
	int num_kernels = src_c * crop_width * crop_height;
	crop_loc_patch_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
		num_kernels
		, src
		, src_w, src_h, src_c
		, crop_width, crop_height
		, w_off, h_off
		, local_patch_data);
	CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void realign_loc_conv_result_kernel2(int count, const Dtype *local_conv_data
		, int loc_num_h, int loc_num_w
		, int loc_out_h, int loc_out_w
		, int num_out
		, int dst_h, int dst_w
		, Dtype *dst_data)
{
	int loc_spatial_dim = loc_out_h * loc_out_w;
	int dst_spatial_dim = dst_h * dst_w;
	int loc_num = loc_num_h * loc_num_w;
	int loc_out_step = loc_spatial_dim * num_out;
	CUDA_KERNEL_LOOP(index, count){
		int loc_count = index / loc_out_step;
		int loc_out_offset = index % loc_out_step;
		int loc_idx_h = loc_count / loc_num_w;
		int loc_idx_w = loc_count % loc_num_w;
		int c = loc_out_offset / loc_spatial_dim;
		int loc_offset = loc_out_offset % loc_spatial_dim;
		int loc_h = loc_offset / loc_out_w;
		int loc_w = loc_offset % loc_out_w;

		int dst_idx = c * dst_spatial_dim + (loc_idx_h * loc_out_h + loc_h) * dst_w + loc_idx_w * loc_out_w + loc_w;
		dst_data[dst_idx] = local_conv_data[index];
	}
}
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::realign_loc_conv_result_gpu(const Dtype *local_conv_data, Dtype *dst_data)
{
	//We are going to launch num_output * height_out * width_out kernels, each kernel responsible for 
	//copying  one local conv result per local region
  //int num_kernels = this->num_output_ * this->output_shape_[0] * this->output_shape_[1]; //for realign_loc_conv_result_kernel()
  int num_kernels = this->num_output_ * this->output_shape_[0] * this->output_shape_[1] * this->L_; //To get bigger size of Block
  realign_loc_conv_result_kernel2<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >>>(
	num_kernels, local_conv_data
	, this->local_region_num_h_ , this->local_region_num_w_
	, this->output_shape_[0], this->output_shape_[1]
	, this->num_output_
	, this->top_height_, this->top_width_
	, dst_data);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void realign_bottoom_diff_kernel(int count, const Dtype *loc_bottom_diff
	, int loc_region_h, int loc_region_w
	, int loc_num_h, int loc_num_w
	, int channels
	, int dst_height, int dst_width
	, const int *loc_idx_to_off_data
	, Dtype *dst_data)
{
	int loc_spatial_dim = loc_region_h * loc_region_w;
	int loc_num = loc_num_h * loc_num_w;
	int loc_step = channels * loc_spatial_dim;

	CUDA_KERNEL_LOOP(index, count){
		int b_c = index / loc_spatial_dim;
		int offset = index % loc_spatial_dim;
		int loc_h = offset / loc_region_w;
		int loc_w = offset % loc_region_w;
		
		int loc_offset = b_c * loc_spatial_dim + loc_h * loc_region_w + loc_w;
		int dst_c_offset = b_c * dst_height * dst_width;
		for (int i = 0; i < loc_num; ++i){
			int loc_idx_h = i / loc_num_w;
			int loc_idx_w = i % loc_num_w;

			int src_idx = loc_offset + i * loc_step;
			int loc_idx_to_off_index = (loc_idx_h * loc_num_w + loc_idx_w) * 2;
			int dst_idx = dst_c_offset + (loc_idx_to_off_data[loc_idx_to_off_index] + loc_h) * dst_width
				+ loc_idx_to_off_data[loc_idx_to_off_index + 1] + loc_w;
			
			dst_data[dst_idx] += loc_bottom_diff[src_idx];
		}
	}
}
template <typename Dtype>
void LocalConvolutionLayer<Dtype>::realign_bottom_diff_gpu(const Dtype *loc_bottom_diff_buffer, Dtype *bottom_diff)
{
	//We are going to launch channels * loc_region_h * loc_region_w kernels, each kernel responsible for 
	//aligning  one local bottom diff per local region
  int conv_input_h = this->conv_input_shape_.cpu_data()[1];
  int conv_input_w = this->conv_input_shape_.cpu_data()[2];
  int conv_input_c = this->conv_input_shape_.cpu_data()[0];
  int num_kernels = conv_input_c * conv_input_h * conv_input_w;

  realign_bottoom_diff_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >>>(
	num_kernels
	, loc_bottom_diff_buffer
	, conv_input_h, conv_input_w
	, this->local_region_num_h_, this->local_region_num_w_
	, conv_input_c
	, this->bottom_height_, this->bottom_width_
	, this->loc_idx_to_offset_.gpu_data()
	, bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void LocalConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	Dtype *loc_bottom_data = loc_bottom_buffer_.mutable_gpu_data();
	Dtype* loc_top_data = loc_top_buffer_.mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();

	const Blob<int> *idx_to_off = &this->loc_idx_to_offset_;
	const int *idx_to_off_data = idx_to_off->cpu_data();
	
	int loc_h = this->conv_input_shape_.cpu_data()[1];
	int loc_w = this->conv_input_shape_.cpu_data()[2];
	for (int i = 0; i < bottom.size(); i++)
	{
		const Dtype* bottom_data = bottom[i]->gpu_data();
		int bottom_w = bottom[i]->width();
		int bottom_h = bottom[i]->height();
		int bottom_c = bottom[i]->channels();
		Dtype* top_data = top[i]->mutable_gpu_data();

		for (int n = 0; n < this->num_; n++) {
			const Dtype *single_bottom_data = bottom_data + bottom[i]->offset(n);
			for (int lh = 0; lh < local_region_num_h_; lh++){
				for (int lw = 0; lw < local_region_num_w_; lw++){
					int loc_num = lh * local_region_num_w_ + lw;
					const Dtype* loc_weight = weight + this->blobs_[0]->offset(loc_num);
					Dtype *loc_bottom = loc_bottom_data + loc_bottom_buffer_.offset(loc_num);
					Dtype *loc_top = loc_top_data + loc_top_buffer_.offset(loc_num);
					crop_loc_patch_gpu(single_bottom_data
						, bottom_w
						, bottom_h
						, bottom_c
						, loc_w
						, loc_h
						, idx_to_off_data[idx_to_off->offset(lh, lw, 1, 0)]
						, idx_to_off_data[idx_to_off->offset(lh, lw, 0, 0)]
						, loc_bottom);
					this->forward_gpu_gemm(loc_bottom, loc_weight, loc_top);
					if (this->bias_term_) {
						const Dtype* bias = this->blobs_[1]->gpu_data() + this->blobs_[1]->offset(loc_num);
						this->forward_gpu_bias(loc_top, bias);
					}
				}
			}
			realign_loc_conv_result_gpu(loc_top_data, top_data + top[i]->offset(n));
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LocalConvolutionLayer);

}//namespace caffe
