#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>

namespace caffe {



template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col,int* all_zero_mask, int * feature_map_mask) {
  //get zero and nonzero maps
	vector<Dtype> activation_sum = vector<Dtype>(channels);
	int kernel_slice_dim = kernel_w*kernel_h;
	if(all_zero_mask && feature_map_mask){
		/*
		  for(int im_c=0;im_c<channels;++im_c){
			  activation_sum[im_c]=caffe_cpu_asum(height*width,data_im+im_c*height*width);
		  }
		  int kth = channels/4;
		  std::nth_element (activation_sum.begin(), activation_sum.begin()+kth, activation_sum.end());
		  for(int im_c=0;im_c<channels;++im_c){
			  feature_map_mask[im_c] =  activation_sum[im_c]>=1;
			  caffe_set(kernel_slice_dim,1-feature_map_mask[im_c],all_zero_mask+kernel_slice_dim*im_c);
		  }*/


		  Dtype thre = 1;
		  for(int im_c=0;im_c<channels;++im_c){
			  feature_map_mask[im_c] = 0;
			  int inner_flag = 0;
			  for(int im_h=0;im_h<height;++im_h){
				  if(inner_flag) break;
				  for(int im_w=0;im_w<width;++im_w){
					  if(abs(data_im[(im_c * height + im_h) * width + im_w])>thre){
						  feature_map_mask[im_c] = 1;
						  inner_flag = 1;
						  break;
					  }
				  }
			  }
			  caffe_set(kernel_slice_dim,1-feature_map_mask[im_c],all_zero_mask+kernel_slice_dim*im_c);
		  }
	}

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  int forward_count = 0;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    //int sum=0;
    //for(int ii=0;ii<forwarding_mask.size();ii++){
    //	sum+=forwarding_mask[ii];
    //}
    if(all_zero_mask && feature_map_mask && !feature_map_mask[c_im]) {
    	continue;
    }
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width){
          //data_col[(c * height_col + h) * width_col + w] =
        	data_col[(forward_count * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        }
        else{
          //data_col[(c * height_col + h) * width_col + w] = 0;
        	data_col[(forward_count * height_col + h) * width_col + w] = 0;
        }
      }
    }
    forward_count++;
  }

  //free(activation_sum);
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col,int* all_zero_mask, int * feature_map_mask);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col,int* all_zero_mask, int * feature_map_mask);

/*
// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col,std::vector<int> forwarding_mask);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col,std::vector<int> forwarding_mask);


template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col,int *all_zero_mask) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  int forward_count = 0;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    //int sum=0;
    //for(int ii=0;ii<forwarding_mask.size();ii++){
    //	sum+=forwarding_mask[ii];
    //}
    //if(!forwarding_mask.empty() && c_im>=sum) {
    //	continue;
    //}

    //first check if elements in current row are all zeros
    int h_pad = 0;
    int w_pad = 0;
    bool inner_break = false;
    if(all_zero_mask){
		all_zero_mask[c] = true;
		for (int h = 0; h < height_col; ++h) {
		  if(inner_break) {
			  break;
		  }
		  for (int w = 0; w < width_col; ++w) {
			  h_pad = h * stride_h - pad_h + h_offset;
			  w_pad = w * stride_w - pad_w + w_offset;
			  if( h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width && abs(data_im[(c_im * height + h_pad) * width + w_pad])>0 ) {
				  all_zero_mask[c] = false;
				  inner_break = true; break;
			  }
		  }
		}
    }
    if(all_zero_mask[c]){
    	all_zero_mask[c] = true;
    }

    //fill data_col and skip all zero rows
    if(!all_zero_mask || !all_zero_mask[c]){
		for (int h = 0; h < height_col; ++h) {
		  for (int w = 0; w < width_col; ++w) {
			h_pad = h * stride_h - pad_h + h_offset;
			w_pad = w * stride_w - pad_w + w_offset;
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width){
			  //data_col[(c * height_col + h) * width_col + w] =
				data_col[(forward_count * height_col + h) * width_col + w] =
				data_im[(c_im * height + h_pad) * width + w_pad];
			}
			else{
			  //data_col[(c * height_col + h) * width_col + w] = 0;
				data_col[(forward_count * height_col + h) * width_col + w] = 0;
			}
		  }
		}
		forward_count++;
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col,int* all_zero_mask);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col,int* all_zero_mask);
*/
template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

}  // namespace caffe
