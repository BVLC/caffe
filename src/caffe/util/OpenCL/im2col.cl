#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#include "definitions.hpp"

template <class T> __kernel void clim2col(const int n, global T* data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_col) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		
		int w_out 		= idx % width_col;
		int h_idx 		= idx / width_col;
		int h_out 		= h_idx % height_col;
		int channel_in 	= h_idx / height_col;
		int channel_out = channel_in * kernel_h * kernel_w;
		int h_in 		= h_out * stride_h - pad_h;
		int w_in 		= w_out * stride_w - pad_w;
		
		global T* data_col_ptr  = data_col;
		data_col_ptr 			+= (channel_out * height_col + h_out) * width_col + w_out;

		global T* data_im_ptr 	= data_im;
		data_im_ptr 			+= (channel_in * height + h_in) * width + w_in;

		for (int i = 0; i < kernel_h; ++i) {
			int h = h_in + i;
			for (int j = 0; j < kernel_w; ++j) {
				int w = w_in + j;
				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col*width_col;
			}
		}
	}
}
template __attribute__((mangled_name(clim2colFloat))) kernel void clim2col(const int n, global float* data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_col);
template __attribute__((mangled_name(clim2colDouble))) kernel void clim2col(const int n, global double* data_im, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_col);

template <class T> __kernel void clim2col_perf(const int n, global T* data_im, const int bottom_step, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_col, const int top_step) {

	unsigned int gidx = get_global_id(0);
	if ( get_work_dim() == 3 ) {
		gidx = get_global_id(2)*get_global_size(0)*get_global_size(1) + get_global_id(1)*get_global_size(0) + get_global_id(0);
	}	

	if ( gidx < n ) {

		int image_idx	= gidx / (num_channels*height_col*width_col);
		int part_idx	= gidx % (num_channels*height_col*width_col);
		
		int w_out 		= part_idx % width_col;
		int h_idx 		= part_idx / width_col;
		int h_out 		= h_idx % height_col;
		int channel_in 	= h_idx / height_col;
		int channel_out = channel_in * kernel_h * kernel_w;
		int h_in 		= h_out * stride_h - pad_h;
		int w_in 		= w_out * stride_w - pad_w;
		
		global T* data_col_ptr  = data_col;
		data_col_ptr 			+= image_idx*top_step;
		data_col_ptr 			+= (channel_out * height_col + h_out) * width_col + w_out;
		
		global T* data_im_ptr 	= data_im;
		data_im_ptr 			+= image_idx*bottom_step;
		data_im_ptr 			+= (channel_in * height + h_in) * width + w_in;

		for (int i = 0; i < kernel_h; ++i) {
			int h = h_in + i;
			for (int j = 0; j < kernel_w; ++j) {
				int w = w_in + j;
				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col*width_col;
			}
		}
	}
}
template __attribute__((mangled_name(clim2col_perfFloat))) kernel void clim2col_perf(const int n, global float* data_im, const int bottom_step, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_col, const int top_step);
template __attribute__((mangled_name(clim2col_perfDouble))) kernel void clim2col_perf(const int n, global double* data_im, const int bottom_step, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_col, const int top_step);

template <class T> __kernel void clim2col_perf2(const int n, global T* data_im, const long unsigned int data_im_step, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_col, const long unsigned int data_col_step) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		
		int w_out 		= idx % width_col;
		int h_idx 		= idx / width_col;
		int h_out 		= h_idx % height_col;
		int channel_in 	= h_idx / height_col;
		int channel_out = channel_in * kernel_h * kernel_w;
		int h_in 		= h_out * stride_h - pad_h;
		int w_in 		= w_out * stride_w - pad_w;
		
		global T* data_col_ptr  = data_col;
		data_col_ptr			+= data_col_step;
		data_col_ptr 			+= (channel_out * height_col + h_out) * width_col + w_out;

		global T* data_im_ptr 	= data_im;
		data_im_ptr				+= data_im_step;
		data_im_ptr 			+= (channel_in * height + h_in) * width + w_in;

		for (int i = 0; i < kernel_h; ++i) {
			int h = h_in + i;
			for (int j = 0; j < kernel_w; ++j) {
				int w = w_in + j;
				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col*width_col;
			}
		}
	}
}
template __attribute__((mangled_name(clim2col_perf2Float))) kernel void clim2col_perf2(const int n, global float* data_im, const long unsigned int data_im_step, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_col, const long unsigned int data_col_step);
template __attribute__((mangled_name(clim2col_perf2Double))) kernel void clim2col_perf2(const int n, global double* data_im, const long unsigned int data_im_step, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_col, const long unsigned int data_col_step);

template <class T> __kernel void clcol2im(const int n, global T* data_col, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_im) {

	int idx = get_global_id(0);
	if ( idx < n ) {

		T val = 0;
		int w = idx % width + pad_w;
		int h = (idx / width) % height + pad_h;
		int c = idx / (width * height);

		// compute the start and end of the output
		int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
		int w_col_end = min(w / stride_w + 1, width_col);
		int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
		int h_col_end = min(h / stride_h + 1, height_col);

		int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
		int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
		int coeff_w_col = (1 - stride_w * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
		  for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
			val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
		  }
		}
		data_im[idx] = val;
	}
}
template __attribute__((mangled_name(clcol2imFloat))) kernel void clcol2im(const int n, global float* data_col, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_im);
template __attribute__((mangled_name(clcol2imDouble))) kernel void clcol2im(const int n, global double* data_col, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_im);

template <class T> __kernel void clcol2im_perf2(const int n, global T* data_col, const int data_col_step, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_im, const int data_im_step) {

	int idx = get_global_id(0);
	if ( idx < n ) {

		global T* data_col_ptr  = data_col;
		data_col_ptr 			+= data_col_step;

		global T* data_im_ptr 	= data_im;
		data_im_ptr 			+= data_im_step;

		T val = 0;
		int w = idx % width + pad_w;
		int h = (idx / width) % height + pad_h;
		int c = idx / (width * height);

		// compute the start and end of the output
		int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
		int w_col_end = min(w / stride_w + 1, width_col);
		int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
		int h_col_end = min(h / stride_h + 1, height_col);

		int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
		int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
		int coeff_w_col = (1 - stride_w * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
		  for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
			val += data_col_ptr[offset + h_col * coeff_h_col + w_col * coeff_w_col];
		  }
		}
		data_im_ptr[idx] = val;
	}
}
template __attribute__((mangled_name(clcol2im_perf2Float))) kernel void clcol2im_perf2(const int n, global float* data_col, const int data_col_step, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_im, const int data_im_step);
template __attribute__((mangled_name(clcol2im_perf2Double))) kernel void clcol2im_perf2(const int n, global double* data_col, const int data_col_step, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_im, const int data_im_step);


template <class T> __kernel void clcol2im_perf(const int n, global T* data_col, const int top_step, const int col_number, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_im, const int bottom_step) {

	int idx = get_global_id(0);
	if ( idx < n ) {

		int col_idx		= idx / (channels*height*width);
		int part_idx	= idx % (channels*height*width);

		global T* data_col_ptr  = data_col;
		data_col_ptr 			+= col_idx*top_step;

		global T* data_im_ptr 	= data_im;
		data_im_ptr 			+= col_idx*bottom_step;

		T val = 0;
		int w = part_idx % width + pad_w;
		int h = (part_idx / width) % height + pad_h;
		int c = part_idx / (width * height);

		// compute the start and end of the output
		int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
		int w_col_end = min(w / stride_w + 1, width_col);
		int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
		int h_col_end = min(h / stride_h + 1, height_col);

		int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
		int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
		int coeff_w_col = (1 - stride_w * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
		  for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
			val += data_col_ptr[offset + h_col * coeff_h_col + w_col * coeff_w_col];
		  }
		}
		data_im_ptr[part_idx] = val;
	}
}
template __attribute__((mangled_name(clcol2im_perfFloat))) kernel void clcol2im_perf(const int n, global float* data_col, const int top_step, const int col_number, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_im, const int bottom_step);
template __attribute__((mangled_name(clcol2im_perfDouble))) kernel void clcol2im_perf(const int n, global double* data_col, const int top_step, const int col_number, const int height, const int width, const int channels, const int patch_h, const int patch_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_im, const int bottom_step);


template <class T> __kernel void clim2col_perf3(const int n, global T* data_im, const int bottom_step, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_col, const int top_step) {

  int image                 = get_global_id(0)/(width*num_channels);
  int channel               = (get_global_id(0)-image*num_channels*width)/width;
  
  int image_channel_offset  = (image*num_channels + channel)*height*width;
  int col_channel_offset    = (image*num_channels + channel)*height_col*width_col*kernel_w*kernel_h;
  
  global T* im_ptr = data_im;
  im_ptr += image_channel_offset;

  global T* col_ptr = data_col;
  col_ptr += col_channel_offset;
  
  int gw = get_global_id(0)% (image*num_channels*width + channel*width); // horizontal pixel coordinate
  int gh = get_global_id(1);                // vertical pixel coordinate
  
  if( gw < width && gh < height ) { // thread is inside image
    int KW2 = kernel_w/2;
    int KH2 = kernel_h/2;

    int offset_w = KW2 - pad_w;
    int offset_h = KH2 - pad_h;
        
    int mod_w = (gw - offset_w) % stride_w; // if this is zero, the image pixel is in src column
    int div_w = (gw - offset_w) / stride_w; // this is the horizontal index of the target pixel
        
    int mod_h = (gh - offset_h) % stride_h; // if this is zero, the image pixel is in src row 
    int div_h = (gh - offset_h) / stride_h; // this is the vertical index of the target pixel

    __local T buffer[OPENCL_BLOCK_SIZE+2][OPENCL_BLOCK_SIZE+2];

    // each thread copies one pixel into the buffer
    int src = width*gh + gw;
    unsigned int lw = get_local_id(0);
    unsigned int lh = get_local_id(1);
    
    buffer[lh+1][lw+1] = im_ptr[src];
    
    if ( lh == 0 ) {
      if ( gh-1 >= 0 ) {
        src = width*(gh-1) + gw;
        buffer[0][lw+1] = im_ptr[src];
      } else {
        buffer[0][lw+1] = 0;        
      }
    }
    if ( lh == get_local_size(1)-1 ) {
      if ( gh+1 < height ) {
        src = width*(gh+1) + gw;
        buffer[lh+2][lw+1] = im_ptr[src];
      } else {
        buffer[lh+2][lw+1] = 0;        
      }
    }
    if ( lw == 0 ) {
      if ( gw-1 >= 0 ) {
        src = width*gh + gw - 1;
        buffer[lh+1][0] = im_ptr[src];
      } else {
        buffer[lh+1][0] = 0;        
      }
    }
    if ( lw == get_local_size(0)-1 ) {
      if ( gw+1 < width ) {
        src = width*gh + gw + 1;
        buffer[lh+1][lw+2] = im_ptr[src];
      } else {
        buffer[lh+1][lw+2] = 0;        
      }
    }
    if ( lh == 0 && lw == 0 ) {
      if ( gh-1 >= 0 && gw-1 >= 0 ) {
        src = width*(gh-1) + gw - 1;
        buffer[0][0] = im_ptr[src];        
      } else {
        buffer[0][0] = 0;                
      }
    }
    if ( lh == 0 && lw == get_local_size(0)-1 ) {
      if ( gh-1 >= 0 && gw+1 < width ) {
        src = width*(gh-1) + gw + 1;
        buffer[0][lw+2] = im_ptr[src];        
      } else {
        buffer[0][lw+2] = 0;                
      }
    }
    if ( lh == get_local_size(1)-1 && lw == 0 ) {
      if ( gh+1 < height && gw-1 >= 0 ) {
        src = width*(gh+1) + gw - 1;
        buffer[lh+2][0] = im_ptr[src];                
      } else {
        buffer[lh+2][0] = 0;                       
      }
    }
    if ( lh == get_local_size(1)-1 && lw == get_local_size(0)-1 ) {
      if ( gh+1 < height && gw+1 < width ) {
        src = width*(gh+1) + gw + 1;
        buffer[lh+2][lw+2] = im_ptr[src];                
      } else {
        buffer[lh+2][lw+2] = 0;                       
      }
    }

    
    barrier(CLK_LOCAL_MEM_FENCE);
       
    if ( mod_w == 0 && div_w < width_col ) {
      if ( mod_h == 0 && div_h  < height_col ) {

        int bw  = 0;
        int bh  = 0;
        int dst = 0;
        int kernel_src = 0;

        for(int dh = -KH2; dh<= KH2; dh++) {
          for(int dw = -KW2; dw<= KW2; dw++) {
             bw = lw + dw + 1;
             bh = lh + dh + 1;
             dst = height_col*width_col*kernel_src + width_col*div_h + div_w;

             if ( bw >= 0 && bw < OPENCL_BLOCK_SIZE+2 && bh >= 0 && bh < OPENCL_BLOCK_SIZE+2 ) {
               col_ptr[dst] = buffer[bh][bw];
             } else {
               col_ptr[dst] = 0;
             }
             kernel_src++;
           }           
         }
       }
     } 
  }
}
template __attribute__((mangled_name(clim2col_perf3Float))) kernel void clim2col_perf3(const int n, global float* data_im, const int bottom_step, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_col, const int top_step);
template __attribute__((mangled_name(clim2col_perf3Double))) kernel void clim2col_perf3(const int n, global double* data_im, const int bottom_step, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_col, const int top_step);

template <class T> __kernel void clim2col_perf4(const int n, global T* data_im, const int im_offset, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global T* data_col, const int col_offset) {

  int channel               = get_global_id(0)/width;
  
  int image_channel_offset  = channel*height*width;
  int col_channel_offset    = channel*height_col*width_col*kernel_w*kernel_h;
  
  global T* im_ptr = data_im;
  im_ptr += im_offset;
  im_ptr += image_channel_offset;

  global T* col_ptr = data_col;
  col_ptr += col_offset;
  col_ptr += col_channel_offset;
  
  int gw = get_global_id(0)% (channel*width); // horizontal pixel coordinate
  int gh = get_global_id(1);                // vertical pixel coordinate
  
  if( gw < width && gh < height ) { // thread is inside image
    int KW2 = kernel_w/2;
    int KH2 = kernel_h/2;

    int offset_w = KW2 - pad_w;
    int offset_h = KH2 - pad_h;
        
    int mod_w = (gw - offset_w) % stride_w; // if this is zero, the image pixel is in src column
    int div_w = (gw - offset_w) / stride_w; // this is the horizontal index of the target pixel
        
    int mod_h = (gh - offset_h) % stride_h; // if this is zero, the image pixel is in src row 
    int div_h = (gh - offset_h) / stride_h; // this is the vertical index of the target pixel

    __local T buffer[OPENCL_BLOCK_SIZE+2][OPENCL_BLOCK_SIZE+2];

    // each thread copies one pixel into the buffer
    int src = width*gh + gw;
    unsigned int lw = get_local_id(0);
    unsigned int lh = get_local_id(1);
    
    buffer[lh+1][lw+1] = im_ptr[src];
    
    if ( lh == 0 ) {
      if ( gh-1 >= 0 ) {
        src = width*(gh-1) + gw;
        buffer[0][lw+1] = im_ptr[src];
      } else {
        buffer[0][lw+1] = 0;        
      }
    }
    if ( lh == get_local_size(1)-1 ) {
      if ( gh+1 < height ) {
        src = width*(gh+1) + gw;
        buffer[lh+2][lw+1] = im_ptr[src];
      } else {
        buffer[lh+2][lw+1] = 0;        
      }
    }
    if ( lw == 0 ) {
      if ( gw-1 >= 0 ) {
        src = width*gh + gw - 1;
        buffer[lh+1][0] = im_ptr[src];
      } else {
        buffer[lh+1][0] = 0;        
      }
    }
    if ( lw == get_local_size(0)-1 ) {
      if ( gw+1 < width ) {
        src = width*gh + gw + 1;
        buffer[lh+1][lw+2] = im_ptr[src];
      } else {
        buffer[lh+1][lw+2] = 0;        
      }
    }
    if ( lh == 0 && lw == 0 ) {
      if ( gh-1 >= 0 && gw-1 >= 0 ) {
        src = width*(gh-1) + gw - 1;
        buffer[0][0] = im_ptr[src];        
      } else {
        buffer[0][0] = 0;                
      }
    }
    if ( lh == 0 && lw == get_local_size(0)-1 ) {
      if ( gh-1 >= 0 && gw+1 < width ) {
        src = width*(gh-1) + gw + 1;
        buffer[0][lw+2] = im_ptr[src];        
      } else {
        buffer[0][lw+2] = 0;                
      }
    }
    if ( lh == get_local_size(1)-1 && lw == 0 ) {
      if ( gh+1 < height && gw-1 >= 0 ) {
        src = width*(gh+1) + gw - 1;
        buffer[lh+2][0] = im_ptr[src];                
      } else {
        buffer[lh+2][0] = 0;                       
      }
    }
    if ( lh == get_local_size(1)-1 && lw == get_local_size(0)-1 ) {
      if ( gh+1 < height && gw+1 < width ) {
        src = width*(gh+1) + gw + 1;
        buffer[lh+2][lw+2] = im_ptr[src];                
      } else {
        buffer[lh+2][lw+2] = 0;                       
      }
    }

    
    barrier(CLK_LOCAL_MEM_FENCE);
       
    if ( mod_w == 0 && div_w < width_col ) {
      if ( mod_h == 0 && div_h  < height_col ) {

        int bw  = 0;
        int bh  = 0;
        int dst = 0;
        int kernel_src = 0;

        for(int dh = -KH2; dh<= KH2; dh++) {
          for(int dw = -KW2; dw<= KW2; dw++) {
             bw = lw + dw + 1;
             bh = lh + dh + 1;
             dst = height_col*width_col*kernel_src + width_col*div_h + div_w;

             if ( bw >= 0 && bw < OPENCL_BLOCK_SIZE+2 && bh >= 0 && bh < OPENCL_BLOCK_SIZE+2 ) {
               col_ptr[dst] = buffer[bh][bw];
             } else {
               col_ptr[dst] = 0;
             }
             kernel_src++;
           }           
         }
       }
     } 
  }
}
template __attribute__((mangled_name(clim2col_perf4Float))) kernel void clim2col_perf4(const int n, global float* data_im, const int im_offset, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global float* data_col, const int col_offset);
template __attribute__((mangled_name(clim2col_perf4Double))) kernel void clim2col_perf4(const int n, global double* data_im, const int im_offset, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col, global double* data_col, const int col_offset);


template <class T> __kernel void clim2col_mask(global T* data_im, global int* mask, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int height_out, const int width_out, global T* data_col) {
  
  int ppc                   = height_out*width_out*kernel_h*kernel_w;
  int image                 = get_global_id(0)/(ppc*num_channels);
  int channel               = (get_global_id(0)-image*ppc*num_channels)/(ppc);
  
  int image_channel_offset  = (image*num_channels + channel)*height*width;
  int col_channel_offset    = (image*num_channels + channel)*height_out*width_out*kernel_h*kernel_w;
  
  global T* im_ptr = data_im;
  im_ptr += image_channel_offset;

  global T* col_ptr = data_col;
  col_ptr += col_channel_offset;
  
  int pixel = get_global_id(0) % ppc;

  if ( pixel < ppc ) {
    col_ptr[pixel] = im_ptr[ mask[pixel] ];
  }
}
template __attribute__((mangled_name(clim2col_maskFloat))) kernel void clim2col_mask(global float* data_im, global int* mask, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int height_out, const int width_out, global float* data_col);
template __attribute__((mangled_name(clim2col_maskDouble))) kernel void clim2col_mask(global double* data_im, global int* mask, const int num_images, const int num_channels, const int height, const int width, const int kernel_h, const int kernel_w, const int height_out, const int width_out, global double* data_col);
    
        
        