#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void MaxPoolForward(const int nthreads, global T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h,	const int kernel_w,	const int stride_h,	const int stride_w,	const int pad_h, const int pad_w, global T* top_data, global int* mask, global T* top_mask) {
	
	unsigned int gidx = get_global_id(0);
	if ( get_work_dim() == 3 ) {
		gidx = get_global_id(2)*get_global_size(0)*get_global_size(1) + get_global_id(1)*get_global_size(0) + get_global_id(0);
	}	
	
	if ( gidx < nthreads ) {
		int pw = gidx % pooled_width;
		int ph = (gidx / pooled_width) % pooled_height;
		int c = (gidx / pooled_width / pooled_height) % channels;
		int n = gidx / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height);
		int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		T maxval = -FLT_MAX;
		int maxidx = -1;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				if (bottom_data[h * width + w] > maxval) {
					maxidx = h * width + w;
					maxval = bottom_data[maxidx];
				}
			}
		}
		top_data[gidx] = maxval;
		if ( mask ) {
			mask[gidx] = maxidx;
		} else {
			top_mask[gidx] = maxidx;
		}
	}
}
template __attribute__((mangled_name(MaxPoolForwardFloat))) kernel void MaxPoolForward(const int nthreads, global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, global float* top_data, global int* mask, global float* top_mask);
template __attribute__((mangled_name(MaxPoolForwardDouble))) kernel void MaxPoolForward(const int nthreads, global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, global double* top_data, global int* mask, global double* top_mask);

template <class T> __kernel void AvePoolForward(const int nthreads, global T* bottom_data, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h,	const int kernel_w,	const int stride_h,	const int stride_w,	const int pad_h, const int pad_w, global T* top_data) {
	
	unsigned int gidx = get_global_id(0);
	if ( get_work_dim() == 3 ) {
		gidx = get_global_id(2)*get_global_size(0)*get_global_size(1) + get_global_id(1)*get_global_size(0) + get_global_id(0);
	}	
	
	if ( gidx < nthreads ) {
		int pw = gidx % pooled_width;
		int ph = (gidx / pooled_width) % pooled_height;
		int c = (gidx / pooled_width / pooled_height) % channels;
		int n = gidx / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
		int pool_size = (hend - hstart) * (wend - wstart);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		T aveval = 0;
		bottom_data += (n * channels + c) * height * width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				aveval += bottom_data[h * width + w];
			}
		}
		top_data[gidx] = aveval / pool_size;
	}
}
template __attribute__((mangled_name(AvePoolForwardFloat))) kernel void AvePoolForward(const int nthreads, global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, global float* top_data);
template __attribute__((mangled_name(AvePoolForwardDouble))) kernel void AvePoolForward(const int nthreads, global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, global double* top_data);

template <class T> __kernel void StoPoolForwardTrain(const int nthreads, global T* bottom_data, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global T* rand_idx, global T* top_data) {
	
	unsigned int gidx = get_global_id(0);

	if ( gidx < nthreads ) {
		int pw = gidx % pooled_width;
		int ph = (gidx / pooled_width) % pooled_height;
		int c = (gidx / pooled_width / pooled_height) % channels;
		int n = gidx / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h;
		int hend = min(hstart + kernel_h, height);
		int wstart = pw * stride_w;
		int wend = min(wstart + kernel_w, width);
		float cumsum = 0.;
		bottom_data += (n * channels + c) * height * width;
	
		// First pass: get sum
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				cumsum += bottom_data[h * width + w];
			}
		}
		float thres = rand_idx[gidx] * cumsum;
		
		// Second pass: get value, and set gidx.
		cumsum = 0;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				cumsum += bottom_data[h * width + w];
				if (cumsum >= thres) {
					rand_idx[gidx] = ((n * channels + c) * height + h) * width + w;
					top_data[gidx] = bottom_data[h * width + w];
					return;
				}
			}
		}
	}
}
template __attribute__((mangled_name(StoPoolForwardTrainFloat))) kernel void StoPoolForwardTrain(const int nthreads, global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global float* rand_idx, global float* top_data);
template __attribute__((mangled_name(StoPoolForwardTrainDouble))) kernel void StoPoolForwardTrain(const int nthreads, global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global double* rand_idx, global double* top_data);

template <class T> __kernel void StoPoolForwardTest(const int nthreads, global T* bottom_data, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,	global T* top_data) {
	
	unsigned int gidx = get_global_id(0);

	if ( gidx < nthreads ) {
		int pw = gidx % pooled_width;
		int ph = (gidx / pooled_width) % pooled_height;
		int c = (gidx / pooled_width / pooled_height) % channels;
		int n = gidx / pooled_width / pooled_height / channels;
		int hstart = ph * stride_h;
		int hend = min(hstart + kernel_h, height);
		int wstart = pw * stride_w;
		int wend = min(wstart + kernel_w, width);
		// We set cumsum to be 0 to avoid divide-by-zero problems
		float cumsum = FLT_MIN;
		float cumvalues = 0.;
		bottom_data += (n * channels + c) * height * width;
		// First pass: get sum
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				cumsum += bottom_data[h * width + w];
				cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
			}
		}
		top_data[gidx] = cumvalues / cumsum;
	}
}
template __attribute__((mangled_name(StoPoolForwardTestFloat))) kernel void StoPoolForwardTest(const int nthreads, global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global float* top_data);
template __attribute__((mangled_name(StoPoolForwardTestDouble))) kernel void StoPoolForwardTest(const int nthreads, global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global double* top_data);

template <class T> __kernel void MaxPoolBackward(const int nthreads, global T* top_diff, global int* mask, global T* top_mask, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h,	const int kernel_w,	const int stride_h,	const int stride_w,	const int pad_h, const int pad_w, global T* bottom_diff) {
	
	unsigned int gidx = get_global_id(0);
	
	if ( gidx < nthreads ) {
		// find out the local index
		// find out the local offset
		int w = gidx % width;
		int h = (gidx / width) % height;
		int c = (gidx / width / height) % channels;
		int n = gidx / width / height / channels;
		int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
		int phend 	= min((h + pad_h) / stride_h + 1, pooled_height);
		int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
		int pwend 	= min((w + pad_w) / stride_w + 1, pooled_width);
		
		T gradient = 0.0;
		int offset = (n * channels + c) * pooled_height * pooled_width;
		top_diff += offset;
		if (mask) {
			mask += offset;
			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					if (mask[ph * pooled_width + pw] == h * width + w) {
						gradient += top_diff[ph * pooled_width + pw];
					}
				}
			}
		} else {
			top_mask += offset;
			for (int ph = phstart; ph < phend; ++ph) {
				for (int pw = pwstart; pw < pwend; ++pw) {
					if (top_mask[ph * pooled_width + pw] == h * width + w) {
						gradient += top_diff[ph * pooled_width + pw];
					}
				}
			}
		}
		bottom_diff[gidx] = gradient;
	}
}
template __attribute__((mangled_name(MaxPoolBackwardFloat))) kernel void MaxPoolBackward(const int nthreads, global float* top_diff, global int* mask, global float* top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, global float* bottom_diff);
template __attribute__((mangled_name(MaxPoolBackwardDouble))) kernel void MaxPoolBackward(const int nthreads, global double* top_diff, global int* mask, global double* top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, global double* bottom_diff);
		
template <class T> __kernel void AvePoolBackward(const int nthreads, global T* top_diff, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h,	const int kernel_w,	const int stride_h,	const int stride_w,	const int pad_h, const int pad_w, global T* bottom_diff) {
	
	unsigned int gidx = get_global_id(0);

	if ( gidx < nthreads ) {
		// find out the local gidx
		// find out the local offset
		int w = gidx % width + pad_w;
		int h = (gidx / width) % height + pad_h;
		int c = (gidx / width / height) % channels;
		int n = gidx / width / height / channels;
		int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
		int phend = min(h / stride_h + 1, pooled_height);
		int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
		int pwend = min(w / stride_w + 1, pooled_width);
		float gradient = 0;
		top_diff += (n * channels + c) * pooled_height * pooled_width;
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				// figure out the pooling size
				int hstart = ph * stride_h - pad_h;
				int wstart = pw * stride_w - pad_w;
				int hend = min(hstart + kernel_h, height + pad_h);
				int wend = min(wstart + kernel_w, width + pad_w);
				int pool_size = (hend - hstart) * (wend - wstart);
				gradient += top_diff[ph * pooled_width + pw] / pool_size;
			}
		}
		bottom_diff[gidx] = gradient;
	}
}
template __attribute__((mangled_name(AvePoolBackwardFloat))) kernel void AvePoolBackward(const int nthreads, global float* top_diff, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h,	const int kernel_w,	const int stride_h,	const int stride_w,	const int pad_h, const int pad_w, global float* bottom_diff);
template __attribute__((mangled_name(AvePoolBackwardDouble))) kernel void AvePoolBackward(const int nthreads, global double* top_diff, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h,	const int kernel_w,	const int stride_h,	const int stride_w,	const int pad_h, const int pad_w, global double* bottom_diff);

template <class T> __kernel void StoPoolBackward( const int nthreads, global T* rand_idx, global T* top_diff, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global T* bottom_diff) {
	
	unsigned int gidx = get_global_id(0);

	if ( gidx < nthreads ) {
		// find out the local gidx
		// find out the local offset
		int w = gidx % width;
		int h = (gidx / width) % height;
		int c = (gidx / width / height) % channels;
		int n = gidx / width / height / channels;
		int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
		int phend = min(h / stride_h + 1, pooled_height);
		int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
		int pwend = min(w / stride_w + 1, pooled_width);
		float gradient = 0;
		rand_idx += (n * channels + c) * pooled_height * pooled_width;
		top_diff += (n * channels + c) * pooled_height * pooled_width;
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				gradient += top_diff[ph * pooled_width + pw] * (gidx == convert_int(rand_idx[ph * pooled_width + pw]));
			}
		}
		bottom_diff[gidx] = gradient;
	}
}
template __attribute__((mangled_name(StoPoolBackwardFloat))) kernel void StoPoolBackward(const int nthreads, global float* rand_idx, global float* top_diff, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global float* bottom_diff);
template __attribute__((mangled_name(StoPoolBackwardDouble))) kernel void StoPoolBackward(const int nthreads, global double* rand_idx, global double* top_diff, const int num, const int channels,	const int height, const int width, const int pooled_height, const int pooled_width,	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, global double* bottom_diff);
