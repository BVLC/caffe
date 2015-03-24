#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


template <class T> __kernel void LRNFillScale(const int nthreads, const global T* in, const int num, const int channels, const int height, const int width, const int size, const T alpha_over_size, const T k, global T* scale) {

	int idx = get_global_id(0);
	if ( idx < nthreads ) {
		
    // find out the local offset
    int w = idx % width;
    int h = (idx / width) % height;
    int n = idx / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    T accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      if (head - size >= 0) {
        accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      }
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      }
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }		
	}
}
template __attribute__((mangled_name(LRNFillScaleFloat))) kernel void LRNFillScale(const int nthreads, const global float* in, const int num, const int channels, const int height, const int width, const int size, const float alpha_over_size, const float k, global float* scale);
template __attribute__((mangled_name(LRNFillScaleDouble))) kernel void LRNFillScale(const int nthreads, const global double* in, const int num, const int channels, const int height, const int width, const int size, const double alpha_over_size, const double k, global double* scale);

// TODO: check if it would be faster to just put it into the previous kernel.
template <class T> __kernel void LRNComputeOutput(const int nthreads, const global T* in, const global T* scale, const T negative_beta, global* out) {
	
	int idx = get_global_id(0);
	if ( idx < nthreads ) {
	    out[idx] = in[idx] * pow(scale[idx], negative_beta);		
	}
}
template __attribute__((mangled_name(LRNComputeOutputFloat))) kernel void LRNComputeOutput(const int nthreads, const global float* in, const global float* scale, const float negative_beta, global float* out);
template __attribute__((mangled_name(LRNComputeOutputDouble))) kernel void LRNComputeOutput(const int nthreads, const global double* in, const global double* scale, const double negative_beta, global double* out);


template <class T> __kernel void LRNComputeDiff(const int nthreads, const global T* bottom_data, const global T* top_data, const global T* scale, const global T* top_diff, const int num, const int channels, const int height, const int width, const int size, const T negative_beta, const T cache_ratio, global T* bottom_diff) {
	
	int idx = get_global_id(0);
	if ( idx < nthreads ) {

    // find out the local offset
    int w = idx % width;
    int h = (idx / width) % height;
    int n = idx / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    T accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff[(head - size) * step] *
            top_data[(head - size) * step] / scale[(head - size) * step];
      }
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff[(head - size) * step] *
            top_data[(head - size) * step] / scale[(head - size) * step];
      }
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }

	
	}
}
template __attribute__((mangled_name(LRNComputeDiffFloat))) kernel void LRNComputeDiff(const int nthreads, const global float* bottom_data, const global float* top_data, const global float* scale, const global float* top_diff, const int num, const int channels, const int height, const int width, const int size, const float negative_beta, const float cache_ratio, global float* bottom_diff);
template __attribute__((mangled_name(LRNComputeDiffDouble))) kernel void LRNComputeDiff(const int nthreads, const global double* bottom_data, const global double* top_data, const global double* scale, const global double* top_diff, const int num, const int channels, const int height, const int width, const int size, const double negative_beta, const double cache_ratio, global double* bottom_diff);
