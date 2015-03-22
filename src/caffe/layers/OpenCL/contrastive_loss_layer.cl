#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void CLLForward(const int count, const int channels, const T margin, const T alpha, global T* y, global T* diff, global T* dist_sq, global T *bottom_diff) {
	
	int idx = get_global_id(0);
		
	if ( idx < count ) {
		int n = idx / channels;  // the num index, to access y and dist_sq
		if ( (int) y[n] ) {  // similar pairs
			bottom_diff[idx] = alpha * diff[idx];
		} else {  // dissimilar pairs
			if ((margin-dist_sq[n]) > 0.0) {
				bottom_diff[idx] = -alpha * diff[idx];
			} else {
				bottom_diff[idx] = 0;
			}
		}
	}	
}
template __attribute__((mangled_name(CLLForwardFloat))) kernel void CLLForward(const int count, const int channels, const float margin, const float alpha, global float* y, global float* diff, global float* dist_sq, global float *bottom_diff)
template __attribute__((mangled_name(CLLForwardDouble))) kernel void CLLForward(const int count, const int channels, const double margin, const double alpha, global double* y, global double* diff, global double* dist_sq, global double *bottom_diff)
