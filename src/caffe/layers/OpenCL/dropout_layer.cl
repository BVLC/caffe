#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void DropoutForward(const int n, global T* in, global unsigned int* mask, const unsigned int threshold, const T scale, global T* out) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		out[idx] = in[idx] * (mask[idx] > threshold) * scale;
	}
}
template __attribute__((mangled_name(DropoutForwardFloat))) kernel void DropoutForward(const int n, global float* in, global unsigned int* mask, const unsigned int threshold, const float scale, global float* out);
template __attribute__((mangled_name(DropoutForwardDouble))) kernel void DropoutForward(const int n, global double* in, global unsigned int* mask, const unsigned int threshold, const double scale, global double* out);

template <class T> __kernel void DropoutBackward(const int n, global T* in_diff, global unsigned int* mask, const unsigned int threshold, const T scale, global T* out_diff) {

	int idx = get_global_id(0);
	if ( idx < n ) {
	    out_diff[idx] = in_diff[idx] * scale * (mask[idx] > threshold);
	}
}
template __attribute__((mangled_name(DropoutBackwardFloat))) kernel void DropoutBackward(const int n, global float* in_diff, global unsigned int* mask, const unsigned int threshold, const float scale, global float* out_diff);
template __attribute__((mangled_name(DropoutBackwardDouble))) kernel void DropoutBackward(const int n, global double* in_diff, global unsigned int* mask, const unsigned int threshold, const double scale, global double* out_diff);
