#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define KBNLL_THRESHOLD 50.0

template <class T> __kernel void BNLLForward(const int n, global T* in, global T* out) {

	int idx = get_global_id(0);
	if ( idx < n ) {
	    out[idx] = in[idx] > 0 ? in[idx] + log(1. + exp(-in[idx])) : log(1. + exp(in[idx]));
	}
}
template __attribute__((mangled_name(BNLLForwardFloat))) kernel void BNLLForward(const int n, global float* in, global float* out);
template __attribute__((mangled_name(BNLLForwardDouble))) kernel void BNLLForward(const int n, global double* in, global double* out);

template <class T> __kernel void BNLLBackward(const int n, global T* in_diff, global T* in_data, global T* out_diff) {

	int idx = get_global_id(0);
	if ( idx < n ) {
	    T expval = exp(min(in_data[idx], ( T ) KBNLL_THRESHOLD));
	    out_diff[idx] = in_diff[idx] * expval / (expval + 1.);
	}
}
template __attribute__((mangled_name(BNLLBackwardFloat))) kernel void BNLLBackward(const int n, global float* in_diff, global float* in_data, global float* out_diff);
template __attribute__((mangled_name(BNLLBackwardDouble))) kernel void BNLLBackward(const int n, global double* in_diff, global double* in_data, global double* out_diff);
