#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void SigmoidForward(const int n, global T* in, global T* out) {

	int idx = get_global_id(0);
	if ( idx < n ) {
	    out[idx] = 1. / (1. + exp(-in[idx]));
	}
}
template __attribute__((mangled_name(SigmoidForwardFloat))) kernel void SigmoidForward(const int n, global float* in, global float* out);
template __attribute__((mangled_name(SigmoidForwardDouble))) kernel void SigmoidForward(const int n, global double* in, global double* out);

template <class T> __kernel void SigmoidBackward(const int n, global T* in_diff, global T* out_data, global T* out_diff) {

	int idx = get_global_id(0);
	if ( idx < n ) {
	    const T sigmoid_x = out_data[idx];
	    out_diff[idx] = in_diff[idx] * sigmoid_x * (1 - sigmoid_x);
	}
}
template __attribute__((mangled_name(SigmoidBackwardFloat))) kernel void SigmoidBackward(const int n, global float* in_diff, global float* out_data, global float* out_diff);
template __attribute__((mangled_name(SigmoidBackwardDouble))) kernel void SigmoidBackward(const int n, global double* in_diff, global double* out_data, global double* out_diff);