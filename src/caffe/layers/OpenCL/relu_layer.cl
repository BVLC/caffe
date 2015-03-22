#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void ReLUForward(const int n, global T* in, global T* out, T negative_slope) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		out[idx] = in[idx] > 0 ? in[idx] : in[idx] * negative_slope;
	}
}
template __attribute__((mangled_name(ReLUForwardFloat))) kernel void ReLUForward(const int n, global float* in, global float* out, float negative_slope);
template __attribute__((mangled_name(ReLUForwardDouble))) kernel void ReLUForward(const int n, global double* in, global double* out, double negative_slope);

template <class T> __kernel void ReLUBackward(const int n, global T* in_diff, global T* in_data, global T* out_diff, T negative_slope) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		out_diff[idx] = in_diff[idx] * ((in_data[idx] > 0) + (in_data[idx] <= 0) * negative_slope);
	}
}
template __attribute__((mangled_name(ReLUBackwardFloat))) kernel void ReLUBackward(const int n, global float* in_diff, global float* in_data, global float* out_diff, float negative_slope)
template __attribute__((mangled_name(ReLUBackwardDouble))) kernel void ReLUBackward(const int n, global double* in_diff, global double* in_data, global double* out_diff, double negative_slope)
