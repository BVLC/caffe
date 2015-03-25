#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


template <class T> __kernel void PReLUForward(const int n, const int channels, const int dim, global T* in, global T* out, global T* slope_data, const int div_factor) {

	int idx = get_global_id(0);
	if ( idx < n ) {
    int c = (idx / dim) % channels / div_factor;
    out[idx] = in[idx] > 0 ? in[idx] : in[idx] * slope_data[c];
	}
}
template __attribute__((mangled_name(PReLUForwardFloat))) kernel void PReLUForward(const int n, const int channels, const int dim, global float* in, global float* out, global float* slope_data, const int div_factor);
template __attribute__((mangled_name(PReLUForwardDouble))) kernel void PReLUForward(const int n, const int channels, const int dim, global double* in, global double* out, global double* slope_data, const int div_factor)

template <class T> __kernel void PReLUBackward(const int n, const int channels, const int dim, global T* in_diff, global T* in_data, global T* out_diff, global T* slope_data, const int div_factor) {

	int idx = get_global_id(0);
	if ( idx < n ) {
    int c = (idx / dim) % channels / div_factor;
    out_diff[idx] = in_diff[idx] * ((in_data[idx] > 0) + (in_data[idx] <= 0) * slope_data[c]);
	}
}
template __attribute__((mangled_name(PReLUBackwardFloat))) kernel void PReLUBackward(const int n, const int channels, const int dim, global float* in_diff, global float* in_data, global float* out_diff, global float* slope_data, const int div_factor);
template __attribute__((mangled_name(PReLUBackwardDouble))) kernel void PReLUBackward(const int n, const int channels, const int dim, global double* in_diff, global double* in_data, global double* out_diff, global double* slope_data, const int div_factor);

template <class T> __kernel void PReLUParamBackward(const int n, global T* in_diff, global T* in_data, global T* out_diff) {

	int idx = get_global_id(0);
	if ( idx < n ) {
    out_diff[idx] = in_diff[idx] * in_data[idx] * (in_data[idx] <= 0);
	}
}
template __attribute__((mangled_name(PReLUParamBackwardFloat))) kernel void PReLUParamBackward(const int n, global float* in_diff, global float* in_data, global float* out_diff);
template __attribute__((mangled_name(PReLUParamBackwardDouble))) kernel void PReLUParamBackward(const int n, global double* in_diff, global double* in_data, global double* out_diff);