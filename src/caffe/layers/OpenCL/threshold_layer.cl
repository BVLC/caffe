#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


template <class T> __kernel void ThresholdForward(const int n, const T threshold, const global T* in, global T* out) {

	int idx = get_global_id(0);
	if ( idx < n ) {
	    out[idx] = in[idx] > threshold ? 1 : 0;
	}
}
template __attribute__((mangled_name(ThresholdForwardFloat))) kernel void ThresholdForward(const int n, const float threshold, const global float* in, global float* out);
template __attribute__((mangled_name(ThresholdForwardDouble))) kernel void ThresholdForward(const int n, const double threshold, const global double* in, global double* out);

