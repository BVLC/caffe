#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void SoftmaxLossForwardGPU(const int nthreads, const global T* prob_data, const global T* label, global T* loss, const int num, const int dim, const int spatial_dim, const int has_ignore_label_, const int ignore_label_, global T* counts) {

	int idx = get_global_id(0);
	if ( idx < nthreads ) {
    const int n = idx / spatial_dim;
    const int s = idx % spatial_dim;
    const int label_value = (int)(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[idx] = 0;
      counts[idx] = 0;
    } else {
      loss[idx] = -log(max(prob_data[n * dim + label_value * spatial_dim + s], ( T )(FLT_MIN)));
      counts[idx] = 1;
    }
	}
}
template __attribute__((mangled_name(SoftmaxLossForwardGPUFloat))) kernel void SoftmaxLossForwardGPU(const int nthreads, const global float* prob_data, const global float* label, global float* loss, const int num, const int dim, const int spatial_dim, const int has_ignore_label_, const int ignore_label_, global float* counts)
template __attribute__((mangled_name(SoftmaxLossForwardGPUDouble))) kernel void SoftmaxLossForwardGPU(const int nthreads, const global double* prob_data, const global double* label, global double* loss, const int num, const int dim, const int spatial_dim, const int has_ignore_label_, const int ignore_label_, global double* counts)



template <class T> __kernel void SoftmaxLossBackwardGPU(const int nthreads, const global T* top, const global T* label, global T* bottom_diff, const int num, const int dim, const int spatial_dim, const int has_ignore_label_, const int ignore_label_, global T* counts) {

	int idx = get_global_id(0);
	if ( idx < nthreads ) {
	  const int channels = dim / spatial_dim;
    const int n = idx / spatial_dim;
    const int s = idx % spatial_dim;
    const int label_value = (int)(label[n * spatial_dim + s]);

    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[idx] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[idx] = 1;
    }
	}
}
template __attribute__((mangled_name(SoftmaxLossBackwardGPUFloat))) kernel void SoftmaxLossBackwardGPU(const int nthreads, const global float* top, const global float* label, global float* bottom_diff, const int num, const int dim, const int spatial_dim, const int has_ignore_label_, const int ignore_label_, global float* counts);
template __attribute__((mangled_name(SoftmaxLossBackwardGPUDouble))) kernel void SoftmaxLossBackwardGPU(const int nthreads, const global double* top, const global double* label, global double* bottom_diff, const int num, const int dim, const int spatial_dim, const int has_ignore_label_, const int ignore_label_, global double* counts);
