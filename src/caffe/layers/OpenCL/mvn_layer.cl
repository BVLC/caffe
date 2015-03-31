#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


template <class T> __kernel void MVNLayerForwardResidual(global T* bottom_data, const int bottom_data_height, const int bottom_data_width,	global T* sum_multiplier, const int sum_multiplier_width,	global T* mean, const int mean_width,	global T* variance, const int variance_width,	const T eps, global T* top_data, const int top_data_height, const int top_data_width) {

	if ( get_work_dim() == 1 ) {
		int idx = get_global_id(0);
		int idx_h = idx / bottom_data_width;
		int idx_w = idx % bottom_data_width;
		
		T divident = (bottom_data[idx] - mean[idx_h]*sum_multiplier[idx_w]);
		T divisor  = (sqrt(variance[idx_h] - mean[idx_h]*mean[idx_h]) + eps)*sum_multiplier[idx_w];
		top_data[idx] = divident/divisor;
	}
	
	if ( get_work_dim() == 2 ) {
		int idx_h = get_global_id(0);
		int idx_w = get_global_id(1);
	
		while( idx_w < bottom_data_width ) {
			int idx   = bottom_data_width*idx_h + idx_w;
			T divident = (bottom_data[idx] - mean[idx_h]*sum_multiplier[idx_w]);
			T divisor  = (sqrt(variance[idx_h] - mean[idx_h]*mean[idx_h]) + eps)*sum_multiplier[idx_w];
			top_data[idx] = divident/divisor;
			idx_w += get_global_size(1);
		}
	}
}
template __attribute__((mangled_name(MVNLayerForwardResidualFloat))) kernel void MVNLayerForwardResidual(global float* bottom_data, const int bottom_data_height, const int bottom_data_width,	global float* sum_multiplier, const int sum_multiplier_width,	global float* mean, const int mean_width,	global float* variance, const int variance_width,	const float eps, global float* top_data, const int top_data_height, const int top_data_width);
template __attribute__((mangled_name(MVNLayerForwardResidualDouble))) kernel void MVNLayerForwardResidual(global double* bottom_data, const int bottom_data_height, const int bottom_data_width,	global double* sum_multiplier, const int sum_multiplier_width,	global double* mean, const int mean_width,	global double* variance, const int variance_width,	const double eps, global double* top_data, const int top_data_height, const int top_data_width);

template <class T> __kernel void MVNLayerForwardMV2(global T* data2D, const int data2D_height, const int data2D_width, global T* data1D, const int data1D_length, global T* linear_term, global T* quadratic_term) {

	int idx = get_global_id(0);

	global T* data2D_ptr = data2D;
	data2D_ptr += idx*data2D_width;
		
	T linear_sum	= 0;
	T quadratic_sum	= 0;
				
	for(int j = 0; j < data2D_width; j++ ) {
	   	linear_sum 		+= data2D_ptr[j]*data1D[j];
	   	quadratic_sum 	+= data2D_ptr[j]*data2D_ptr[j]*data1D[j];
	}
	
	linear_term[idx] 	= linear_sum/data2D_width;
	quadratic_term[idx] = quadratic_sum/data2D_width;
}
template __attribute__((mangled_name(MVNLayerForwardMV2Float))) kernel void MVNLayerForwardMV2(global float* data2D, const int data2D_height, const int data2D_width, global float* data1D, const int data1D_length, global float* linear_term, global float* quadratic_term);
template __attribute__((mangled_name(MVNLayerForwardMV2Double))) kernel void MVNLayerForwardMV2(global double* data2D, const int data2D_height, const int data2D_width, global double* data1D, const int data1D_length, global double* linear_term, global double* quadratic_term);

template <class T> __kernel void MVNLayerForward(global T* data2D_in, const int data2D_in_height, const int data2D_in_width, global T* data1D_in, const int data1D_in_length, global T* linear_term, const int linear_term_length, global T* quadratic_term, const int quadratic_term_length, const T eps, global T* data2D_out) {

	int idx = get_global_id(0);
	
	int idx_h = idx / data2D_in_width;
	int idx_w = idx % data2D_in_width;
	
	global T* data2D_in_ptr = data2D_in;
	data2D_in_ptr += idx;
	
	global T* data1D_in_ptr = data1D_in;
	data1D_in_ptr += idx_w;
	
	global T* linear_term_ptr = linear_term;
	linear_term_ptr += idx_h;

	global T* quadratic_term_ptr = quadratic_term;
	quadratic_term_ptr += idx_h;

	global T* data2D_out_ptr = data2D_out;
	data2D_out_ptr += idx;
	
	T divident = (data2D_in_ptr[0] - linear_term_ptr[0]*data1D_in_ptr[0]);
	T quotient = (sqrt(quadratic_term_ptr[0] - linear_term_ptr[0]*linear_term_ptr[0]) + eps)*data1D_in_ptr[0];
	
	data2D_out_ptr[0] = divident/quotient;

}
template __attribute__((mangled_name(MVNLayerForwardFloat))) kernel void MVNLayerForward(global float* data2D_in, const int data2D_in_height, const int data2D_in_width, global float* data1D_in, const int data1D_in_length, global float* linear_term, const int linear_term_length, global float* quadratic_term, const int quadratic_term_length, const float eps, global float* data2D_out);
template __attribute__((mangled_name(MVNLayerForwardDouble))) kernel void MVNLayerForward(global double* data2D_in, const int data2D_in_height, const int data2D_in_width, global double* data1D_in, const int data1D_in_length, global double* linear_term, const int linear_term_length, global double* quadratic_term, const int quadratic_term_length, const double eps, global double* data2D_out);

template <class T> __kernel void MVNLayerForwardS2(global T* data2D_in, const int data2D_in_height, const int data2D_in_width, global T* data1D_in, const int data1D_in_length, global T* data2D_out) {

	int idx = get_global_id(0);
	
	int idx_h = idx / data2D_in_width;
	int idx_w = idx % data2D_in_width;
	
	global T* data2D_in_ptr = data2D_in;
	data2D_in_ptr += idx;
	
	global T* data1D_in_ptr = data1D_in;
	data1D_in_ptr += idx_w;
	
	global T* data2D_out_ptr = data2D_out;
	data2D_out_ptr += idx;
	
	T val = 0;
	for(int j = 0; j < data2D_in_width; j++) {
		val += data2D_in[idx_h*data2D_in_width+j]*data1D_in[j];
	}
	val /= data2D_in_width;
	
	data2D_out_ptr[0] = data2D_in_ptr[0] - val*data1D_in_ptr[0];

}
template __attribute__((mangled_name(MVNLayerForwardS2Float))) kernel void MVNLayerForwardS2(global float* data2D_in, const int data2D_in_height, const int data2D_in_width, global float* data1D_in, const int data1D_in_length, global float* data2D_out);
template __attribute__((mangled_name(MVNLayerForwardS2Double))) kernel void MVNLayerForwardS2(global double* data2D_in, const int data2D_in_height, const int data2D_in_width, global double* data1D_in, const int data1D_in_length, global double* data2D_out);



template <class T> __kernel void MVNLayerBackwardMV2(global T* data2D, global T* diff2D, const int data2D_height, const int data2D_width, global T* data1D, const int data1D_length, global T* linear_term, global T* quadratic_term) {

	int idx = get_global_id(0);

	global T* data2D_ptr = data2D;
	data2D_ptr += idx*data2D_width;

	global T* diff2D_ptr = diff2D;
	diff2D_ptr += idx*data2D_width;

	T linear_sum	= 0;
	T quadratic_sum	= 0;
				
	for(int j = 0; j < data2D_width; j++ ) {
	   	linear_sum 		+= diff2D_ptr[j]*data1D[j];
	   	quadratic_sum 	+= data2D_ptr[j]*diff2D_ptr[j]*data1D[j];
	}
	
	linear_term[idx] 	= linear_sum;
	quadratic_term[idx] = quadratic_sum;
}
template __attribute__((mangled_name(MVNLayerBackwardMV2Float))) kernel void MVNLayerBackwardMV2(global float* data2D, global float* diff2D, const int data2D_height, const int data2D_width, global float* data1D, const int data1D_length, global float* linear_term, global float* quadratic_term);
template __attribute__((mangled_name(MVNLayerBackwardMV2Double))) kernel void MVNLayerBackwardMV2(global double* data2D, global double* diff2D, const int data2D_height, const int data2D_width, global double* data1D, const int data1D_length, global double* linear_term, global double* quadratic_term);

template <class T> __kernel void MVNLayerBackwardS1(global T* data2D, global T* diff2D, const int data2D_height, const int data2D_width, global T* data1D, const int data1D_length, global T* linear_term, const int linear_term_length, global T* quadratic_term, const int quadratic_term_length, global T* data2D_out) {

	int idx = get_global_id(0);

	int idx_h = idx / data2D_width;
	int idx_w = idx % data2D_width;

	global T* data2D_ptr = data2D;
	data2D_ptr += idx;

	global T* diff2D_ptr = diff2D;
	diff2D_ptr += idx;

	global T* data1D_ptr = data1D;
	data1D_ptr += idx_w;

	global T* linear_term_ptr = linear_term;
	linear_term_ptr += idx_h;

	global T* quadratic_term_ptr = quadratic_term;
	quadratic_term_ptr += idx_h;

	global T* data2D_out_ptr = data2D_out;
	data2D_out_ptr += idx;
	
	data2D_out_ptr[0] = diff2D_ptr[0] - 1.0/data2D_width * data1D_ptr[0] * ( linear_term_ptr[0] + quadratic_term_ptr[0]*data2D_ptr[0] ); 
}
template __attribute__((mangled_name(MVNLayerBackwardS1Float))) kernel void MVNLayerBackwardS1(global float* data2D, global float* diff2D, const int data2D_height, const int data2D_width, global float* data1D, const int data1D_length, global float* linear_term, const int linear_term_length, global float* quadratic_term, const int quadratic_term_length, global float* data2D_out);
template __attribute__((mangled_name(MVNLayerBackwardS1Double))) kernel void MVNLayerBackwardS1(global double* data2D, global double* diff2D, const int data2D_height, const int data2D_width, global double* data1D, const int data1D_length, global double* linear_term, const int linear_term_length, global double* quadratic_term, const int quadratic_term_length, global double* data2D_out);

template <class T> __kernel void MVNLayerBackward(global T* data2D_in, const int data2D_in_height, const int data2D_in_width, global T* data1D_in, const int data1D_in_length, global T* linear_term, const int linear_term_length, global T* quadratic_term, const int quadratic_term_length, const T eps, global T* data2D_out) {

	int idx = get_global_id(0);
	
	int idx_h = idx / data2D_in_width;
	int idx_w = idx % data2D_in_width;
	
	global T* data2D_in_ptr = data2D_in;
	data2D_in_ptr += idx;
	
	global T* data1D_in_ptr = data1D_in;
	data1D_in_ptr += idx_w;
	
	global T* linear_term_ptr = linear_term;
	linear_term_ptr += idx_h;

	global T* quadratic_term_ptr = quadratic_term;
	quadratic_term_ptr += idx_h;

	global T* data2D_out_ptr = data2D_out;
	data2D_out_ptr += idx;
	
	T divident = data2D_in_ptr[0];
	T quotient = (sqrt(quadratic_term_ptr[0] - linear_term_ptr[0]*linear_term_ptr[0]) + eps)*data1D_in_ptr[0];
	
	data2D_out_ptr[0] = divident/quotient;

}
template __attribute__((mangled_name(MVNLayerBackwardFloat))) kernel void MVNLayerBackward(global float* data2D_in, const int data2D_in_height, const int data2D_in_width, global float* data1D_in, const int data1D_in_length, global float* linear_term, const int linear_term_length, global float* quadratic_term, const int quadratic_term_length, const float eps, global float* data2D_out);
template __attribute__((mangled_name(MVNLayerBackwardDouble))) kernel void MVNLayerBackward(global double* data2D_in, const int data2D_in_height, const int data2D_in_width, global double* data1D_in, const int data1D_in_length, global double* linear_term, const int linear_term_length, global double* quadratic_term, const int quadratic_term_length, const double eps, global double* data2D_out);

//	caffe::OpenCL::clMVNLayerForwardMV2(bottom_data, num, dim, sum_multiplier_.gpu_data(), dim, (Dtype*) mean_.mutable_gpu_data(), (Dtype*) variance_.mutable_gpu_data());
//	caffe::OpenCL::clMVNLayerForward(bottom_data, num, dim, sum_multiplier_.gpu_data(), dim, (Dtype*) mean_.mutable_gpu_data(), num, (Dtype*) variance_.mutable_gpu_data(), num, eps, top_data);

template <class T> __kernel void MVNLayerForward_perf(global T* A2D_top, global T* A2D_top_diff, const int top_height, const int top_width, global T* A2D_bottom, global T* A2D_bottom_diff, const int bottom_height, const int bottom_width, global T* A1D_sum_multiplier, global T* A1D_buffer, const int sum_multiplier_length, const T eps, global T* A2D_out) {

	int idx = get_global_id(0);
	
	int idx_h = idx / top_width;
	int idx_w = idx % top_width;
	
	global T* _A2D_top = A2D_top;
	_A2D_top += idx;

	global T* _A2D_top_diff = A2D_top_diff;
	_A2D_top_diff += idx;

	global T* _A2D_bottom = A2D_bottom;
	_A2D_bottom += idx;

	global T* _A2D_bottom_diff = A2D_bottom_diff;
	_A2D_bottom_diff += idx;

	global T* _A1D_sum_multiplier = A1D_sum_multiplier;
	_A1D_sum_multiplier += idx_w;

	global T* _A1D_buffer = A1D_buffer;
	_A1D_buffer += idx_w;

	global T* _A2D_out = A2D_out;
	_A2D_out += idx;

	T sum1 	= 0.0;
	T sum2 	= 0.0;
		
	sum1 = 0.0;
	for( int i = 0; i < bottom_width; i++ ) {
		sum1 += A2D_bottom[idx_h*bottom_width+i]*A1D_sum_multiplier[i];
	}
	sum1 /= bottom_width;

	sum2 = 0.0;
	for( int i = 0; i < bottom_width; i++ ) {
		sum2 += A2D_bottom[idx_h*bottom_width+i]*A2D_bottom[idx_h*bottom_width+i]*A1D_sum_multiplier[i];
	}
	sum2 /= bottom_width;

	T divident = (_A2D_bottom[0] - sum1*_A1D_sum_multiplier[0]);
	T quotient = (sqrt(sum2 - sum1*sum1) + eps)*_A1D_sum_multiplier[0];
	
	_A2D_out[0] = divident/quotient;
}
template __attribute__((mangled_name(MVNLayerForward_perfFloat))) kernel void MVNLayerForward_perf(global float* A2D_top, global float* A2D_top_diff, const int top_height, const int top_width, global float* A2D_bottom, global float* A2D_bottom_diff, const int bottom_height, const int bottom_width, global float* A1D_sum_multiplier, global float* A1D_buffer, const int sum_multiplier_length, const T eps, global float* A2D_out);
template __attribute__((mangled_name(MVNLayerForward_perfDouble))) kernel void MVNLayerForward_perf(global double* A2D_top, global double* A2D_top_diff, const int top_height, const int top_width, global double* A2D_bottom, global double* A2D_bottom_diff, const int bottom_height, const int bottom_width, global double* A1D_sum_multiplier, global double* A1D_buffer, const int sum_multiplier_length, const T eps, global double* A2D_out);


template <class T> __kernel void MVNLayerBackward_perf(global T* A2D_top, global T* A2D_top_diff, const int top_height, const int top_width, global T* A2D_bottom, global T* A2D_bottom_diff, const int bottom_height, const int bottom_width, global T* A1D_sum_multiplier, global T* A1D_buffer, const int sum_multiplier_length, const T eps, global T* A2D_out) {

	int idx = get_global_id(0);
	
	int idx_h = idx / top_width;
	int idx_w = idx % top_width;
	
	global T* _A2D_top = A2D_top;
	_A2D_top += idx;

	global T* _A2D_top_diff = A2D_top_diff;
	_A2D_top_diff += idx;

	global T* _A2D_bottom = A2D_bottom;
	_A2D_bottom += idx;

	global T* _A2D_bottom_diff = A2D_bottom_diff;
	_A2D_bottom_diff += idx;

	global T* _A1D_sum_multiplier = A1D_sum_multiplier;
	_A1D_sum_multiplier += idx_w;

	global T* _A1D_buffer = A1D_buffer;
	_A1D_buffer += idx_w;

	global T* _A2D_out = A2D_out;
	_A2D_out += idx;

	T sum1 	= 0.0;
	T sum2 	= 0.0;
	T sum3 	= 0.0;
	T sum4 	= 0.0;
	T value = 0.0;
		
	sum1 = 0.0;
	for( int i = 0; i < top_width; i++ ) {
		sum1 += A2D_top_diff[idx_h*top_width+i]*A1D_sum_multiplier[i];
	}

	sum2 = 0.0;
	for( int i = 0; i < top_width; i++ ) {
		sum2 += A2D_top[idx_h*top_width+i]*A2D_top_diff[idx_h*top_width+i]*A1D_sum_multiplier[i];
	}
	
	T S1 = _A2D_top_diff[0] - 1.0/top_width * A1D_sum_multiplier[idx_w] * ( sum1 + sum2*_A2D_top[0] ); 
		
	sum3 = 0.0;
	for( int i = 0; i < top_width; i++ ) {
		sum3 += A2D_bottom[idx_h*top_width+i]*A1D_sum_multiplier[i];
	}
	sum3 /= top_width;

	sum4 = 0.0;
	for( int i = 0; i < top_width; i++ ) {
		sum4 += A2D_bottom[idx_h*top_width+i]*A2D_bottom[idx_h*top_width+i]*A1D_sum_multiplier[i];
	}
	sum4 /= top_width;
	
	T divident = S1;
	T quotient = (sqrt(sum4 - sum3*sum3) + eps)*A1D_sum_multiplier[idx_w];
	
	_A2D_out[0] = divident/quotient;
}
template __attribute__((mangled_name(MVNLayerBackward_perfFloat))) kernel void MVNLayerBackward_perf(global float* A2D_top, global float* A2D_top_diff, const int top_height, const int top_width, global float* A2D_bottom, global float* A2D_bottom_diff, const int bottom_height, const int bottom_width, global float* A1D_sum_multiplier, global float* A1D_buffer, const int sum_multiplier_length, const T eps, global float* A2D_out);
template __attribute__((mangled_name(MVNLayerBackward_perfDouble))) kernel void MVNLayerBackward_perf(global double* A2D_top, global double* A2D_top_diff, const int top_height, const int top_width, global double* A2D_bottom, global double* A2D_bottom_diff, const int bottom_height, const int bottom_width, global double* A1D_sum_multiplier, global double* A1D_buffer, const int sum_multiplier_length, const T eps, global double* A2D_out);

