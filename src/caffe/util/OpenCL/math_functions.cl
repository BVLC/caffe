#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

template <class T> __kernel void clsign(const int n, global T* x, global T* y) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		y[idx] = sign(x[idx]);
	}
}
template __attribute__((mangled_name(clsignFloat))) kernel void clsign(const int n, global float* x, global float* y); 
template __attribute__((mangled_name(clsignDouble))) kernel void clsign(const int n, global double* x, global double* y);

template <class T> __kernel void clsgnbit(const int n, global T* x, global T* y) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		y[idx] = signbit(x[idx]);
	}
}
template __attribute__((mangled_name(clsgnbitFloat))) kernel void clsgnbit(const int n, global float* x, global float* y); 
template __attribute__((mangled_name(clsgnbitDouble))) kernel void clsgnbit(const int n, global double* x, global double* y);

template <class T> __kernel void clabs(const int n, global T* x, global T* y) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		y[idx] = fabs(x[idx]);
	}
}
template __attribute__((mangled_name(clabsFloat))) kernel void clabs(const int n, global float* x, global float* y); 
template __attribute__((mangled_name(clabsDouble))) kernel void clabs(const int n, global double* x, global double* y);

template <class T> __kernel void cldiv(const int n, global T* x, global T* y, global T* z) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		z[idx] = x[idx] / y[idx];
	}
}
template __attribute__((mangled_name(cldivFloat))) kernel void cldiv(const int n, global float* x, global float* y, global float* z); 
template __attribute__((mangled_name(cldivDouble))) kernel void cldiv(const int n, global double* x, global double* y, global double* z);

template <class T> __kernel void clmul(const int n, global T* x, global T* y, global T* z) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		z[idx] = x[idx] * y[idx];
	}
}
template __attribute__((mangled_name(clmulFloat))) kernel void clmul(const int n, global float* x, global float* y, global float* z); 
template __attribute__((mangled_name(clmulDouble))) kernel void clmul(const int n, global double* x, global double* y, global double* z);

template <class T> __kernel void clFillBuffer(const int n, const T alpha, global T* x) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		x[idx] = alpha;
	}
}
template __attribute__((mangled_name(clFillBufferChar))) kernel void clFillBuffer(const int n, const char alpha, global char* x);
template __attribute__((mangled_name(clFillBufferInt))) kernel void clFillBuffer(const int n, const int alpha, global int* x);
template __attribute__((mangled_name(clFillBufferFloat))) kernel void clFillBuffer(const int n, const float alpha, global float* x); 
template __attribute__((mangled_name(clFillBufferDouble))) kernel void clFillBuffer(const int n, const double alpha, global double* x);

template <class T> __kernel void clGPU2GPU(const int n, global T* x, const int offset_x, global T* y, const int offset_y) {

	int idx = get_local_id(0);
	int localSize = get_local_size(0);
	for(int i = 0; i < n; i += localSize ) {
		y[idx+offset_y+n-1-i-idx] = x[idx+offset_x+n-1-i-idx];
	}
}
template __attribute__((mangled_name(clGPU2GPUChar))) kernel void clGPU2GPU(const int n, global char* x, const int offset_x, global char* y, const int offset_y);
template __attribute__((mangled_name(clGPU2GPUInt))) kernel void clGPU2GPU(const int n, global int* x, const int offset_x, global int* y, const int offset_y);
template __attribute__((mangled_name(clGPU2GPUFloat))) kernel void clGPU2GPU(const int n, global float* x, const int offset_x, global float* y, const int offset_y); 
template __attribute__((mangled_name(clGPU2GPUDouble))) kernel void clGPU2GPU(const int n, global double* x, const int offset_x, global double* y, const int offset_y);

template <class T> __kernel void clsub(const int n, global T* x, global T* y, global T* z) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		z[idx] = x[idx] - y[idx];
	}
}
template __attribute__((mangled_name(clsubFloat))) kernel void clsub(const int n, global float* x, global float* y, global float* z); 
template __attribute__((mangled_name(clsubDouble))) kernel void clsub(const int n, global double* x, global double* y, global double* z);

template <class T> __kernel void cladd(const int n, global T* x, global T* y, global T* z) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		z[idx] = x[idx] + y[idx];
	}
}
template __attribute__((mangled_name(claddFloat))) kernel void cladd(const int n, global float* x, global float* y, global float* z); 
template __attribute__((mangled_name(claddDouble))) kernel void cladd(const int n, global double* x, global double* y, global double* z);

template <class T> __kernel void cladd_scalar(const int N, const T alpha, global T* Y) {

	int idx = get_global_id(0);
	if ( idx < N ) {
	    Y[idx] += alpha;
	}
}
template __attribute__((mangled_name(cladd_scalarFloat))) kernel void cladd_scalar(const int N, const float alpha, global float* Y); 
template __attribute__((mangled_name(cladd_scalarDouble))) kernel void cladd_scalar(const int N, const double alpha, global double* Y);

template <class T> __kernel void clpowx(const int n, global T* x, const T alpha, global T* z) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		// z[idx] = pow((x[idx],alpha); original
		//z[idx] = pow((float), (float) alpha); // NV fix

		if ( alpha == 2.0 ) {
			z[idx] = pow((float) fabs(x[idx]), (float) alpha); // MVN fix, ContrastiveLossLayerTest, AdaGradSolverTest
		} else {
			z[idx] = pow((float) x[idx], (float) alpha); // MVN fix, ContrastiveLossLayerTest, AdaGradSolverTest
		}
		//printf("z[%d] = %f <> pow(%f, %f)\n", idx, z[idx], x[idx], alpha);
	}
}
template __attribute__((mangled_name(clpowxFloat))) kernel void clpowx(const int n, global float* x, float alpha, global float* z); 
template __attribute__((mangled_name(clpowxDouble))) kernel void clpowx(const int n, global double* x, double alpha, global double* z);

template <class T> __kernel void clexp(const int n, global T* x, global T* y) {

	int idx = get_global_id(0);
	if ( idx < n ) {
		y[idx] = exp(x[idx]);
	}
}
template __attribute__((mangled_name(clexpFloat))) kernel void clexp(const int n, global float* x, global float* y); 
template __attribute__((mangled_name(clexpDouble))) kernel void clexp(const int n, global double* x, global double* y);

/* Source: OpenCL Programming Guide by authors Munshi, Gaster, Mattson, Fung, Ginsburg
 * Ch 21 Matrix Multiplication with OpenCL
 */
template <class T> __kernel void mmul(const int M, const int N, const int K, global T* A, global T* B, global T* C, local T* Bwrk ) {

	int i 		= get_global_id(0);
	int iloc 	= get_local_id(0);
	int nloc	= get_local_size(0);
	
	T Awrk[4096];
	T tmp;
	
	for( int k = 0; k < K; k++ ) {
		Awrk[k] = A[i*N+k];
	}
	
	for(int j = 0; j < M; j++ )  {
		for(int k=iloc; k<K; k += nloc ) {
			Bwrk[k] = B[k*K+j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		tmp = 0.0;
		for ( int k = 0; k < K; k++ ) {
			tmp += Awrk[k]*Bwrk[k];
		}
		C[i*N+j] += tmp;
	}
}
template __attribute__((mangled_name(mmulFloat))) kernel void mmul(const int M, const int N, const int K, global float* A, global float* B, global float* C, local float* Bwrk);
template __attribute__((mangled_name(mmulDouble))) kernel void mmul(const int M, const int N, const int K, global double* A, global double* B, global double* C, local double* Bwrk);

