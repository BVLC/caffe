#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
		
#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if defined(cl_nv_pragma_unroll)
#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable
#define NV_PLATFORM
#endif

#define OPENCL_LOCAL_SIZE 32
#define BLOCK_SIZE 32

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
template <class T> __kernel void mmul(const int M, const int N, const int K, const T alpha, global T* A, global T* B, const T beta, global T* C, local T* rowBufferA, local T* colBufferA ) {

#if defined(NV_PLATFORM)
  local T colBufferB[OPENCL_LOCAL_SIZE];
#else
  local T colBufferB[OPENCL_LOCAL_SIZE];
#endif
  
  int colBufferSteps = 0;
  if ( K % OPENCL_LOCAL_SIZE == 0 ) {
    colBufferSteps    = K / OPENCL_LOCAL_SIZE;
  } else {
    colBufferSteps    = K / OPENCL_LOCAL_SIZE + 1;    
  }

  T tmp;
  int idx_n = get_global_id(0);
    
  for( int s = 0; s < colBufferSteps; s++ ) {

    int idx_k = s*OPENCL_LOCAL_SIZE + get_local_id(1);
    if ( idx_k < K ) {
      colBufferB[get_local_id(1)] = B[idx_k*N+idx_n];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int idx_m = 0; idx_m < M; idx_m++ )  {

      int idx_k = s*OPENCL_LOCAL_SIZE + get_local_id(1);
      if ( idx_k < K ) {
        rowBufferA[idx_k] = A[idx_m*K+idx_k];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      
      if ( get_local_id(1) == 0 ) {
        if ( s == 0 ) {
          colBufferA[idx_m] = 0.0;
        }
        tmp = 0.0;
        for( int k = 0; k < OPENCL_LOCAL_SIZE && k < K; k++ ) {
          int idx_k = s*OPENCL_LOCAL_SIZE + k;
          if ( idx_k < K ) {
            //tmp += alpha*A[idx_m*K+idx_k]*B[idx_k*N+idx_n];       // direct
            //tmp += alpha*rowBufferA[idx_k]*B[idx_k*N+idx_n];      // with rowBufferA
            //tmp += alpha*A[idx_m*K+idx_k]*colBufferB[k];          // with colBufferB
            tmp += alpha*rowBufferA[idx_k]*colBufferB[k];     // with rowBufferA & colBufferB
          }
        }
        colBufferA[idx_m] += tmp;
        
        if ( s == colBufferSteps - 1 ) {
          C[idx_m*N+idx_n] = colBufferA[idx_m];     
        }
      }
    }
  }
  
}
template __attribute__((mangled_name(mmulFloat))) kernel void mmul(const int M, const int N, const int K, const float alpha, global float* A, global float* B, const float beta, global float* C,  local float* rowBufferA,  local float* colBufferA);
template __attribute__((mangled_name(mmulDouble))) kernel void mmul(const int M, const int N, const int K, const double alpha, global double* A, global double* B, const double beta, global double* C,  local double* rowBufferA,  local double* colBufferA);


template <class T> __kernel void mmul2(const int M, const int N, const int K, const T alpha, global T* A, global T* B, const T beta, global T* C, local T* rowBufferA, local T* colBufferA ) {

  // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
      // Index of the first sub-matrix of A processed 
      // by the block
      int aBegin = K * BLOCK_SIZE * by;
   
      // Index of the last sub-matrix of A processed 
      // by the block
      int aEnd   = aBegin + K - 1;
   
      // Step size used to iterate through the 
      // sub-matrices of A
      int aStep  = BLOCK_SIZE;
   
      // Index of the first sub-matrix of B processed 
      // by the block
      int bBegin = BLOCK_SIZE * bx;
   
      // Step size used to iterate through the 
      // sub-matrices of B
      int bStep  = BLOCK_SIZE * N;
      
      T Csub = 0.0;
   
      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      for (int a = aBegin, b = bBegin;
               a <= aEnd;
               a += aStep, b += bStep) 
      {
  
          // Declaration of the local memory array As 
          // used to store the sub-matrix of A
          __local T As[BLOCK_SIZE][BLOCK_SIZE];
   
          // Declaration of the local memory array Bs 
          // used to store the sub-matrix of B
          __local T Bs[BLOCK_SIZE][BLOCK_SIZE];
   
          // Load the matrices from global memory
          // to local memory; each thread loads
          // one element of each matrix
          if ( a + K * ty + tx < M*K ) {
            As[ty][tx] = A[a + K * ty + tx];
          }
          //else {
          //  As[ty][tx] = 0.0;
          //}
          if (b + N * ty + tx < K*N ) {
            Bs[ty][tx] = B[b + N * ty + tx];
          } 
          //else {
          //  Bs[ty][tx] = 0.0;
          //}
   
          // Synchronize to make sure the matrices 
          // are loaded
          barrier(CLK_LOCAL_MEM_FENCE);
   
          // Multiply the two matrices together;
          // each thread computes one element
          // of the block sub-matrix
          int limit = BLOCK_SIZE;
          
          if ( K % BLOCK_SIZE != 0 && a == aEnd ) {
            limit = K/BLOCK_SIZE;
          }
          
          if ( K < BLOCK_SIZE ) {
            limit = K;
          }
          
          for (int k = 0; k < limit; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
          }
   
          // Synchronize to make sure that the preceding
          // computation is done before loading two new
          // sub-matrices of A and B in the next iteration
          barrier(CLK_LOCAL_MEM_FENCE);   
      }
   
      // Write the block sub-matrix to device memory;
      // each thread writes one element
      if ( get_global_id(0) < N && get_global_id(1) < M ) {
        int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        if (c + N * ty + tx < M*N ) {
          C[c + N * ty + tx] = Csub;
        }
      }
}
template __attribute__((mangled_name(mmul2Float))) kernel void mmul2(const int M, const int N, const int K, const float alpha, global float* A, global float* B, const float beta, global float* C,  local float* rowBufferA,  local float* colBufferA);
template __attribute__((mangled_name(mmul2Double))) kernel void mmul2(const int M, const int N, const int K, const double alpha, global double* A, global double* B, const double beta, global double* C,  local double* rowBufferA,  local double* colBufferA);
