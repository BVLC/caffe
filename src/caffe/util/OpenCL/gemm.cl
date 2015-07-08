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

#include "definitions.hpp"

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_NA_NB(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
    
  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;

  // first index of first thread reading B in local workgroup
  int b_bgn = OPENCL_BLOCK_SIZE * tile_x;
  
  // last index of first thread reading B in local workgroup
  int b_end = b_bgn + N*(K-1);

  // step taken by each thread reading B in local workgroup
  int b_stp  = OPENCL_BLOCK_SIZE * N;
      
  // accumulates the result
  T sum = 0.0;

  int global_x = 0;
  int global_y = 0;

  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];
  
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
   
    // each thread in workgroup reads one element of matrix A from global to local memory
    if ( thread_x + global_x < K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[a + K * thread_y + thread_x];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }
      
    // each thread in workgroup reads one element of matrix B from global to local memory
    if ( thread_y + global_y < K ) {
      localMemB[thread_y][thread_x] = B_ptr[b + N * thread_y + thread_x];
    } else { // needed on AMD
      localMemB[thread_y][thread_x] = 0.0;
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    for (int k = 0; k < OPENCL_BLOCK_SIZE; k++) {
      sum += localMemA[thread_y][k] * localMemB[k][thread_x];
    }

    // Synchronize all sub-results
    barrier(CLK_LOCAL_MEM_FENCE);   
  }

  // write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    int c = N * OPENCL_BLOCK_SIZE * tile_y + OPENCL_BLOCK_SIZE * tile_x;
    if (c + N * thread_y + thread_x < M*N ) {
      C_ptr[c + N * thread_y + thread_x] = sum + beta*C_ptr[c + N * thread_y + thread_x];
    }
  }
}
template __attribute__((mangled_name(mmul_NA_NBFloat))) kernel void mmul_NA_NB(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NBDouble))) kernel void mmul_NA_NB(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_NA_NB_v4(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
    
  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;

  // first index of first thread reading B in local workgroup
  int b_bgn = OPENCL_BLOCK_SIZE * tile_x;
  
  // last index of first thread reading B in local workgroup
  int b_end = b_bgn + N*(K-1);

  // step taken by each thread reading B in local workgroup
  int b_stp  = OPENCL_BLOCK_SIZE * N;
      
  // accumulates the result
  T sum = 0.0;

  int global_x = 0;
  int global_y = 0;

  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // Initialise the accumulation registers
  T acc[OPENCL_WPT];
  for (int w=0; w<OPENCL_WPT; w++) {
    acc[w] = 0.0;
  }
  
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
   
    for (int w=0; w<OPENCL_WPT; w++) {
      // each thread in workgroup reads one element of matrix A from global to local memory
      if ( thread_x + w*OPENCL_RTS + global_x < K ) {
        localMemA[thread_y][thread_x + w*OPENCL_RTS] = alpha*A_ptr[a + K * thread_y + thread_x + w*OPENCL_RTS];
      } else { // needed on AMD
        localMemA[thread_y][thread_x + w*OPENCL_RTS] = 0.0;
      }
      
      // each thread in workgroup reads one element of matrix B from global to local memory
      if ( thread_y + global_y < K ) {
        localMemB[thread_y][thread_x + w*OPENCL_RTS] = B_ptr[b + N * thread_y + thread_x + w*OPENCL_RTS];
      } else { // needed on AMD
        localMemB[thread_y][thread_x + w*OPENCL_RTS] = 0.0;
      }
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    for (int k = 0; k < OPENCL_BLOCK_SIZE; k++) {
      for (int w=0; w<OPENCL_WPT; w++) {
        acc[w] += localMemA[thread_y][k] * localMemB[k][thread_x + w*OPENCL_RTS];
      }
    }

    // Synchronize all sub-results
    barrier(CLK_LOCAL_MEM_FENCE);   
  }

  // write all results back to global memory
  //if ( get_global_id(0) < N && get_global_id(1) < M ) {
    for (int w=0; w<OPENCL_WPT; w++) {
      int c = N * OPENCL_BLOCK_SIZE * tile_y + OPENCL_BLOCK_SIZE * tile_x;
      if (c + N * thread_y + thread_x + w*OPENCL_RTS < M*N ) {
        C_ptr[c + N * thread_y + thread_x + w*OPENCL_RTS] = acc[w] + beta*C_ptr[c + N * thread_y + thread_x + w*OPENCL_RTS];
      }
    }
  //}
}
template __attribute__((mangled_name(mmul_NA_NB_v4Float))) kernel void mmul_NA_NB_v4(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_v4Double))) kernel void mmul_NA_NB_v4(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);


/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_TA_TB(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
    
  // first index of first thread reading A in local workgroup
  int a_bgn = OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  //int a_end   = a_bgn + K - 1;
  int a_end = a_bgn + M*K;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE * M;

  // first index of first thread reading A in local workgroup
  int b_bgn = K * OPENCL_BLOCK_SIZE * tile_x;

  // last index to first thread reading A in local workgroup
  int b_end = b_bgn + K - 1;

  // step taken by each thread reading A
  int b_stp = OPENCL_BLOCK_SIZE;
      
  // accumulates the result
  T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {

    // local memory for matrix A
    __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

    // local memory for matrix B
    __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];
   
    // each thread in workgroup reads one element of matrix A from global to local memory
    if ( thread_x + global_x < K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[a + M * thread_x + thread_y];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }
    
    // each thread in workgroup reads one element of matrix B from global to local memory
    if ( thread_y + global_y < K ) {
      localMemB[thread_y][thread_x] = B_ptr[b + K * thread_x + thread_y];
    } else { // needed on AMD
      localMemB[thread_y][thread_x] = 0.0;
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    for (int k = 0; k < OPENCL_BLOCK_SIZE; k++) {
      sum += localMemA[thread_y][k] * localMemB[k][thread_x];
    }

    // Synchronize all sub-results
    barrier(CLK_LOCAL_MEM_FENCE);   
  }

  // write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    int c = N * OPENCL_BLOCK_SIZE * tile_y + OPENCL_BLOCK_SIZE * tile_x;
    if (c + N * thread_y + thread_x < M*N ) {
      C_ptr[c + N * thread_y + thread_x] = sum + beta*C_ptr[c + N * thread_y + thread_x];
    }
  }
}
template __attribute__((mangled_name(mmul_TA_TBFloat))) kernel void mmul_TA_TB(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_TA_TBDouble))) kernel void mmul_TA_TB(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);


/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_TA_NB(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
    
  // first index of first thread reading A in local workgroup
  int a_bgn = OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  //int a_end   = a_bgn + K - 1;
  int a_end = a_bgn + M*K;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE * M;

  // first index of first thread reading B in local workgroup
  int b_bgn = OPENCL_BLOCK_SIZE * tile_x;
  
  // last index of first thread reading B in local workgroup
  int b_end = b_bgn + N*(K-1);

  // step taken by each thread reading B in local workgroup
  int b_stp  = OPENCL_BLOCK_SIZE * N;
      
  // accumulates the result
  T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {

    // local memory for matrix A
    __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

    // local memory for matrix B
    __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];
   
    // each thread in workgroup reads one element of matrix A from global to local memory
    if ( thread_x + global_x < K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[a + M * thread_x + thread_y];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }
    
    // each thread in workgroup reads one element of matrix B from global to local memory
    if ( thread_y + global_y < K ) {
      localMemB[thread_y][thread_x] = B_ptr[b + N * thread_y + thread_x];
    } else { // needed on AMD
      localMemB[thread_y][thread_x] = 0.0;
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    for (int k = 0; k < OPENCL_BLOCK_SIZE; k++) {
      sum += localMemA[thread_y][k] * localMemB[k][thread_x];
    }

    // Synchronize all sub-results
    barrier(CLK_LOCAL_MEM_FENCE);   
  }

  // write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    int c = N * OPENCL_BLOCK_SIZE * tile_y + OPENCL_BLOCK_SIZE * tile_x;
    if (c + N * thread_y + thread_x < M*N ) {
      C_ptr[c + N * thread_y + thread_x] = sum + beta*C_ptr[c + N * thread_y + thread_x];
    }
  }
}
template __attribute__((mangled_name(mmul_TA_NBFloat))) kernel void mmul_TA_NB(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_TA_NBDouble))) kernel void mmul_TA_NB(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_NA_TB(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
    
  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp = OPENCL_BLOCK_SIZE;

  // first index of first thread reading A in local workgroup
  int b_bgn = K * OPENCL_BLOCK_SIZE * tile_x;

  // last index to first thread reading A in local workgroup
  int b_end = b_bgn + K - 1;

  // step taken by each thread reading A
  int b_stp = OPENCL_BLOCK_SIZE;

  // accumulates the result
  T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  
  // each work group moves horizontally over matrix A and vertically over matrix B
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE )  {

    // local memory for matrix A
    __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

    // local memory for matrix B
    __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];
   
    // each thread in workgroup reads one element of matrix A from global to local memory
    if ( thread_x + global_x < K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[a + K * thread_y + thread_x];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }

    // each thread in workgroup reads one element of matrix B from global to local memory
    if ( thread_y + global_y < K ) {
      localMemB[thread_y][thread_x] = B_ptr[b + K * thread_x + thread_y];
    } else { // needed on AMD
      localMemB[thread_y][thread_x] = 0.0;
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);

     // multiply matrix A and B using local memory
     for (int k = 0; k < OPENCL_BLOCK_SIZE; k++) {
       sum += localMemA[thread_y][k] * localMemB[k][thread_x];
     }

     // Synchronize all sub-results
     barrier(CLK_LOCAL_MEM_FENCE);   
  }

  // write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    int c = N * OPENCL_BLOCK_SIZE * tile_y + OPENCL_BLOCK_SIZE * tile_x;
    if (c + N * thread_y + thread_x < M*N ) {
      C_ptr[c + N * thread_y + thread_x] = sum + beta*C_ptr[c + N * thread_y + thread_x];
    }
  }
}
template __attribute__((mangled_name(mmul_NA_TBFloat))) kernel void mmul_NA_TB(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_TBDouble))) kernel void mmul_NA_TB(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE_1D_X x OPENCL_BLOCK_SIZE_1D_Y] elements
 * and satisfies
 * 
 * OPENCL_BLOCK_SIZE_1D_Y % OPENCL_BLOCK_SIZE_1D_X == 0
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE_1D_Y == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE_1D_X
 *   local_size[1] := OPENCL_BLOCK_SIZE_1D_Y
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE_1D_X*OPENCL_BLOCK_SIZE_1D_Y
 */
template <class T> __kernel void mmul_NA_NB_YmodX(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE_1D_X x OPENCL_BLOCK_SIZE_1D_Y]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE_Y * tile_y;
   
  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;
   
  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE_Y;
   
  // first index of first thread reading B in local workgroup
  int b_bgn = OPENCL_BLOCK_SIZE_X * tile_x;
   
  // step taken by each thread reading B in local workgroup
  int b_stp  = OPENCL_BLOCK_SIZE_Y * N;
  
  unsigned int idx = 0;

  // accumulates the result
  T sum = 0.0;
   
  // each work group moves horizontally over matrix A and vertically over matrix B
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp)  {
  
    // local memory for matrix A
    __local T localMemA[OPENCL_BLOCK_SIZE_Y][OPENCL_BLOCK_SIZE_Y];
   
    // local memory for matrix B
    __local T localMemB[OPENCL_BLOCK_SIZE_Y][OPENCL_BLOCK_SIZE_X];
   
    // each thread in workgroup reads several element of matrix A from global to local memory due to the buffer not being quadratic
    for( int i = 0; i < OPENCL_BLOCK_SIZE_Y/OPENCL_BLOCK_SIZE_X; i++ ) {
      idx = a + K * thread_y + thread_x + i*OPENCL_BLOCK_SIZE_X;
      if ( idx < M*K ) {
        localMemA[thread_y][thread_x + i*OPENCL_BLOCK_SIZE_X] = alpha*A_ptr[idx];
      } else { // needed on AMD
        localMemA[thread_y][thread_x + i*OPENCL_BLOCK_SIZE_X] = 0.0;
      }
    }

    // each thread in workgroup reads one element of matrix B from global to local memory
    idx =  b + N * thread_y + thread_x;
    if (idx < K*N ) {
      localMemB[thread_y][thread_x] = B_ptr[idx];
    } else { // needed on AMD
      localMemB[thread_y][thread_x] = 0.0;
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);
   
    // compute limit for loop to stop accumulating when the boundary of the matrix is reached
    int limit = OPENCL_BLOCK_SIZE_Y;
          
    if ( K % OPENCL_BLOCK_SIZE_Y != 0 && a == a_end ) {
      limit = K/OPENCL_BLOCK_SIZE_Y;
    }
          
    if ( K < OPENCL_BLOCK_SIZE_Y ) {
      limit = K;
    }

    // multiply matrix A and B using local memory
    for (int k = 0; k < limit; ++k) {
      sum += localMemA[thread_y][k] * localMemB[k][thread_x];
    }
   
    // Synchronize all sub-results
     barrier(CLK_LOCAL_MEM_FENCE);   
  }
   
  // write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    int c = N * OPENCL_BLOCK_SIZE_Y * tile_y + OPENCL_BLOCK_SIZE_X * tile_x;
    if (c + N * thread_y + thread_x < M*N ) {
      C_ptr[c + N * thread_y + thread_x] = sum + beta*C_ptr[c + N * thread_y + thread_x];
    }
  }
}
template __attribute__((mangled_name(mmul_NA_NB_YmodXFloat))) kernel void mmul_NA_NB_YmodX(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_YmodXDouble))) kernel void mmul_NA_NB_YmodX(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE_1D_X x OPENCL_BLOCK_SIZE_1D_Y] elements
 * and satisfies
 * 
 * OPENCL_BLOCK_SIZE_1D_X % OPENCL_BLOCK_SIZE_1D_Y == 0
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE_1D_Y == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE_1D_X
 *   local_size[1] := OPENCL_BLOCK_SIZE_1D_Y
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE_1D_X*OPENCL_BLOCK_SIZE_1D_Y
 */
template <class T> __kernel void mmul_NA_NB_XmodY(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE_1D_X x OPENCL_BLOCK_SIZE_1D_Y]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
 
  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE_Y * tile_y;
   
  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;
   
  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE_X;
   
  // first index of first thread reading B in local workgroup
  int b_bgn = OPENCL_BLOCK_SIZE_X * tile_x;
   
  // step taken by each thread reading B in local workgroup
  int b_stp  = OPENCL_BLOCK_SIZE_X * N;
  
  unsigned int idx = 0;

  // accumulates the result
  T sum = 0.0;
    
  // each work group moves horizontally over matrix A and vertically over matrix B
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp)  {
  
    // local memory for matrix A
    __local T localMemA[OPENCL_BLOCK_SIZE_Y][OPENCL_BLOCK_SIZE_X];
   
    // local memory for matrix A
    __local T localMemB[OPENCL_BLOCK_SIZE_X][OPENCL_BLOCK_SIZE_X];
   
    // each thread in workgroup reads one element of matrix A from global to local memory
    idx = a + K * thread_y + thread_x;
    if ( idx < M*K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[idx];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }

    // each thread in workgroup reads several element of matrix B from global to local memory due to the buffer not being quadratic
    for( int i = 0; i < OPENCL_BLOCK_SIZE_X/OPENCL_BLOCK_SIZE_Y; i++ ) {
      idx =  b + N * thread_y + thread_x + i*N*OPENCL_BLOCK_SIZE_Y;
      if (idx < K*N ) {
        localMemB[thread_y + i*OPENCL_BLOCK_SIZE_Y][thread_x] = B_ptr[idx];
      } else { // needed on AMD
        localMemB[thread_y + i*OPENCL_BLOCK_SIZE_Y][thread_x] = 0.0;
      }    
    }
    
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);
   
    // compute limit for loop to stop accumulating when the boundary of the matrix is reached
    int limit = OPENCL_BLOCK_SIZE_X;
          
    if ( K % OPENCL_BLOCK_SIZE_X != 0 && a == a_end ) {
       limit = K/OPENCL_BLOCK_SIZE_X;
     }
          
     if ( K < OPENCL_BLOCK_SIZE_X ) {
       limit = K;
     }

     // multiply matrix A and B using local memory
     for (int k = 0; k < limit; ++k) {
       sum += localMemA[thread_y][k] * localMemB[k][thread_x];
     }
   
     // Synchronize all sub-results
     barrier(CLK_LOCAL_MEM_FENCE);   
  }
   
  // write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    int c = N * OPENCL_BLOCK_SIZE_Y * tile_y + OPENCL_BLOCK_SIZE_X * tile_x;
    if (c + N * thread_y + thread_x < M*N ) {
      C_ptr[c + N * thread_y + thread_x] = sum + beta*C_ptr[c + N * thread_y + thread_x];
    }
  }
}
template __attribute__((mangled_name(mmul_NA_NB_XmodYFloat))) kernel void mmul_NA_NB_XmodY(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_XmodYDouble))) kernel void mmul_NA_NB_XmodY(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication for the case when K == 1 using 
 * two local memory buffers of size OPENCL_BLOCK_SIZE_1D_X and 
 * OPENCL_BLOCK_SIZE_1D_Y]
 * 
 * Dimensions:
 *   Matrix A is [Mx1] and A is not transposed
 *   Matrix B is [1xN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE_1D_Y == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE_1D_X
 *   local_size[1] := OPENCL_BLOCK_SIZE_1D_Y
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE_1D_X*OPENCL_BLOCK_SIZE_1D_Y
 */
template <class T> __kernel void mmul_NA_NB_MN1(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // local index of each thread
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
    
  unsigned int idx = 0;
     
  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE_1D_Y];
   
  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE_1D_X];
   
  // each thread in workgroup reads one element of matrix A from global to local memory
  idx = get_global_id(1);
  if ( thread_x == 0 ) {
    if ( idx < M ) {
      localMemA[thread_y]= alpha*A_ptr[idx];
    } else { // needed on AMD
      localMemA[thread_y] = 0.0;
    }
  }

  // each thread in workgroup reads one element of matrix B from global to local memory
  idx = get_global_id(0);
  if ( thread_y == 0 ) {
    if ( idx < N ) {
      localMemB[thread_x] = B_ptr[idx];
    } else { // needed on AMD
      localMemB[thread_x] = 0.0;
    }
  }
    
  // Synchronize the reads of A and B
  barrier(CLK_LOCAL_MEM_FENCE);
   
  // multiply matrix A and matrix B and write all results back to global memory
  if ( get_global_id(0) < N && get_global_id(1) < M ) {
    idx = N * get_global_id(1) + get_global_id(0);
    C_ptr[idx] = localMemA[thread_y]*localMemB[thread_x] + beta*C_ptr[idx];
  }
}
template __attribute__((mangled_name(mmul_NA_NB_MN1Float))) kernel void mmul_NA_NB_MN1(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_MN1Double))) kernel void mmul_NA_NB_MN1(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication for the case when K == 1 without using a single local variable
 * 
 * Dimensions:
 *   Matrix A is [Mx1] and A is not transposed
 *   Matrix B is [1xN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE_1D_X == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE_1D_Y == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE_1D_X
 *   local_size[1] := OPENCL_BLOCK_SIZE_1D_Y
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE_1D_X*OPENCL_BLOCK_SIZE_1D_Y
 */
template <class T> __kernel void mmul_NA_NB_MN1_v2(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // global index of each thread
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
  
  T localVarA = alpha*A_ptr[gy];

  // Synchronize the reads of A
  barrier(CLK_LOCAL_MEM_FENCE);

  // multiply matrix A and matrix B and write all results back to global memory
  if ( gx < N && gy < M ) {
    C_ptr[N * gy + gx] = localVarA*B_ptr[gx] + beta*C_ptr[N * gy + gx];
  }
}
template __attribute__((mangled_name(mmul_NA_NB_MN1_v2Float))) kernel void mmul_NA_NB_MN1_v2(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_MN1_v2Double))) kernel void mmul_NA_NB_MN1_v2(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_NA_NB_v2(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
  
  int global_idx_i = get_global_id(0);
  
  __private T privateMemA[256];
  for( int k = 0; k < 256; k++ ) {
    privateMemA[k] = alpha*A[global_idx_i*K + k];
  }

  __local T localMemB[256];

  for( int idx_n = 0; idx_n < N; idx_n++ ) {
    
    for( int i = 0; i < get_num_groups(0); i++ ) {
      int idx_k = i*get_local_size(0) + get_local_id(0);
      int idx_B = idx_k*N + idx_n;
      
      localMemB[ idx_k ] = B[ idx_B ];    
    }
    barrier(CLK_LOCAL_MEM_FENCE);   
  
    // multiply matrix A and B using local memory
    T sum = 0.0;
    
    for (int k = 0; k < K; k++) {
      sum += privateMemA[k] * localMemB[k];
    }
    C_ptr[ global_idx_i*N + idx_n ] = sum + beta*C_ptr[ global_idx_i*N + idx_n ];
    //barrier(CLK_LOCAL_MEM_FENCE);
  }
}
template __attribute__((mangled_name(mmul_NA_NB_v2Float))) kernel void mmul_NA_NB_v2(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_v2Double))) kernel void mmul_NA_NB_v2(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void mmul_NA_NB_v3(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
  
  int global_idx_i = get_global_id(0);
  
  __private T privateMemA[256];
  for( int k = 0; k < 256; k++ ) {
    privateMemA[k] = alpha*A[global_idx_i*K + k];
  }

  //__local T localMemB[256];

  for( int idx_n = 0; idx_n < N; idx_n++ ) {
    
    int idx_k = global_idx_i;
    int idx_B = idx_k*N + idx_n;
    //localMemB[ idx_k ] = B[ idx_B ];    
    //barrier(CLK_LOCAL_MEM_FENCE);   
  
    // multiply matrix A and B using local memory
    T sum = 0.0;
    
    for (int k = 0; k < K; k++) {
      //sum += privateMemA[k] * localMemB[k];
      int idx_k = global_idx_i;
      int idx_B = idx_k*N + k;

      sum += privateMemA[k] * B[idx_B];
    }
    C_ptr[ global_idx_i*N + idx_n ] = sum + beta*C_ptr[ global_idx_i*N + idx_n ];
    //barrier(CLK_LOCAL_MEM_FENCE);
  }
}
template __attribute__((mangled_name(mmul_NA_NB_v3Float))) kernel void mmul_NA_NB_v3(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_v3Double))) kernel void mmul_NA_NB_v3(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/* 
 * Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 * 
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed
 *   Matrix C is [MxN]
 * 
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *  
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void group_mmul_NA_NB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {

  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);
   
  // local index of each thread inside tile
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // global coordinates for each elemnt in C
  int x = get_global_id(0);
  int y = get_global_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;
  
  int group_size_n  = N / GN;
  int group_n       = x / group_size_n;
  int group_x       = x % group_size_n;  
  int group_y       = y;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;
  
  // first index of first thread reading B in local workgroup
  //int b_bgn = OPENCL_BLOCK_SIZE*tile_x;
  int b_bgn = 0;
  
  // last index of first thread reading B in local workgroup
  //int b_end = b_bgn + group_size_n*(K-1);
  int b_end = K;

  // step taken by each thread reading B in local workgroup
  //int b_stp  = OPENCL_BLOCK_SIZE * group_size_n;
  int b_stp = OPENCL_BLOCK_SIZE;
      
  // accumulates the result
  T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  
  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];
  
  for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
    
    // each thread in workgroup reads one element of matrix A from global to local memory
    if ( (thread_x + global_x) < K  ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[a + K * thread_y + thread_x];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }
      
    // each thread in workgroup reads one element of matrix B from global to local memory
    if ( thread_y + global_y < K ) {      
      int addr = group_n*(group_size_n*K) + group_x + (thread_y+global_y)*group_size_n;
      localMemB[thread_y][thread_x] = B_ptr[addr];
    } else { // needed on AMD
      localMemB[thread_y][thread_x] = 0.0;
    }
   
    // Synchronize the reads of A and B
    barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    for (int k = 0; k < OPENCL_BLOCK_SIZE; k++) {
      sum += localMemA[thread_y][k] * localMemB[k][thread_x];
    }

    // Synchronize all sub-results
    barrier(CLK_LOCAL_MEM_FENCE);   
  }

  // write all results back to global memory
  if ( x < N && y < M ) {

    int addr = group_n*group_size_n*M + y*group_size_n + group_x;
    if (addr < M*N ) {
      C_ptr[addr] = sum + beta*C_ptr[addr];
    }
  }
}
template __attribute__((mangled_name(group_mmul_NA_NBFloat))) kernel void group_mmul_NA_NB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(group_mmul_NA_NBDouble))) kernel void group_mmul_NA_NB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

