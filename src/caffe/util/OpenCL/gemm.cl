#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if defined(cl_amd_printf)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define PRINT_COORDINATES printf("COORDINATES: %dD GS = (", get_work_dim());\
  printf("GS = ");\
  for ( int d_ = 0; d_ < get_work_dim(); d_++ ) {\
    if ( d_ < get_work_dim() - 1 ) {\
      printf("%d x ", get_global_size(d_));\
    } else {\
      printf("%d) GR", get_global_size(d_));\
    }\
  }\
  for ( int d_ = 0; d_ < get_work_dim(); d_++ ) {\
    if ( d_ < get_work_dim() - 1 ) {\
      printf("[%d]", get_group_id(d_));\
    } else {\
      printf("[%d] GID", get_group_id(d_));\
    }\
  }\
  for ( int d_ = 0; d_ < get_work_dim(); d_++ ) {\
    if ( d_ < get_work_dim() - 1 ) {\
      printf("[%d]", get_global_id(d_));\
    } else {\
      printf("[%d] LS = (", get_global_id(d_));\
    }\
  }\
  for ( int d_ = 0; d_ < get_work_dim(); d_++ ) {\
    if ( d_ < get_work_dim() - 1 ) {\
      printf("%d x ", get_local_size(d_));\
    } else {\
      printf("%d) LID", get_local_size(d_));\
    }\
  }\
  for ( int d_ = 0; d_ < get_work_dim(); d_++ ) {\
    if ( d_ < get_work_dim() - 1 ) {\
      printf("[%d]", get_local_id(d_));\
    } else {\
      printf("[%d]\n", get_local_id(d_));\
    }\
  }


#else
#define PRINT_COORDINATES
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
    if ( thread_y + global_y < K && get_global_id(0) < N ) {
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
template <class T> __kernel void mmul_NA_NB_MNS(const int M, const int N, const int K, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {
  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  // int tile_size_x = get_local_size(0);
  // int tile_size_y = get_local_size(1);
  // int tile_x      = get_group_id(0);
  // int tile_y      = get_group_id(1);

  // local index of each thread
  // int thread_x = get_local_id(0);
  // int thread_y = get_local_id(1);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;

  // accumulates the result
  // T sum = 0.0;


#define BS 128
  // local memory for sub-matrix A
  // __local T localMemA[BS][BS];
  // __local T localMemB[BS][BS];

  // local memory for sub-matrix B
  // __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for sub-matrix C
  // __local T localMemC[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // __private T privateMemA[PS][PS];
  // __private T privateMemB[PS][PS];
  // __private T privateMemC[PS][PS];

  // __local T localMemA[1024];
  // __local T localMemB[1024];
  __local T localMemBuffer[BS];
  __private T privateMemA[1024/BS];
  __private T privateMemB[1024/BS];

  int addr;
  int x;
  int y;
  T value = 0.0;
  for (int step_x = 0;  step_x < K; step_x += BS) {
    x = step_x + get_local_id(0);
    y = get_global_id(1);
    addr = K*y + x;

    privateMemA[step_x/BS] = A_ptr[addr];
    privateMemB[step_x/BS] = B_ptr[addr];
    // localMemA[x] = A_ptr[addr];
    // localMemB[x] = B_ptr[addr];
  }

  for (int k = 0; k < 1024/BS; k++ ) {
    value += privateMemA[k]*privateMemB[k];
  }
  localMemBuffer[get_local_id(0)] = value;
  barrier(CLK_LOCAL_MEM_FENCE);

  for ( int mod = 2; mod <= BS; mod *= 2 ) {
    if ( get_local_id(0) % mod == 0 ) {
      localMemBuffer[get_local_id(0)] += localMemBuffer[get_local_id(0)+mod/2];
    }
  }



  /*
  __private sum = 0.0;
  for (int step_x = 0;  step_x < K; step_x += BS) {
    x = step_x + get_local_id(0);
    privateMemA[step_x/BS] = localMemA[x];
    privateMemB[step_x/BS] = localMemB[x];
  }
  // localMemBuffer[get_local_id(0)] = sum;
    */
  /*
  // PRINT_COORDINATES
  for (int step_x = 0;  step_x < N; step_x += BS) {
    for (int step_y = 0; step_y < BS; step_y++ ) {
      x = step_x*BS + get_local_id(0);
      y = BS*get_global_id(1) + step_y;
      addr = K*y + x;
      localMemA[step_y][get_local_id(0)] = A_ptr[addr];
      addr = N*y + x;
      localMemB[get_local_id(0)][step_y] = B_ptr[addr];
    }
  }
  */

  /*
  __private T localMemA[1024];
  __private T privateMemB[1024];

  for (int step_x = 0;  step_x < K; step_x += BS) {
    x = step_x + get_local_id(0);
    y = get_global_id(1);
    addr = K*y + x;
    localMemA[x] = A_ptr[addr];
  }
  */
  /*
  for (int step_x = 0; step_x < N; step_x += BS) {
    x = step_x + get_local_id(0);
    for (int step_y = 0;  step_y < K; step_y++ ) {
      y = step_y;
      addr = N*y + x;
      privateMemB[y] = B_ptr[addr];
    }
    T sum = 0.0;
    // for ( int k = 0; k < K; k++ ) {
    //  sum += privateMemA[k]*privateMemB[k];
    // }
    y = get_global_id(1);
    addr = y*N + x;
    C_ptr[addr] = sum;
  }
  */


 /*
  int steps = 0;
  for (int k = 0; k < K; k += PS ) {
  // for (int a = a_bgn, b = b_bgn; a <= a_end; a += a_stp, b += b_stp, global_x += tile_size_x, global_y += tile_size_x, steps++)  {
      // each thread reads PSxPS elements of Matrix A from global memory to local memory
      for (int j = 0; j < PS; j++ ) {
        for (int i = 0; i < PS; i++ ) {
          int x = i + k;
          int y = get_global_id(1)*PS + j;
          addr = K*y + x;
          // addr = a + K * thread_y + j*K + i;
          // if ( thread_x + global_x + j < K ) {
            privateMemA[j][i] = alpha*A_ptr[addr];
            // printf("GLOBAL[%d][%d] to privateMemA[%d|%d] = %f addr = %d\n", get_global_id(1), get_global_id(0), j, i, privateMemA[j][i], addr);
          // } else { // needed on AMD
          //  privateMemA[j][i] = 0.0;
          //  printf("thread_x = %d global_x = %d j = %d K = %d\n", thread_x, global_x, j, K);
          //  printf("GLOBAL[%d][%d] to privateMemA[%d|%d] = %f addr = %d\n", get_global_id(0), get_global_id(1), j, i, 0.0, addr);
          // }
        }
      }

      // each thread reads PSxPS elements of Matrix B from global memory to local memory
      for (int j = 0; j < PS; j++ ) {
        for (int i = 0; i < PS; i++ ) {
          int x = get_global_id(0)*PS + i;
          int y = j + k;
          addr = N*y + x;
          // addr = b + N * thread_y + thread_x + j*N + i;
          // if ( thread_y + global_y + j < K ) {
            privateMemB[j][i] = B_ptr[addr];
           // printf("GLOBAL[%d][%d] to privateMemB[%d|%d] = %f addr = %d\n", get_global_id(1), get_global_id(0), j, i, privateMemB[j][i], addr);
          // } else { // needed on AMD
          //  privateMemB[j][i] = 0.0;
          //  printf("thread_x = %d global_x = %d j = %d K = %d\n", thread_x, global_x, j, K);
          //  printf("GLOBAL[%d][%d] to privateMemB[%d|%d] = %f addr = %d\n", get_global_id(0), get_global_id(1), j, i, 0.0, addr);
          // }
        }
      }

      // multiply matrix A and B using private memory
      for (int j = 0; j < PS; j++ ) {
        for (int i = 0; i < PS; i++ ) {
          sum = 0.0;
          for (int k = 0; k < PS; k++ ) {
            sum += privateMemA[j][k] * privateMemB[k][i];
          }
          // privateMemC[j][i] = sum;
          int x = get_local_id(0)*PS + i;
          int y = get_local_id(1)*PS + j;
         // printf("GLOBAL[%d][%d] to localMemC[%d|%d] = %f\n", get_global_id(1), get_global_id(0), y, x, privateMemC[j][i]);

          localMemC[y][x] += sum;//privateMemC[j][i];
        }
      }

      // Synchronize the reads of A and B
  }
  // barrier(CLK_LOCAL_MEM_FENCE);

  // copy private to local
  for (int j = 0; j < PS; j++ ) {
    for (int i = 0; i < PS; i++ ) {
      int Gx = get_global_id(0)*PS + i;
      int Gy = get_global_id(1)*PS + j;
      int Lx = get_local_id(0)*PS + i;
      int Ly = get_local_id(1)*PS + j;

      // int x = get_local_size(0)*get_group_id(0)*PS + get_local_id(0)*PS + i;
      // int y = get_local_size(1)*get_group_id(1)*PS + get_local_id(1)*PS + j;
      addr = N*Gy + Gx;

      // addr = N * tile_size_y * tile_y * PS + N*(PS*thread_y + j) + tile_size_x * PS * tile_x + i;
      C_ptr[addr] = localMemC[Ly][Lx];
      // printf("GLOBAL[%d][%d] from localMemC[%d|%d] = %f addr = %d\n", get_global_id(1), get_global_id(0), Ly, Lx, localMemC[Ly][Lx], addr);
    }
  }
  */
}
template __attribute__((mangled_name(mmul_NA_NB_MNSFloat))) kernel void mmul_NA_NB_MNS(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_MNSDouble))) kernel void mmul_NA_NB_MNS(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);


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
  // int a_end   = a_bgn + K - 1;
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
    if ( thread_y + global_y < K && get_global_id(0) < N ) {
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
  // int a_end   = a_bgn + K - 1;
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
     if ( thread_y + global_y < K && get_global_id(0) < N ) {
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
     if ( thread_y + global_y < K && get_global_id(0) < N ) {
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
    for ( int i = 0; i < OPENCL_BLOCK_SIZE_Y/OPENCL_BLOCK_SIZE_X; i++ ) {
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
    for ( int i = 0; i < OPENCL_BLOCK_SIZE_X/OPENCL_BLOCK_SIZE_Y; i++ ) {
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
  int addr = 0;

  __private T privateMemA[256];
  for ( int k = 0; k < 256; k++ ) {
    if ( k < K ) {
      privateMemA[k] = alpha*A[global_idx_i*K + k];
    } else {
      privateMemA[k] = ( T ) 0.0;
    }
  }

  __local T localMemB[256];

  for ( int idx_n = 0; idx_n < N; idx_n++ ) {
    for ( int i = 0; i < get_num_groups(0); i++ ) {
      int idx_k = i*get_local_size(0) + get_local_id(0);
      int idx_B = idx_k*N + idx_n;

      localMemB[ idx_k ] = B[ idx_B ];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    T sum = 0.0;

    for (int k = 0; k < 256; k++) {
      sum += privateMemA[k] * localMemB[k];
    }
    C_ptr[ global_idx_i*N + idx_n ] = sum + beta*C_ptr[ global_idx_i*N + idx_n ];
    // barrier(CLK_LOCAL_MEM_FENCE);
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
  for ( int k = 0; k < 256; k++ ) {
    privateMemA[k] = alpha*A[global_idx_i*K + k];
  }

  // __local T localMemB[256];

  for ( int idx_n = 0; idx_n < N; idx_n++ ) {
    int idx_k = global_idx_i;
    int idx_B = idx_k*N + idx_n;
    // localMemB[ idx_k ] = B[ idx_B ];
    // barrier(CLK_LOCAL_MEM_FENCE);

    // multiply matrix A and B using local memory
    T sum = 0.0;

    for (int k = 0; k < K; k++) {
      // sum += privateMemA[k] * localMemB[k];
      int idx_k = global_idx_i;
      int idx_B = idx_k*N + k;

      sum += privateMemA[k] * B[idx_B];
    }
    C_ptr[ global_idx_i*N + idx_n ] = sum + beta*C_ptr[ global_idx_i*N + idx_n ];
    // barrier(CLK_LOCAL_MEM_FENCE);
  }
}
template __attribute__((mangled_name(mmul_NA_NB_v3Float))) kernel void mmul_NA_NB_v3(const int M, const int N, const int K, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(mmul_NA_NB_v3Double))) kernel void mmul_NA_NB_v3(const int M, const int N, const int K, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/*
 * Group Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed and partitions dimension N into GN parts
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
  int group_x       = ( x % group_size_n );
  int group_y       = y;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;

  // accumulates the result
  __private T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  int addr;
  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  for (int a = a_bgn; a <= a_end; a += a_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
    // each thread in workgroup reads one element of matrix A from global to local memory
    addr = a + K * thread_y + thread_x;

    if ( (thread_x + global_x) < K && addr < M*K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[addr];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }

    // each thread in workgroup reads one element of matrix B from global to local memory
    addr = group_n*(group_size_n*K) + ( x % group_size_n ) + (thread_y+global_y)*group_size_n;
    if ( thread_y + global_y < K  && addr < K*N ) {
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
    addr = group_n*group_size_n*M + y*group_size_n + ( x % group_size_n );
    if (addr < M*N ) {
      C_ptr[addr] = sum + beta*C_ptr[addr];
    }
  }
}
template __attribute__((mangled_name(group_mmul_NA_NBFloat))) kernel void group_mmul_NA_NB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(group_mmul_NA_NBDouble))) kernel void group_mmul_NA_NB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/*
 * Group Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is not transposed
 *   Matrix B is [KxN] and B is not transposed and partitions dimension N into GN parts
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
template <class T> __kernel void group_mmul_NA_TB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {
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
  int group_x       = ( x % group_size_n );
  int group_y       = y;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;

  // accumulates the result
  __private T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  int addr;
  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  for (int a = a_bgn; a <= a_end; a += a_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
    // each thread in workgroup reads one element of matrix A from global to local memory
    addr = a + K * thread_y + thread_x;

    if ( (thread_x + global_x) < K && addr < M*K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[addr];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }

    // each thread in workgroup reads one element of matrix B from global to local memory
    // addr = group_n*(group_size_n*K) + ( x % group_size_n ) + (thread_y+global_y)*group_size_n;
    addr = group_n*(group_size_n*K) + ( x % group_size_n ) * K + (thread_y+global_y);
    if ( thread_y + global_y < K  && addr < K*N ) {
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
    addr = group_n*group_size_n*M + y*group_size_n + ( x % group_size_n );
    if (addr < M*N ) {
      C_ptr[addr] = sum + beta*C_ptr[addr];
    }
  }
}
template __attribute__((mangled_name(group_mmul_NA_TBFloat))) kernel void group_mmul_NA_TB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(group_mmul_NA_TBDouble))) kernel void group_mmul_NA_TB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/*
 * Group Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxK] and A is transposed
 *   Matrix B is [KxN] and B is transposed and partitions dimension N into GN parts
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
template <class T> __kernel void group_mmul_TA_TB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {
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
  int group_x       = ( x % group_size_n );
  int group_y       = y;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;

  // accumulates the result
  __private T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  int addr;
  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  for (int a = a_bgn; a <= a_end; a += a_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
    // each thread in workgroup reads one element of matrix A from global to local memory
    // addr = a + K * thread_y + thread_x;
    addr = a + thread_y + (thread_x*K);

    if ( (thread_x + global_x) < K && addr < M*K ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[addr];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }

    // each thread in workgroup reads one element of matrix B from global to local memory
    // addr = group_n*(group_size_n*K) + ( x % group_size_n ) + (thread_y+global_y)*group_size_n;
    addr = group_n*(group_size_n*K) + ( x % group_size_n ) * K + (thread_y+global_y);
    if ( thread_y + global_y < K  && addr < K*N ) {
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
    addr = group_n*group_size_n*M + y*group_size_n + ( x % group_size_n );
    if (addr < M*N ) {
      C_ptr[addr] = sum + beta*C_ptr[addr];
    }
  }
}
template __attribute__((mangled_name(group_mmul_TA_TBFloat))) kernel void group_mmul_TA_TB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(group_mmul_TA_TBDouble))) kernel void group_mmul_TA_TB(const int M, const int N, const int K, const int GM, const int GN, const int GK, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

/*
 * Group_3D Matrix-Matrix-Multiplication using local memory as a buffer
 * that has [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE] elements
 *
 * Dimensions:
 *   Matrix A is [MxKxG] and A is not transposed
 *   Matrix B is [KxNxG] and B is not transposed and partitions dimension N into GN parts
 *   Matrix C is [MxNxG]
 *
 * Global Index Space
 *   global_size[0] := global_size[0] % OPENCL_BLOCK_SIZE == 0 && global_size[0] >= N
 *   global_size[1] := global_size[1] % OPENCL_BLOCK_SIZE == 0 && global_size[1] >= M
 *   global_size[2] := G
 *
 * Local Index Space
 *   local_size[0] := OPENCL_BLOCK_SIZE
 *   local_size[1] := OPENCL_BLOCK_SIZE
 *   local_size[2] := 1
 *
 * Number of Threads in each local workgroup
 *   localThreadCount := OPENCL_BLOCK_SIZE*OPENCL_BLOCK_SIZE
 */
template <class T> __kernel void group_3D_mmul_NA_NB(const int M, const int N, const int K, const int G, const int GM, const int GN, const int GK, const T alpha, global T* A, const unsigned long idx_offset_A, global T* B, const unsigned long idx_offset_B, const T beta, global T* C, const unsigned long idx_offset_C) {
  // coordinates for each tile of [OPENCL_BLOCK_SIZE x OPENCL_BLOCK_SIZE]
  int tile_x = get_group_id(0);
  int tile_y = get_group_id(1);

  // local index of each thread inside tile
  int thread_x = get_local_id(0);
  int thread_y = get_local_id(1);

  // global coordinates for each elemnt in C
  int x = get_global_id(0);
  int y = get_global_id(1);
  int g = get_global_id(2);

  // offset pointers in global memory
  global T* A_ptr = A + idx_offset_A;
  global T* B_ptr = B + idx_offset_B;
  global T* C_ptr = C + idx_offset_C;

  int group_size_n  = N / GN;
  int group_n       = x / group_size_n;
  int group_x       = ( x % group_size_n );
  int group_y       = y;

  // first index of first thread reading A in local workgroup
  int a_bgn = K * OPENCL_BLOCK_SIZE * tile_y;

  // last index to first thread reading A in local workgroup
  int a_end   = a_bgn + K - 1;

  // step taken by each thread reading A
  int a_stp  = OPENCL_BLOCK_SIZE;

  // accumulates the result
  __private T sum = 0.0;

  int global_x = 0;
  int global_y = 0;
  int addr;
  // local memory for matrix A
  __local T localMemA[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  // local memory for matrix B
  __local T localMemB[OPENCL_BLOCK_SIZE][OPENCL_BLOCK_SIZE];

  for (int a = a_bgn; a <= a_end; a += a_stp, global_x += OPENCL_BLOCK_SIZE, global_y += OPENCL_BLOCK_SIZE)  {
    // each thread in workgroup reads one element of matrix A from global to local memory
    addr = a + M*K*g + K * thread_y + thread_x;

    if ( (thread_x + global_x) < K && addr < M*K*G ) {
      localMemA[thread_y][thread_x] = alpha*A_ptr[addr];
    } else { // needed on AMD
      localMemA[thread_y][thread_x] = 0.0;
    }

    // each thread in workgroup reads one element of matrix B from global to local memory
    //addr = group_n*(group_size_n*K) + ( x % group_size_n ) + (thread_y+global_y)*group_size_n;
    addr = group_n*(group_size_n*K)*G + (group_size_n*K)*g + ( x % group_size_n ) + (thread_y+global_y)*group_size_n;
    if ( thread_y + global_y < K  && addr < K*N*G ) {
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
    //addr = group_n*group_size_n*M + y*group_size_n + ( x % group_size_n );
    addr = (group_n*group_size_n*M)*G + (group_size_n*M)*g + y*group_size_n + ( x % group_size_n );
    if (addr < M*N*G ) {
      C_ptr[addr] = sum + beta*C_ptr[addr];
    }
  }
}
template __attribute__((mangled_name(group_3D_mmul_NA_NBFloat))) kernel void group_3D_mmul_NA_NB(const int M, const int N, const int K, const int G, const int GM, const int GN, const int GK, const float alpha, global float* A, const unsigned long idx_offset_A, global float* B, const unsigned long idx_offset_B, const float beta, global float* C, const unsigned long idx_offset_C);
template __attribute__((mangled_name(group_3D_mmul_NA_NBDouble))) kernel void group_3D_mmul_NA_NB(const int M, const int N, const int K, const int G, const int GM, const int GN, const int GK, const double alpha, global double* A, const unsigned long idx_offset_A, global double* B, const unsigned long idx_offset_B, const double beta, global double* C, const unsigned long idx_offset_C);

