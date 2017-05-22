void TEMPLATE(matvec_mul_trail_rows,Dtype)(unsigned int M,
                           unsigned int N,
                           int row_gid,
                           int lid,
                           const __global Dtype* src0_read,
                           int lda,
                           const __global Dtype* src1_read,
                           int incv,
                           __local Dtype4* work,
                           Dtype alpha,
                           Dtype beta,
                           __global Dtype* result,
                           int incr)
{
  __local Dtype* work_each = (__local Dtype*)work;

  int rows = M - row_gid * 4;

  Dtype4 dot[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};

  int i = lid;
  while( i < N / 4) {
    const Dtype4 b0 = {src1_read[i*4*incv], src1_read[(i*4+1)*incv], src1_read[(i*4+2)*incv], src1_read[(i*4+3)*incv]};
#pragma unroll
    for(int j = 0; j < rows; ++j) {
      dot[j] += b0 * vload4(i, src0_read + j * lda);
    }

    i += get_local_size(0);
  }
#pragma unroll
  for(int j = 0; j < rows; ++j) {
    work_each[lid * 4 + j] = dot[j].x + dot[j].y + dot[j].z + dot[j].w;
  }

  if(i == N / 4) {
    short trail_item = N % 4;

    if(trail_item != 0) {
      const __global Dtype *src0_trail = src0_read + i * 4;
      const __global Dtype *src1_trail = src1_read + i * 4 * incv;
#pragma unroll
      for(short i = 0; i < trail_item; ++i) {
        const Dtype bt = src1_trail[i*incv];
#pragma unroll
        for(int j = 0; j < rows; ++j) {
          work_each[lid * 4 + j] += bt * src0_trail[i + j * lda];
        }
      }
    }
  }

  for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < stride)
      work[lid] += work[lid+stride];
  }

  if(lid == 0) {
#pragma unroll
    for(int j = 0; j < rows; ++j) {
      result[(row_gid * 4  + j) * incr] = alpha * work_each[j] + beta * result[(row_gid * 4 + j) * incr];
    }
  }

}

__kernel void TEMPLATE(matvec_mul,Dtype)(
          unsigned int M,
          unsigned int N,
          __global const Dtype * A,
          int offA,
          int lda,
          __global const Dtype * v,
          int offv,
          int incv,
          KERNEL_ARG_DTYPE alpha,
          KERNEL_ARG_DTYPE beta,
          __global Dtype * result,
          int offr,
          int incr)
{
  int row_gid = get_group_id(0);
  int lid = get_local_id(0);
  const __global Dtype *src0_read = A + row_gid * 4 * lda + offA;
  const __global Dtype *src1_read = v + offv;
  result = result + offr;

  src1_read += incv > 0 ? 0 : (1 - N) * incv;
  result += incr > 0 ? 0 : (1 - M) * incr;
  __local Dtype4 work[128];
  __local Dtype* work_each = (__local Dtype*)work;

  if(row_gid == M / 4)
    TEMPLATE(matvec_mul_trail_rows,Dtype)(M, N, row_gid, lid, src0_read, lda, src1_read, incv, work, alpha, beta, result, incr);
  else
  {
    Dtype4 dot[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.f), (Dtype4)(0.f)};
    int i = lid;
    while( i < N / 4) {
      const Dtype4 b0 = {src1_read[i*4*incv], src1_read[(i*4+1)*incv], src1_read[(i*4+2)*incv], src1_read[(i*4+3)*incv]};
#pragma unroll
      for(int j = 0; j < 4; ++j) {
        dot[j] += b0 * vload4(i, src0_read + j * lda);
      }
      i += get_local_size(0);
    }
#pragma unroll
    for(int j = 0; j < 4; ++j) {
      work_each[lid * 4 + j] = dot[j].x + dot[j].y + dot[j].z + dot[j].w;
    }

    if(i == N / 4) {
      short trail_item = N % 4;
      if(trail_item != 0) {
        const __global Dtype *src0_trail = src0_read + i * 4;
        const __global Dtype *src1_trail = src1_read + i * 4 * incv;
#pragma unroll
        for(short i = 0; i < trail_item; ++i) {
          const Dtype bt = src1_trail[i * incv];
#pragma unroll
          for(int j = 0; j < 4; ++j) {
            work_each[lid * 4 + j] += bt * src0_trail[i + j * lda];
          }
        }
      }
    }

    for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lid < stride)
        work[lid] += work[lid+stride];
    }

    if(lid == 0) {
      // vstore4(alpha * work[0] + beta * vload4(row_gid, result), row_gid, result);
      result[row_gid*4*incr] = alpha * work[0].s0 + beta * result[row_gid*4*incr];
      result[(row_gid*4+1)*incr] = alpha * work[0].s1 + beta * result[(row_gid*4+1)*incr];
      result[(row_gid*4+2)*incr] = alpha * work[0].s2 + beta * result[(row_gid*4+2)*incr];
      result[(row_gid*4+3)*incr] = alpha * work[0].s3 + beta * result[(row_gid*4+3)*incr];
    }
  }
}

__kernel void TEMPLATE(trans_matvec_mul,Dtype)(
          unsigned int M,
          unsigned int N,
          __global const Dtype * A,
          int offA,
          int lda,
          __global const Dtype * v,
          int offv,
          int incv,
          KERNEL_ARG_DTYPE alpha,
          KERNEL_ARG_DTYPE beta,
          __global Dtype * result,
          int offr,
          int incr)
{
  int col_gid = get_global_id(0);
  A += offA + col_gid;
  v += offv;
  result += offr;

  v += incv > 0 ? 0 : (1 - M) * incv;
  result += incr > 0 ? 0 : (1 - N) * incr;

  Dtype dot_prod = 0;
  int row_id = 0;
#pragma unroll
  for(int row = 0; row < M; ++row) {
    dot_prod += A[row_id] * v[row * incv];
    row_id += lda;
  }
  result[col_gid * incr] = beta * result[col_gid * incr];
  result[col_gid * incr] += alpha * dot_prod;
}
