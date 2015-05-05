#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

/*
// Very naive implementation
__kernel void TEMPLATE(convolution_ip4v0,Dtype)(__global const float *w,
                                                __global const float *in,
                                                const int height,
                                                const int width,
                                                __global float *out) {

  // TESTING
  // Current tests for IP4.
  // 10x10 kernel,

  const int kernel_h = 10;
  const int kernel_w = 10;
  const int ext_kernel_h = 73;
  const int ext_kernel_w = 73;
  const int fout_count = 1024;
  const int fin_count = 192;
  const int kstride_h = 8;
  const int kstride_w = 8;
  const int stride_h = 1;
  const int stride_w = 1;

  const int out_h = (height - ext_kernel_h) / stride_h + 1;
  const int out_w = (width - ext_kernel_w) / stride_w + 1;

  // Across y-dimension
  for (int yoff = get_global_id(2); yoff < height - ext_kernel_h + 1; yoff +=
      get_global_size(2)) {

    // Across x-dimension
    for (int xoff = get_global_id(1); xoff < width - ext_kernel_w + 1; xoff +=
        get_global_size(1)) {

      // Across output features
      for (int fout = get_global_id(0); fout < fout_count; fout +=
          get_global_size(0)) {

        int fout_w_ptr = fout * fin_count * kernel_h * kernel_w;

        // Across input features
        float outval = 0.0;
        for (int fin = 0; fin < fin_count; ++fin) {
          // Across the kernel itself
          int fin_ptr = fin * width * height;
          int finout_w_ptr = fin * kernel_h * kernel_w + fout_w_ptr;
          for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {

              outval += w[j + i * kernel_w + finout_w_ptr]
                  * in[(xoff + j * kstride_w) + (yoff + i * kstride_h) * width
                      + fin_ptr];
            }
          }
        }
        out[xoff + yoff * out_w + fout * out_w * out_h] = outval;
      }
    }
  }
}

// More optimized
__kernel void TEMPLATE(convolution_ip4v1,Dtype)(__global const Dtype *w,
                                                __global const Dtype *in,
                                                __global float *out) {

  // Kernel uses Z-workers across batches and output features
  // Y-workers across Y-input
  // X-workers across X-input

  // TESTING
  // Current tests for IP4.
  // 10x10 kernel,

  const int width = 200;
  const int height = 200;
  const int kernel_h = 10;
  const int kernel_w = 10;
  const int fout_count = 1024;
  const int fin_count = 192;
  const int kstride_h = 8;
  const int kstride_w = 8;
  const int stride_h = 1;
  const int stride_w = 1;
  const int batch_size = 1;

  const int ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
  const int ext_kernel_w = (kernel_w - 1) * kstride_w + 1;

  const int out_h = (height - ext_kernel_h) / stride_h + 1;
  const int out_w = (width - ext_kernel_w) / stride_w + 1;

  const int fin_fraction = 16;

  // Clear the output
  {
#pragma unroll 1
    for (int i =
        get_global_id(
            0)+get_global_id(1)*get_global_size(0)+get_global_id(2)*get_global_size(0)*get_global_size(1);
        i < fout_count * out_h * out_w;
        i += get_global_size(0) * get_global_size(1) * get_global_size(2)) {
      out[i] = 0.0;
    }
  }

  // Local weight buffer
  __local Dtype wl[16 * 100];

  // Across output features
#pragma unroll 1
  for (int fout = get_global_id(2); fout < fout_count;
      fout += get_global_size(2)) {

    const int fout_w_ptr = fout * fin_count * kernel_h * kernel_w;

    // Across input features
#pragma unroll 1
    for (int fin = 0; fin < fin_count; fin += fin_fraction) {
      const int finout_w_ptr = fin * kernel_h * kernel_w + fout_w_ptr;

      // Load local weights
      // TODO: Correction for non-fitting fraction divisors
#pragma unroll 1
      for (int k = 0; k < fin_fraction; ++k) {
#pragma unroll 1
        for (int i = get_local_id(1); i < kernel_h; i += get_local_size(1)) {
#pragma unroll 1
          for (int j = get_local_id(0); j < kernel_w; j += get_local_size(0)) {
            wl[j + i * kernel_w + k * kernel_w * kernel_h] = w[j + i * kernel_w
                + k * kernel_w * kernel_h + finout_w_ptr];
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      // Across batches (keeps weights in local memory)
#pragma unroll 1
      for (int batch = 0; batch < batch_size; ++batch) {

        const int batch_in_off = batch * width * height * fin_count;
        const int batch_out_off = batch * out_w * out_h * fout_count;

        // Across a fraction of input features
#pragma unroll 1
        for (int finoff = 0; finoff < fin_fraction; ++finoff) {
          const int finoff_ptr = (finoff + fin) * width * height;

          // Across y-dimension
#pragma unroll 1
          for (int yoff = get_global_id(1); yoff < height - ext_kernel_h + 1;
              yoff += get_global_size(1)) {

            // Across x-dimension
#pragma unroll 1
            for (int xoff = get_global_id(0); xoff < width - ext_kernel_w + 1;
                xoff += get_global_size(0)) {

              Dtype outval = out[xoff + yoff * out_w + fout * out_w * out_h
                  + batch_out_off];

              // Across the kernel itself
#pragma unroll 2
              for (int i = 0; i < kernel_h; ++i) {
#pragma unroll 2
                for (int j = 0; j < kernel_w; ++j) {
                  outval = fma(
                      wl[j + i * kernel_w + finoff * kernel_w * kernel_h],
                      in[(xoff + j * kstride_w) + (yoff + i * kstride_h) * width
                          + finoff_ptr + batch_in_off],
                      outval);
                }
              }

              out[xoff + yoff * out_w + fout * out_w * out_h + batch_out_off] =
                  outval;
            }
          }
        }
      }barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}*/

// Fits into 32 KB
__kernel void TEMPLATE(convolution_ip4v2,Dtype)(__global const Dtype *w,
                                                __global const Dtype *in,
                                                __global float *out) {
  const int width = 584;
  const int height = 584;
  const int kernel_h = 10;
  const int kernel_w = 10;
  const int fout_count = 1024;
  const int fin_count = 192;
  const int kstride_h = 8;
  const int kstride_w = 8;
  const int stride_h = 1;
  const int stride_w = 1;
  const int batch_size = 1;

  const int ext_kernel_h = (kernel_h - 1) * kstride_h + 1;
  const int ext_kernel_w = (kernel_w - 1) * kstride_w + 1;

  const int out_h = (height - ext_kernel_h) / stride_h + 1;
  const int out_w = (width - ext_kernel_w) / stride_w + 1;

  // Clear the output
  {
#pragma unroll 1
    for (int i =
        get_global_id(
            0)+get_global_id(1)*get_global_size(0)+get_global_id(2)*get_global_size(0)*get_global_size(1);
        i < fout_count * out_h * out_w;
        i += get_global_size(0) * get_global_size(1) * get_global_size(2)) {
      out[i] = 0.0;
    }
  }

  // Local weight buffer
  __local Dtype wl[10*10];

  // Across output features
#pragma unroll 1
  for (int fout = get_global_id(2); fout < fout_count;
      fout += get_global_size(2)) {

    // Across input features
#pragma unroll 1
    for (int fin = 0; fin < fin_count; ++fin) {

      // Load local weights
#pragma unroll 1
        for (int i = get_local_id(1); i < kernel_h; i += get_local_size(1)) {
#pragma unroll 1
          for (int j = get_local_id(0); j < kernel_w; j += get_local_size(0)) {
            wl[i+j*kernel_w] = w[j + i * kernel_w + fout * fin_count * kernel_h * kernel_w + fin * kernel_h * kernel_w];
          }
        }

      barrier(CLK_LOCAL_MEM_FENCE);

      // Across batches (keeps weights in local memory)
#pragma unroll 1
      for (int batch = 0; batch < batch_size; ++batch) {

        const int batch_in_off = batch * width * height * fin_count;
        const int batch_out_off = batch * out_w * out_h * fout_count;

          // Across y-dimension
#pragma unroll 1
          for (int yoff = get_global_id(1); yoff < height - ext_kernel_h + 1;
              yoff += get_global_size(1)) {

            // Across x-dimension
#pragma unroll 1
            for (int xoff = get_global_id(0); xoff < width - ext_kernel_w + 1;
                xoff += get_global_size(0)) {

              Dtype outval = out[xoff + yoff * out_w + fout * out_w * out_h
                  + batch_out_off];

              // Across the kernel itself
#pragma unroll 5
              for (int i = 0; i < kernel_h; ++i) {
#pragma unroll 5
                for (int j = 0; j < kernel_w; ++j) {
                  outval = fma(
                      wl[i+j*kernel_w],
                      in[(xoff + j * kstride_w) + (yoff + i * kstride_h) * width
                          + fin*width*height + batch_in_off],
                      outval);
                }
              }

              out[xoff + yoff * out_w + fout * out_w * out_h + batch_out_off] =
                  outval;
            }
        }
      } barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
