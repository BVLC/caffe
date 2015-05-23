#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

/*__kernel void TEMPLATE(convolution_ip4v3,Dtype)(__global const Dtype *w,
                                                __global const Dtype *in,
                                                __global Dtype *out) {

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
  const int buff_w = 73;
  const int buff_h = 73;

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

  // Local weight buffer (in local memory)
  __local Dtype wl[10 * 10];
  // Local input buffer (in local memory)
  __local Dtype il[73 * 73];
  // Local accumulators (in registers)
  Dtype al[2 * 2];

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
          wl[j + i * kernel_w] = w[j + i * kernel_w
              + fout * fin_count * kernel_h * kernel_w
              + fin * kernel_h * kernel_w];
        }
      }

      // Across batches (keeps weights in local memory)
#pragma unroll 1
      for (int batch = 0; batch < batch_size; ++batch) {

        const int batch_in_off = batch * width * height * fin_count;
        const int batch_out_off = batch * out_w * out_h * fout_count;

        // Shift the patch window across width and height
        for (int yoff = 0; yoff < height; yoff += buff_h) {
          for (int xoff = 0; xoff < width; xoff += buff_w) {

            // Load image patch
#pragma unroll 1
            for (int i = get_local_id(1); i < buff_h; i += get_local_size(1)) {
              int yidx = (i + yoff);
#pragma unroll 1
              for (int j = get_local_id(0); j < buff_w;
                  j += get_local_size(0)) {
                int xidx = (j + xoff);
                if (xidx < width && yidx < height) {
                  il[j + i * buff_w] = in[xidx + yidx * width
                      + fin * width * height + batch_in_off];
                }
              }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Kernel inner loop
#pragma unroll 1
            for (int i = get_local_id(1); i < buff_h; i += get_local_size(1)) {
#pragma unroll 1
              for (int j = get_local_id(0); j < buff_w;
                  j += get_local_size(0)) {

                // Load accumulators
#pragma unroll 1
                for (int k = 0; k < 4; k++) {
                  int xidx = (j + xoff - k % 2 * buff_w);
                  int yidx = (i + yoff - k / 2 * buff_h);
                  if (xidx >= 0 && xidx < out_w && yidx >= 0 && yidx < out_h) {
                    al[k] = out[xidx + yidx * out_w + fout * out_w * out_h
                        + batch_out_off];
                  }
                }

#pragma unroll 1
                for (int ki = 0; ki < kernel_h; ++ki) {
                  int ilpos_i = ((i + ki * kstride_h) % buff_h) * buff_w;
                  int alpos_i = (i + ki * kstride_h) / buff_h * 2;
#pragma unroll 10
                  for (int kj = 0; kj < kernel_w; ++kj) {
                    al[(j + kj * kstride_w) / buff_w + alpos_i] += wl[kj
                        + ki * kernel_w]
                        * il[(j + kj * kstride_w) % buff_w + ilpos_i];
                  }
                }

                // Store accumulators
#pragma unroll 1
                for (int k = 0; k < 4; k++) {
                  int xidx = (j + xoff - k % 2 * buff_w);
                  int yidx = (i + yoff - k / 2 * buff_h);
                  if (xidx >= 0 && xidx < out_w && yidx >= 0 && yidx < out_h) {
                    out[xidx + yidx * out_w + fout * out_w * out_h
                        + batch_out_off] = al[k];
                  }
                }
              }
            }barrier(CLK_LOCAL_MEM_FENCE);
          }
        }
      }
    }
  }
}*/

// Fits into 32 KB
__kernel void TEMPLATE(convolution_ip4v2,Dtype)(__global const Dtype *w,
                                                __global const Dtype *in,
                                                __global Dtype *out) {
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
  __local Dtype wl[10 * 10];

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
          wl[j + i * kernel_w] = w[j + i * kernel_w
              + fout * fin_count * kernel_h * kernel_w
              + fin * kernel_h * kernel_w];
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
#pragma unroll 10
            for (int i = 0; i < kernel_h; ++i) {
#pragma unroll 10
              for (int j = 0; j < kernel_w; ++j) {
                outval = fma(
                    wl[j + i * kernel_w],
                    in[(xoff + j * kstride_w) + (yoff + i * kstride_h) * width
                        + fin * width * height + batch_in_off],
                    outval);
              }
            }

            out[xoff + yoff * out_w + fout * out_w * out_h + batch_out_off] =
                outval;
          }
        }
      }barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
