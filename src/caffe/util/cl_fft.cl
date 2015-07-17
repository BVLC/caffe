#define DtypeComplex Dtype2

__kernel void copy2buffer_cyclic_shift_in(
    __global Dtype* fft_gpu_weights_real, const int offset_fft_gpu_weights_real,
    __global Dtype* weight, const int offset_weight,
    const int ker_size, const int ch_gr, const int ker_size_ch_gr,
    const int ker_w, const int ker_c_h, const int ker_c_w, 
    const int fft_height, const int fft_width, const int complex_w_len) {
  fft_gpu_weights_real += offset_fft_gpu_weights_real;
  weight += offset_weight;
  int gId = get_global_id(0);
  int out = gId / ker_size_ch_gr;
  int c = (gId - out * ker_size_ch_gr) / ker_size;
  int map_offset = out * ch_gr + c;
  int map_offset_ker_size = map_offset * ker_size;
  int pos_in_map = gId - map_offset_ker_size;
  int h = pos_in_map / ker_w;
  int h_ker_w = h * ker_w;
  int w = pos_in_map - h_ker_w;
  int src_idx = map_offset_ker_size + h_ker_w + w;
  int ky = h - ker_c_h;
  if (ky < 0) ky += fft_height;
  int kx = w - ker_c_w;
  if (kx < 0) kx += fft_width;
  int dst_idx = (map_offset * fft_height + ky) * complex_w_len + kx;
  fft_gpu_weights_real[dst_idx] = weight[src_idx];
}

/* Use when width < 4 */
__kernel void copy2buffer_left_top_in_naive(__global Dtype* map_out,
    const int offset_map_out, 
    const __global Dtype* map_in, const int offset_map_in, 
    const int size, 
    const int height_out, const int width_out, 
    const int height, const int width, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int h = gId / width;
  int w = gId - (h * width);
  int dst_idx = (h*stride_h + pad_h)*width_out + (w*stride_w + pad_w);
  map_out[dst_idx] = map_in[gId];
}

/* Use when width < 4 */
__kernel void copy2buffer_left_top_in_naive_2d(__global Dtype* map_out,
    const int offset_map_out, 
    const __global Dtype* map_in, const int offset_map_in, 
    const int map_out_size, const int size, const int count,
    const int height_out, const int width_out, 
    const int height, const int width, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId_x = get_global_id(0);
  int gId_y = get_global_id(1); 
  int h = gId_x / width;
  int w = gId_x - (h * width);
  int src_idx = gId_y * size + gId_x;
  int dst_idx = gId_y * map_out_size + 
      (h * stride_h + pad_h) * width_out + (w * stride_w + pad_w);
  map_out[dst_idx] = map_in[src_idx];
}

/* Use when width >= 4 */
__kernel void copy2buffer_left_top_in(__global Dtype* map_out,
    const int offset_map_out,
    const __global Dtype* map_in, const int offset_map_in,
    const int size,
    const int height_out, const int width_out, 
    const int height, const int width, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int count = size >> 2;
  int gId4 = gId << 2;
  int h = gId4 / width;
  int w = gId4 - (h * width);
  int dst_h = h*stride_h + pad_h;
  int dst_w = w*stride_w + pad_w;
  int dst_idx = dst_h*width_out + dst_w;
  if (gId < count) {
    Dtype4 map_in_cache4 = vload4(gId, map_in);
    int has_pad = width - dst_w; 
    if (has_pad >= 4) {
      vstore4(map_in_cache4, dst_idx >> 2, map_out);
    } else { 
      if (0 == has_pad) {
        dst_idx += width_out + pad_w - dst_w;
      }
      map_out[dst_idx] = map_in_cache4.x;
      if (1 == has_pad) {
        dst_idx += width_out + pad_w - dst_w - 1;
      }
      map_out[dst_idx+1] = map_in_cache4.y;
      if (2 == has_pad) {
        dst_idx += width_out + pad_w - dst_w - 2;
      }
      map_out[dst_idx+2] = map_in_cache4.z;
      if (3 == has_pad) {
        dst_idx += width_out + pad_w - dst_w - 3;
      }
      map_out[dst_idx+3] = map_in_cache4.w;
      dst_h += 1;
      dst_w = pad_w;
    }
  } else if (gId == count) {
    int res = size - (count << 2); /* size % 4 */
    if (res > 0) {
      Dtype4 map_in_cache4 = 0.f;
      if (res >= 1) 
        map_in_cache4.x = map_in[gId4];
      if (res >= 2)
        map_in_cache4.y = map_in[gId4+1];
      if (res == 3)
        map_in_cache4.z = map_in[gId4+2];
      int has_pad = width - dst_w; 
      if (has_pad >= 4) {
        vstore4(map_in_cache4, dst_idx >> 2, map_out);
      } else { 
        if (0 == has_pad) {
          dst_idx += width_out + pad_w - dst_w;
        }
        map_out[dst_idx] = map_in_cache4.x;
        if (1 == has_pad) {
          dst_idx += width_out + pad_w - dst_w - 1;
        }
        map_out[dst_idx+1] = map_in_cache4.y;
        if (2 == has_pad) {
          dst_idx += width_out + pad_w - dst_w - 2;
        }
        map_out[dst_idx+2] = map_in_cache4.z;
        if (3 == has_pad) {
          dst_idx += width_out + pad_w - dst_w - 3;
        }
        map_out[dst_idx+3] = map_in_cache4.w;
        dst_h += 1;
        dst_w = pad_w;
      }
    }
  }
}

/* Use when width >= 4 */
__kernel void copy2buffer_left_top_in_2d(__global Dtype* map_out,
    const int offset_map_out,
    const __global Dtype* map_in, const int offset_map_in,
    const int map_out_size, const int size, const int count,
    const int height_out, const int width_out, 
    const int height, const int width, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int gId_y = get_global_id(1);
  int gId4 = gId << 2;
  int h = gId4 / width;
  int w = gId4 - (h * width);
  int dst_h = h*stride_h + pad_h;
  int dst_w = w*stride_w + pad_w;
  int dst_idx = dst_h*width_out + dst_w;
  const __global Dtype* map_in_2d = map_in + gId_y * size;
  __global Dtype* map_out_2d = map_out + gId_y * map_out_size;
  if (gId < count) {
    Dtype4 map_in_cache4 = vload4(gId, map_in_2d);
    int has_pad = width - dst_w; 
    if (has_pad >= 4) {
      vstore4(map_in_cache4, dst_idx >> 2, map_out_2d);
    } else { 
      if (0 == has_pad) {
        dst_idx += width_out + pad_w - dst_w;
      }
      map_out_2d[dst_idx] = map_in_cache4.x;
      if (1 == has_pad) {
        dst_idx += width_out + pad_w - dst_w - 1;
      }
      map_out_2d[dst_idx+1] = map_in_cache4.y;
      if (2 == has_pad) {
        dst_idx += width_out + pad_w - dst_w - 2;
      }
      map_out_2d[dst_idx+2] = map_in_cache4.z;
      if (3 == has_pad) {
        dst_idx += width_out + pad_w - dst_w - 3;
      }
      map_out_2d[dst_idx+3] = map_in_cache4.w;
      dst_h += 1;
      dst_w = pad_w;
    }
  } else if (gId == count) {
    int res = size - (count << 2); /* size % 4 */
    if (res > 0) {
      Dtype4 map_in_cache4 = 0.f;
      if (res >= 1) 
        map_in_cache4.x = map_in_2d[gId4];
      if (res >= 2)
        map_in_cache4.y = map_in_2d[gId4+1];
      if (res == 3)
        map_in_cache4.z = map_in_2d[gId4+2];
      int has_pad = width - dst_w; 
      if (has_pad >= 4) {
        vstore4(map_in_cache4, dst_idx >> 2, map_out_2d);
      } else { 
        if (0 == has_pad) {
          dst_idx += width_out + pad_w - dst_w;
        }
        map_out_2d[dst_idx] = map_in_cache4.x;
        if (1 == has_pad) {
          dst_idx += width_out + pad_w - dst_w - 1;
        }
        map_out_2d[dst_idx+1] = map_in_cache4.y;
        if (2 == has_pad) {
          dst_idx += width_out + pad_w - dst_w - 2;
        }
        map_out_2d[dst_idx+2] = map_in_cache4.z;
        if (3 == has_pad) {
          dst_idx += width_out + pad_w - dst_w - 3;
        }
        map_out_2d[dst_idx+3] = map_in_cache4.w;
        dst_h += 1;
        dst_w = pad_w;
      }
    }
  }
}

/* Use when width_out < 4 */
__kernel void copy2buffer_left_top_out_naive(__global Dtype* map_out,
    const int offset_map_out, 
    const __global Dtype* map_in, const int offset_map_in, 
    const int size,
    const int height_out, const int width_out, 
    const int fft_height, const int fft_width, 
    const int ker_center_h, const int ker_center_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in += offset_map_in;
  int gId = get_global_id(0);
  int h_out = gId / width_out;
  int w_out = gId - (h_out * width_out);
  int h = h_out * stride_h + ker_center_h;
  int w = w_out * stride_w + ker_center_w;
  int src_idx = h*fft_width + w;
  map_out[gId] = map_in[src_idx];
}

/* Use when width_out < 4 */
__kernel void copy2buffer_left_top_out_naive_2d(__global Dtype* map_out,
    const int offset_map_out, 
    const __global Dtype* map_in, const int offset_map_in,
    const int size, const int count, const int map_in_size,
    const int height_out, const int width_out, 
    const int fft_height, const int fft_width, 
    const int ker_center_h, const int ker_center_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in += offset_map_in;
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int h_out = gId / width_out;
  int w_out = gId - (h_out * width_out);
  int h = h_out * stride_h + ker_center_h;
  int w = w_out * stride_w + ker_center_w;
  int src_idx = out * map_in_size + h*fft_width + w;
  int dst_idx = out * size + gId;
  map_out[dst_idx] = map_in[src_idx];
}

/* Use when width_out >= 4 */
__kernel void copy2buffer_left_top_out(__global Dtype* map_out,
    const int offset_map_out,
    const __global Dtype* map_in, const int offset_map_in, 
    const int size,
    const int height_out, const int width_out, 
    const int fft_height, const int fft_width, 
    const int ker_c_h, const int ker_c_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int count = size >> 2;
  int gId4 = gId << 2;
  int h_out = gId4 / width_out;
  int w_out = gId4 - (h_out * width_out);
  int h = h_out * stride_h + ker_c_h;
  int w = w_out * stride_w + ker_c_w;
  int src_idx = h*fft_width + w;
  if (gId < count) {
    Dtype4 map_in_cache4;
    int has_pad = width_out - (w - pad_w); 
    if (has_pad >= 4) {
      map_in_cache4 = vload4(src_idx >> 2, map_in);
    } else {
      int right_elements = fft_width - width_out;
      if (0 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.x = map_in[src_idx];
      if (1 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.y = map_in[src_idx+1];
      if (2 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.z = map_in[src_idx+2];
      if (3 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.w = map_in[src_idx+3];
    }
    vstore4(map_in_cache4, gId, map_out);
  } else if (gId == count) {
    int res = size - (count << 2); /* size % 4 */
    if (res > 0) {
      for (int i = gId4; i < size; ++i) {
        map_out[i] = map_in[src_idx];
        src_idx++;
      }
    }
  }
}

/* Use when width_out >= 4 */
__kernel void copy2buffer_left_top_out_2d(__global Dtype* map_out,
    const int offset_map_out,
    const __global Dtype* map_in, const int offset_map_in, 
    const int size, const int count, const int map_in_size,
    const int height_out, const int width_out, 
    const int fft_height, const int fft_width, 
    const int ker_c_h, const int ker_c_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int gId4 = gId << 2;
  int h_out = gId4 / width_out;
  int w_out = gId4 - (h_out * width_out);
  int h = h_out * stride_h + ker_c_h;
  int w = w_out * stride_w + ker_c_w;
  int src_idx = h*fft_width + w;
  const __global Dtype* map_in_2d = map_in + out * map_in_size;
  __global Dtype* map_out_2d = map_out + out * size;
  if (gId < count) {
    Dtype4 map_in_cache4;
    int has_pad = width_out - (w - pad_w); 
    if (has_pad >= 4) {
      map_in_cache4 = vload4(src_idx >> 2, map_in_2d);
    } else {
      int right_elements = fft_width - width_out;
      if (0 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.x = map_in_2d[src_idx];
      if (1 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.y = map_in_2d[src_idx+1];
      if (2 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.z = map_in_2d[src_idx+2];
      if (3 == has_pad) {
        src_idx += right_elements;
      }
      map_in_cache4.w = map_in_2d[src_idx+3];
    }
    vstore4(map_in_cache4, gId, map_out_2d);
  } else if (gId == count) {
    int res = size - (count << 2); /* size % 4 */
    if (res > 0) {
      const __global Dtype4* map_in_2d_4 =
            (const __global Dtype4*)(map_in_2d + src_idx);
      __global Dtype4* map_out_2d_4 = (__global Dtype4*)(map_out_2d + gId4);
      if (res == 3) {
        map_out_2d_4[0].xyz = map_in_2d_4[0].xyz;
      } else if (res == 2) {
        map_out_2d_4[0].xy = map_in_2d_4[0].xy;
      } else if (res == 1) {
        map_out_2d_4[0].x = map_in_2d_4[0].x;
      }
    }
  }
}

__kernel void copy2buffer_cyclic_shift_out(__global Dtype* map_out,
    const int offset_map_out, 
    const __global Dtype* map_in, const int offset_map_in, 
    const int width_out, 
    const int fft_height, const int fft_width, 
    const int ker_center_h, const int ker_center_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int h_out = gId / width_out;
  int w_out = gId - (h_out * width_out);
  int h = h_out * stride_h + pad_h;
  int w = w_out * stride_w + pad_w;
  int ky = h - ker_center_h;
  if (ky < 0) ky += fft_height;
  int kx = w - ker_center_w;
  if (kx < 0) kx += fft_width;
  int src_idx = ky*fft_width + kx;
  map_out[gId] = map_in[src_idx];
}

__kernel void copy2buffer_cyclic_shift_out_2d(__global Dtype* map_out,
    const int offset_map_out, 
    const __global Dtype* map_in, const int offset_map_in,
    const int map_out_size, const int map_in_size, 
    const int width_out, 
    const int fft_height, const int fft_width, 
    const int ker_center_h, const int ker_center_w,
    const int stride_h, const int stride_w, 
    const int pad_h, const int pad_w) {
  map_out += offset_map_out;
  map_in  += offset_map_in;
  int gId = get_global_id(0);
  int gId_y = get_global_id(1);
  int h_out = gId / width_out;
  int w_out = gId - (h_out * width_out);
  int h = h_out * stride_h + pad_h;
  int w = w_out * stride_w + pad_w;
  int ky = h - ker_center_h;
  if (ky < 0) ky += fft_height;
  int kx = w - ker_center_w;
  if (kx < 0) kx += fft_width;
  int src_idx = gId_y * map_in_size + ky*fft_width + kx;
  int dst_idx = gId_y * map_out_size + gId;
  map_out[dst_idx] = map_in[src_idx];
}

__kernel void complex_conjugate_multiplication_1d(__global Dtype* dst,
    const int offset_dst, 
    const __global Dtype* src1, const int offset_src1,
    const __global Dtype* src2, const int offset_src2, 
    const int ch_gr) {
  dst += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int gId = get_global_id(0); 
  int size = get_global_size(0);
  Dtype4 dst_cache = 0.f;
  int src_idx;
  Dtype4 s1_cache;
  Dtype4 s2_cache;
  for (int c = 0; c < ch_gr; ++c) {
    src_idx = size * c + gId;
    s1_cache = vload4(src_idx, src1);
    s2_cache = vload4(src_idx, src2);
    dst_cache.x +=  s1_cache.x * s2_cache.x + s1_cache.y * s2_cache.y;
    dst_cache.y += -s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;
    dst_cache.z +=  s1_cache.z * s2_cache.z + s1_cache.w * s2_cache.w;
    dst_cache.w += -s1_cache.z * s2_cache.w + s1_cache.w * s2_cache.z;
  }
  ((__global Dtype4*)(&dst[gId<<2]))[0] += dst_cache; 
}

__kernel void complex_conjugate_multiplication_2d(__global Dtype* dst,
    const int offset_dst, 
    const __global Dtype* src1, const int offset_src1, 
    const __global Dtype* src2, const int offset_src2,
    const int out_gr, const int map_size, const int ch_gr) {
  dst += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int src1_idx, src2_idx;
  int dst_map_offset = map_size * out;
  int dst_idx = dst_map_offset + gId;
  Dtype4 s1_cache, s2_cache;
  Dtype4 dst_cache = 0.f;
  int map_offset = dst_map_offset * ch_gr;
  for (int i = 0; i < ch_gr; ++i) {
    src1_idx = map_size * i + gId;
    src2_idx = map_offset + src1_idx;
    s1_cache = vload4(src1_idx, src1);
    s2_cache = vload4(src2_idx, src2);
    dst_cache.xz += mad( s1_cache.xz, s2_cache.xz, s1_cache.yw * s2_cache.yw);
    dst_cache.yw += mad(-s1_cache.xz, s2_cache.yw, s1_cache.yw * s2_cache.xz);
  }
  vstore4(dst_cache, dst_idx, dst);
}

__kernel void complex_conjugate_multiplication_2d_SLM(
    __global Dtype* restrict dst, const int offset_dst,
    const __global Dtype* restrict src1, const int offset_src1, 
    __local Dtype* local_src1, 
    const __global Dtype* restrict src2, const int offset_src2, 
    const int out_gr, const int map_size, const int ch_gr) {
  int gId = get_global_id(0);
  if (gId >= map_size) return; /* Do not remove this */
  int out = get_global_id(1);
  if (out >= out_gr) return;   /* Do not remove this */
  dst += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int tId = get_local_id(0);
  int local_out = get_local_id(1);
  int tile_size = get_local_size(0);
  Dtype4 s1_cache;
  if (local_out == 0) {
    for (int c = 0; c < ch_gr; ++c) {
      s1_cache = vload4(map_size * c + gId, src1);
      vstore4(s1_cache, tile_size * c + tId, local_src1); 
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int dst_map_offset = map_size * out;
  int dst_idx = (dst_map_offset + gId) << 2;
  Dtype4 dst_cache = 0.f;
  Dtype4 s2_cache;
  int ch_offset = 0; 
  int map_offset = dst_map_offset * ch_gr; 
  for (int c = 0; c < ch_gr; ++c) {
    ch_offset = map_size * c;
    s1_cache = vload4(tile_size * c + tId, local_src1);
    s2_cache = vload4(map_offset + ch_offset + gId, src2);
    dst_cache.xz += mad( s1_cache.xz, s2_cache.xz, s1_cache.yw * s2_cache.yw);
    dst_cache.yw += mad(-s1_cache.xz, s2_cache.yw, s1_cache.yw * s2_cache.xz);
  }
  ((__global Dtype4*)(&dst[dst_idx]))[0] += dst_cache; 
}

__kernel void complex_conjugate_multiplication_3d(__global Dtype* dst,
    const int offset_dst, 
    const __global Dtype* src1, const int offset_src1,
    const __global Dtype* src2, const int offset_src2, 
    const int out_gr, const int size, const int ch_gr) {
  dst  += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int ch  = get_global_id(2);
  Dtype4 dst_cache = 0.f;
  Dtype4 s1_cache  = ((__global Dtype4*)(&(src1[(size*ch+gId)<<2])))[0];
  Dtype4 s2_cache  = ((__global Dtype4*)(&(src2[(size*(out*ch_gr+ch)+gId)<<2])))[0];
  dst_cache.x =  s1_cache.x * s2_cache.x + s1_cache.y * s2_cache.y;
  dst_cache.y = -s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;
  dst_cache.z =  s1_cache.z * s2_cache.z + s1_cache.w * s2_cache.w;
  dst_cache.w = -s1_cache.z * s2_cache.w + s1_cache.w * s2_cache.z;
  ((__global Dtype4*)(&dst[(size*out+gId)<<2]))[0] += dst_cache;
}

__kernel void complex_conjugate_multiplication_3d_SLM(__global Dtype* dst,
    const int offset_dst, __local Dtype* local_dst,  
    const __global Dtype* src1, const int offset_src1, 
    __local Dtype* local_src1, const __global Dtype* src2, 
    const int offset_src2, const int out_gr, const int map_size, 
    const int ch_gr) {
  int gId = get_global_id(0);
  if (gId >= map_size) return; /* Do not remove this */
  int out = get_global_id(1);
  if (out >= out_gr) return;   /* Do not remove this */
  int ch = get_global_id(2);
  if (ch >= ch_gr) return;     /* Do not remove this */
  dst += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int tId = get_local_id(0);
  int local_out = get_local_id(1);
  int tile_size = get_local_size(0);
  Dtype4 s1_cache;
  if (local_out == 0) {
    s1_cache = vload4(map_size * ch + gId, src1);
    vstore4(s1_cache, tile_size * ch + tId, local_src1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int dst_map_offset = map_size * out;
  int dst_idx = (dst_map_offset + gId) << 2;
  Dtype4 dst_cache = 0.f;
  Dtype4 s2_cache;
  s1_cache = vload4(tile_size * ch + tId, local_src1);
  s2_cache = vload4((dst_map_offset * ch_gr) + (map_size * ch) + gId, src2);
  dst_cache.x +=  s1_cache.x * s2_cache.x + s1_cache.y * s2_cache.y;
  dst_cache.y += -s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;
  dst_cache.z +=  s1_cache.z * s2_cache.z + s1_cache.w * s2_cache.w;
  dst_cache.w += -s1_cache.z * s2_cache.w + s1_cache.w * s2_cache.z;
  ((__global Dtype4*)(&dst[dst_idx]))[0] += dst_cache;
}

__kernel void complex_multiplication_1d(__global Dtype* dst,
    const int offset_dst, 
    const __global Dtype* src1, const int offset_src1, 
    const __global Dtype* src2, const int offset_src2,
    const int size, const int ch_gr) {
  dst += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int gId = get_global_id(0);
  Dtype4 s2_cache;
  Dtype4 dst_cache = 0.f;
  int idx_with_ch;
  Dtype4 s1_cache = vload4(gId, src1);
  for (int ch = 0; ch < ch_gr; ++ch) {
    idx_with_ch = size * ch + gId;
    s2_cache = vload4(idx_with_ch, src2);
    dst_cache.xz = s1_cache.xz * s2_cache.xz - s1_cache.yw * s2_cache.yw;
    dst_cache.yw = s1_cache.xz * s2_cache.yw + s1_cache.yw * s2_cache.xz;
    ((__global Dtype4*)(&dst[idx_with_ch<<2]))[0] += dst_cache;
  }
}

__kernel void complex_multiplication_2d_SLM(__global Dtype* restrict dst,
    const int offset_dst, __local Dtype* local_dst,
    const __global Dtype* restrict src1, const int offset_src1, 
    const __global Dtype* restrict src2, const int offset_src2,
    const int num_output, const int size, const int ch_gr) {
  int gId = get_global_id(0);
  if (gId >= size) return;
  int out = get_global_id(1);
  if (out >= num_output) return;
  dst += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int tId = get_local_id(0);
  int tOut = get_local_id(1);
  int tile_size = get_local_size(0);
  int local_out_size = get_local_size(1);
  int out_offset = out * size;
  int out_ch_offset = out_offset * ch_gr;
  int tile_size_in_all_ch = tile_size * ch_gr;
  int local_out_ch_offset = tOut * tile_size_in_all_ch;
  int src2_idx, local_dst_idx;
  Dtype4 s2_cache, dst_cache;
  int src1_idx = out_offset + gId;
  Dtype4 s1_cache = vload4(src1_idx, src1);
  for (int ch = 0; ch < ch_gr; ++ch) {
    src2_idx = out_ch_offset + ch * size + gId;
    s2_cache = vload4(src2_idx, src2);
    dst_cache.xz = s1_cache.xz * s2_cache.xz - s1_cache.yw * s2_cache.yw;
    dst_cache.yw = s1_cache.xz * s2_cache.yw + s1_cache.yw * s2_cache.xz;
    local_dst_idx = local_out_ch_offset + ch * tile_size + tId;
    vstore4(dst_cache, local_dst_idx, local_dst);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int start_idx, half_start_idx;
  int ch_offset;
  int this_idx, that_idx;
  for (int offset = local_out_size >>= 1; offset > 0; offset >>=1) {
    if (tOut < offset) {
      start_idx = tOut * tile_size_in_all_ch + tId;
      half_start_idx = (tOut + offset) * tile_size_in_all_ch + tId;
      for (int ch = 0; ch < ch_gr; ++ch) {
        ch_offset = ch * tile_size;
        this_idx = (start_idx + ch_offset) << 2;
        that_idx = (half_start_idx + ch_offset) << 2;
        ((__local Dtype4*)(&local_dst[this_idx]))[0] += 
            ((__local Dtype4*)(&local_dst[that_idx]))[0];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tOut == 0) {
    for (int ch = 0; ch < ch_gr; ++ch) {
      dst_cache = vload4(tile_size * ch + tId, local_dst);
      ((__global Dtype4*)(&dst[(size * ch + gId)<<2]))[0] += dst_cache;
    }
  }
}

__kernel void complex_multiplication_3d(__global Dtype* dst,
    const int offset_dst, 
    const __global Dtype* src1, const int offset_src1, 
    const __global Dtype* src2, const int offset_src2,
    const int size, const int ch_gr, const int out_gr, const int num_output) {
  dst  += offset_dst;
  src1 += offset_src1;
  src2 += offset_src2;
  int gId = get_global_id(0);
  int ch  = get_global_id(1);
  int out = get_global_id(2);
  int g = out / out_gr;
  ch += (g * ch_gr);
  int c_offset = ch - ((ch / ch_gr) * ch_gr); 
  __global Dtype2* dst_ch = ((__global Dtype2*)(dst)) + (size * ch);
  __global Dtype2* src1_out = ((__global Dtype2*)(src1)) + (size * out);
  __global Dtype2* src2_out_ch = ((__global Dtype2*)(src2)) + (size * (out * ch_gr + c_offset));
  Dtype2 s1_cache  = src1_out[gId];
  Dtype2 s2_cache  = src2_out_ch[gId];
  Dtype2 dst_cache = 0.f;
  dst_cache.x = s1_cache.x * s2_cache.x - s1_cache.y * s2_cache.y;
  dst_cache.y = s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;
  dst_ch[gId] += dst_cache;
}

/* Convert [RRRR...GGGG...BBBB...] to [RGBRGBRGBRGB...] */
/* Reshape 2 */
__kernel void convert_data_to_channel_major(__global Dtype2* dst, 
    const __global Dtype2* src, const int size, const int ch_gr) {
  int gId = get_global_id(0);
  __global Dtype* dst_ptr = (__global Dtype*)(dst + (gId * ch_gr));
  const __global Dtype* src_ptr = (const __global Dtype*)(src + gId);
  Dtype2 s;
  int src_idx = 0;
  for (int i = 0; i < ch_gr; ++i) {
    s = vload2(src_idx, src_ptr);
    vstore2(s, i, dst_ptr);
    src_idx += size;
  }
}
/* Reshape 1 */
/*__kernel void convert_data_to_channel_major(__global Dtype4* dst,
    const __global Dtype4* src, const int size, const int ch_gr) {
  int gId = get_global_id(0);
  const __global Dtype4* src_ptr4 = src + gId; 
  __global Dtype4* dst_ptr4 = dst + (gId * ch_gr);
  for (int i = 0; i < ch_gr; ++i) {
      dst_ptr4[i] = src_ptr4[i*size];
  }
}
*/

/* Convert multiple [RRRR...GGGG...BBBB...] to multiple [RGBRGBRGBRGB...] */
/* Reshape 2 */
__kernel void convert_weight_to_channel_major(__global Dtype2* dst, 
    const __global Dtype2* src, const int size, const int ch_gr,
    const int num_output) {
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int out_offset = out * (size * ch_gr);
  __global Dtype* dst_ptr = (__global Dtype*)(dst + out_offset + (gId * ch_gr));
  const __global Dtype* src_ptr = 
      (const __global Dtype*)(src + out_offset + gId);
  Dtype2 s;
  int src_idx = 0;
  for (int i = 0; i < ch_gr; ++i) {
    s = vload2(src_idx, src_ptr);
    vstore2(s, i, dst_ptr);
    src_idx += size;
  }
}
/* Reshape 1 */
/*
__kernel void convert_weight_to_channel_major(__global Dtype4* dst,
    const __global Dtype4* src, const int size, const int ch_gr,
    const int out_gr) {
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int out_offset = out * (size * ch_gr);
  __global Dtype4* dst_ptr4 = dst + out_offset + (gId * ch_gr);
  const __global Dtype4* src_ptr4 = src + out_offset + gId;
  for (int i = 0; i < ch_gr; ++i) {
    dst_ptr4[i] = src_ptr4[size * i];
  }
}
*/

/* Cdotc per element */
/* Reshape 1 */
/*
__kernel void batchedCdotc(__global Dtype4* dst, 
    const __global Dtype4* src1, const __global Dtype4* src2,  
    const int size, const int ch_gr, const int out_gr) {  
  int gId = get_global_id(0); 
  int out = get_global_id(1); 
  int ch_offset = gId * ch_gr; 
  int out_offset = out * size; 
  const __global Dtype* src1_ptr = (const __global Dtype*)(src1 + ch_offset);  
  const __global Dtype* src2_ptr = (const __global Dtype*)(src2 + (out_offset * ch_gr) + ch_offset); 
  Dtype4 cdotc = 0.f; 
  Dtype4 s1, s2; 
  for (int c = 0; c < ch_gr; ++c) { 
    s1 = vload4(c, src1_ptr); 
    s2 = vload4(c, src2_ptr); 
    cdotc.xz += mad( s1.xz, s2.xz, s1.yw * s2.yw); 
    cdotc.yw += mad(-s1.xz, s2.yw, s1.yw * s2.xz); 
  } 
  __global Dtype4* dst_ptr4 = dst + out_offset + gId; 
  dst_ptr4[0] += cdotc; 
}
*/

/* Cdotc per two elements */
/* Reshape 2 */
__kernel void batchedCdotc(__global Dtype2* dst,
    const __global Dtype2* src1, const __global Dtype2* src2, 
    const int size, const int ch_gr, const int out_gr) {
  int gId = get_global_id(0);
  int out = get_global_id(1);
  int ch_offset = gId * ch_gr;
  const __global Dtype* src1_ptr = (const __global Dtype*)(src1 + ch_offset); 
  const __global Dtype* src2_ptr = 
      (const __global Dtype*)(src2 + (out * size * ch_gr) + ch_offset);
  Dtype4 cdotc4 = 0.f;
  Dtype2 cdotc = 0.f;
  Dtype4 s1, s2;
  int n = ch_gr >> 1;
  int r = ch_gr - (n << 1);
  for (int i = 0; i < n; ++i) {
    s1 = vload4(i, src1_ptr);
    s2 = vload4(i, src2_ptr);
    cdotc4.xz += mad( s1.xz, s2.xz, s1.yw * s2.yw);
    cdotc4.yw += mad(-s1.xz, s2.yw, s1.yw * s2.xz);
  }
  cdotc.x += dot(cdotc4.xz, (float2)(1));
  cdotc.y += dot(cdotc4.yw, (float2)(1));
  if (r == 1) {
    const __global Dtype* src1_ptr2 = 
        (const __global Dtype*)(((const __global Dtype4*)(src1_ptr)) + n);
    const __global Dtype* src2_ptr2 = 
        (const __global Dtype*)(((const __global Dtype4*)(src2_ptr)) + n);
    Dtype2 t1 = vload2(0, src1_ptr2); 
    Dtype2 t2 = vload2(0, src2_ptr2);
    cdotc.x += mad( t1.x, t2.x, t1.y * t2.y);
    cdotc.y += mad(-t1.x, t2.y, t1.y * t2.x);
  }
  __global Dtype* dst_ptr = (__global Dtype*)(dst + (out * size) + gId);
  vstore2(cdotc, 0, dst_ptr);
}
