
__kernel void caffe_Fill_Buffer(int size, Dtype pattern,__global Dtype* buffer)
        {
                const uint i = get_global_id(0);
                buffer[i] = pattern; 
        }

__kernel void R123_rng_flt32(__global ulong* seed, int N, 
    __global float* result) {
  OCL_KERNEL_LOOP(index, N) {
    threefry2x32_key_t key = {{seed[1], seed[2]}};
    threefry2x32_ctr_t ctr = {{0, seed[0]+index*N}};
    threefry2x32_ctr_t rand = threefry2x32(ctr, key);
    result[index] = (float)rand.v[0] / (float)UINT_MAX;
  }
}

__kernel void R123_rng_uint32(__global ulong* seed, int N, 
    __global unsigned int* result) {
  OCL_KERNEL_LOOP(index, N) {
    threefry2x32_key_t key = {{seed[1], seed[2]}};
    threefry2x32_ctr_t ctr = {{0, seed[0]+index*N}};
    threefry2x32_ctr_t rand = threefry2x32(ctr, key);
    result[index] = rand.v[0];
  }
}

#define TWO_PI 6.2831853071795
__kernel void rng_normal_flt32(__global ulong* seed, int N, float mu,
    float sigma, __global float* result) {
  OCL_KERNEL_LOOP(index, N) {
    threefry2x32_key_t key = {{seed[1], seed[2]}};
    threefry2x32_ctr_t ctr = {{0, seed[0]+index*N}};
    threefry2x32_ctr_t rand = threefry2x32(ctr, key);
    float x1 = (float)rand.v[0] / (float)UINT_MAX;
    float x2 = (float)rand.v[1] / (float)UINT_MAX;
    
    if(x1 < 1e-100)
      x1 = 1e-100;
    x1 = -2.0f * log(x1);
    x2 = x2 * TWO_PI;
    result[index] = (sigma * sqrt(x1) * cos(x2)) + mu;
  }
}

__kernel void Tadd_scalar(const int N, const Dtype alpha, __global Dtype* Y,
    const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    Y[y_off + index] += alpha;
  }
}

__kernel void Tadd(const int N, __global Dtype* a, const int a_off,
    __global Dtype* b, const int b_off, __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = a[a_off + index] + b[b_off + index];
  }
}

__kernel void Tsub(const int N, __global Dtype* a, const int a_off,
    __global Dtype* b, const int b_off, __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = a[a_off + index] - b[b_off + index];
  }
}

__kernel void Tmul(const int N, __global Dtype* a, const int a_off,
    __global Dtype* b, const int b_off, __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = a[a_off + index] * b[b_off + index];
  }
}

__kernel void Tdiv(const int N, __global Dtype* a, const int a_off,
    __global Dtype* b, const int b_off, __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = a[a_off + index] / b[b_off + index];
  }
}

__kernel void Tabs(const int N, __global Dtype* a, const int a_off,
    __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = fabs(a[a_off + index]);
  }
}

__kernel void Texp(const int N, __global Dtype* a, const int a_off,
    __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = exp(a[a_off + index]);
  }
}

__kernel void Tlog(const int N, __global Dtype* a, const int a_off,
    __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = log(a[a_off + index]);
  }
}

__kernel void Tpowx(const int N, __global Dtype* a, const int a_off,
    const Dtype alpha, __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, N) {
    y[y_off + index] = pow(a[a_off + index], alpha);
  }
}

__kernel void Tsign(const int n, __global Dtype* x, const int x_off,
    __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, n) { 
    y[y_off + index] = (0.f < x[x_off + index]) - (x[x_off + index] < 0.f);
  } 
}

__kernel void Tsignbit(const int n, __global Dtype* x, const int x_off,
    __global Dtype* y, const int y_off) {
  OCL_KERNEL_LOOP(index, n) {
    y[y_off + index] = signbit(x[x_off + index]);
  } 
}

__kernel void set_kernel(const int n, const Dtype alpha, __global Dtype* y) {
  OCL_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

