__kernel void MaxForward(const int nthreads, __global const Dtype* bottom_data_a,
    __global const Dtype* bottom_data_b, const int blob_idx, __global Dtype* top_data,
    __global int* mask) {
  OCL_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}
__kernel void MaxBackward(const int nthreads, __global const Dtype* top_diff,
    const int blob_idx, __global const int* mask, __global Dtype* bottom_diff) {
  OCL_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}