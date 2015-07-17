__kernel void SoftmaxLossForward(const int size, 
    const __global Dtype* prob_data, const __global Dtype* label, 
    __global Dtype* loss, const int num, const int dim, const int spatial_dim,
    const int has_ignore_label, const int ignore_label,
    __global Dtype* counts) {
  OCL_KERNEL_LOOP(index, size) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int)(label[n * spatial_dim + s]);
    if (has_ignore_label && label_value == ignore_label) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s], 
          (Dtype)(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

__kernel void SoftmaxLossBackward(const int size, const __global Dtype* top,
    const __global Dtype* label, __global Dtype* bottom_diff, 
    const int num, const int dim, const int spatial_dim, 
    const int has_ignore_label, const int ignore_label, 
    __global Dtype* counts) {
  const int channels = dim / spatial_dim;
  OCL_KERNEL_LOOP(index, size) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int)(label[n * spatial_dim + s]);

    if (has_ignore_label && label_value == ignore_label) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
