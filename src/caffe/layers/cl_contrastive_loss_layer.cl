__kernel void CLLBackward(const int count, const int channels,
    const Dtype margin, const Dtype alpha, __global Dtype* y,
    __global Dtype* diff, __global Dtype* dist_sq,
    __global Dtype* bottom_diff) {
  OCL_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (trunc(y[n]) != 0.f) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist = 0.f;
      Dtype beta = 0.f;
	  Dtype dist = sqrt(dist_sq[n]);
	  mdist = (margin - dist);
	  beta = -alpha * mdist / (dist + (Dtype)(1e-4)) * diff[i];
      if (mdist > 0.f) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

__kernel void CLLBackwardLegacy(const int count, const int channels,
    const Dtype margin, const Dtype alpha, __global Dtype* y,
    __global Dtype* diff, __global Dtype* dist_sq,
    __global Dtype* bottom_diff) {
  OCL_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (trunc(y[n]) != 0.f) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist = 0.f;
      Dtype beta = 0.f;
      mdist = (margin - dist_sq[n]);
      beta = -alpha;
      if (mdist > 0.f) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}
