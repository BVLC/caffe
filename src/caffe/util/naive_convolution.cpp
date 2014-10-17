

namespace caffe {

// Helper function for convolutional data access
// This function takes a function object `accumulator` which
// is called inside convolution loop.
template <typename ACCUMULATOR>
static void naive_conv_loop(
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ACCUMULATOR accumulator) {

  int top_h = (bottom_h + 2 * pad_h - weight_h) / stride_h + 1;
  int top_w = (bottom_w + 2 * pad_w - weight_w) / stride_w + 1;

  // Image
  for (int ti = 0; ti < num_output * top_h * top_w; ti++) {
    // position on output
    int t_c = ti / (top_h * top_w);
    int t_y = (ti / top_w) % top_h;
    int t_x = ti % top_w;

    // Filter begin
    for (int ki = 0; ki < channels * weight_h * weight_w; ki++) {
      // position on weight
      int k_c = ki / (weight_h * weight_w);
      int k_y = (ki / weight_w) % weight_h;
      int k_x = ki % weight_w;
      // position on input
      int b_y = t_y * stride_h + k_y - pad_h;
      int b_x = t_x * stride_w + k_x - pad_w;
      if ((b_y < 0) || (b_y >= bottom_h) || (b_x < 0) || (b_x >= bottom_w)) {
        // skip if padded area
        continue;
      }
      // Resolve linear memory accessing from spatial coordinate
      int t = (t_c * top_h+ t_y) * top_w + t_x;
      int k = ((t_c * channels + k_c) * weight_h + k_y) * weight_w + k_x;
      int b = (k_c * bottom_h + b_y) * bottom_w + b_x;
      accumulator(t, k, b);
    }
    // Filter end
  }
  // Image end
}


// Accumulator function object for `naive_conv`
template <typename Dtype>
class AccumNaiveConv {
 public:
  AccumNaiveConv(const Dtype* bottom, const Dtype* weight, Dtype* top) :
      bottom_(bottom), weight_(weight), top_(top) {}
  void operator()(int t, int k, int b) {
    top_[t] += weight_[k] * bottom_[b];
  }
 protected:
  const Dtype* bottom_;
  const Dtype* weight_;
  Dtype* top_;
};

// Accumulator function object for `naive_conv_grad_weight`
template <typename Dtype>
class AccumNaiveConvGradWeight {
 public:
  AccumNaiveConvGradWeight(
      const Dtype* bottom, Dtype* weight, const Dtype* top) :
      bottom_(bottom), weight_(weight), top_(top) {}
  void operator()(int t, int k, int b) {
    weight_[k] += top_[t] * bottom_[b];
  }
 protected:
  const Dtype* bottom_;
  Dtype* weight_;
  const Dtype* top_;
};

// Accumulator function object for `naive_conv_grad_bottom`
template <typename Dtype>
class AccumNaiveConvGradBottom {
 public:
  AccumNaiveConvGradBottom(
      Dtype* bottom, const Dtype* weight, const Dtype* top) :
      bottom_(bottom), weight_(weight), top_(top) {}
  void operator()(int t, int k, int b) {
    bottom_[b] += weight_[k] * top_[t];
  }
 protected:
  Dtype* bottom_;
  const Dtype* weight_;
  const Dtype* top_;
};


// Convolution
template <typename Dtype>
void naive_conv(
    const Dtype* bottom, const Dtype* weight, Dtype* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  naive_conv_loop<AccumNaiveConv<Dtype> >(channels, bottom_h, bottom_w,
            num_output, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w,
            AccumNaiveConv<Dtype>(bottom, weight, top));
}

// Backprop to weight
template <typename Dtype>
void naive_conv_grad_weight(
    const Dtype* bottom, Dtype* weight, const Dtype* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  naive_conv_loop<AccumNaiveConvGradWeight<Dtype> >(
      channels, bottom_h, bottom_w,
      num_output, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w,
      AccumNaiveConvGradWeight<Dtype>(bottom, weight, top));
}

// Backprop to bottom
template <typename Dtype>
void naive_conv_grad_bottom(
    Dtype* bottom, const Dtype* weight, const Dtype* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  naive_conv_loop<AccumNaiveConvGradBottom<Dtype> >(
      channels, bottom_h, bottom_w,
      num_output, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w,
      AccumNaiveConvGradBottom<Dtype>(bottom, weight, top));
}

// Template instanciation
template void naive_conv(
    const float* bottom, const float* weight, float* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void naive_conv(
    const double* bottom, const double* weight, double* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void naive_conv_grad_weight(
    const float* bottom, float* weight, const float* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void naive_conv_grad_weight(
    const double* bottom, double* weight, const double* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void naive_conv_grad_bottom(
    float* bottom, const float* weight, const float* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void naive_conv_grad_bottom(
    double* bottom, const double* weight, const double* top,
    int channels, int bottom_h, int bottom_w,
    int num_output, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w);

// end of namespace
}
