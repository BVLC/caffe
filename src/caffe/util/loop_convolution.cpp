

namespace caffe {

/*
 * Convolution loop accesser function
 */
template <typename ACCUMULATOR>
static void conv_loop(
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ACCUMULATOR accumulator) {

  int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

  // Image
  for (int oi = 0; oi < num_output * out_h * out_w; oi++) {
    // position on output
    int c = oi / (out_h * out_w);
    int y = (oi / out_w) % out_h;
    int x = oi % out_w;

    // Filter begin
    for (int ii = 0; ii < channels * kernel_h * kernel_w; ii++) {
      // position on kernel
      int cc = ii / (kernel_h * kernel_w);
      int yy = (ii / kernel_w) % kernel_h;
      int xx = ii % kernel_w;
      // position on input
      int yyy = y * stride_h + yy - pad_h;
      int xxx = x * stride_w + xx - pad_w;
      if ((yyy < 0) || (yyy >= in_h) || (xxx < 0) || (xxx >= in_w)) {
        // skip if padded area
        continue;
      }
      // accumulate
      int ooo = (c * out_h+ y) * out_w + x;
      int kkk = ((c * channels + cc) * kernel_h + yy) * kernel_w + xx;
      int iii = (cc * in_h + yyy) * in_w + xxx;
      accumulator(ooo, kkk, iii);
    }
    // Filter end
  }
  // Image end
}


// Accumulator

// function object for conv_filter
template <typename Dtype>
class ConvFilter {
 public:
  ConvFilter(const Dtype* in, const Dtype* kernel, Dtype* out) :
      in_(in), kernel_(kernel), out_(out) {}
  void operator()(int o, int k, int i) {
    out_[o] += kernel_[k] * in_[i];
  }
 protected:
  const Dtype* in_;
  const Dtype* kernel_;
  Dtype* out_;
};

// function object for conv_filter
template <typename Dtype>
class ConvWeight {
 public:
  ConvWeight(const Dtype* in, Dtype* kernel, const Dtype* out) :
      in_(in), kernel_(kernel), out_(out) {}
  void operator()(int o, int k, int i) {
    kernel_[k] += out_[o] * in_[i];
  }
 protected:
  const Dtype* in_;
  Dtype* kernel_;
  const Dtype* out_;
};

// function object for conv_weight
template <typename Dtype>
class ConvImage {
 public:
  ConvImage(Dtype* in, const Dtype* kernel, const Dtype* out) :
      in_(in), kernel_(kernel), out_(out) {}
  void operator()(int o, int k, int i) {
    in_[i] += kernel_[k] * out_[o];
  }
 protected:
  Dtype* in_;
  const Dtype* kernel_;
  const Dtype* out_;
};


//
template <typename Dtype>
void conv_filter(
    const Dtype* in, const Dtype* kernel, Dtype* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  conv_loop<ConvFilter<Dtype> >(channels, in_h, in_w,
            num_output, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            ConvFilter<Dtype>(in, kernel, out));
}

// 
template <typename Dtype>
void conv_weight(
    const Dtype* in, Dtype* kernel, const Dtype* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  conv_loop<ConvWeight<Dtype> >(channels, in_h, in_w,
            num_output, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            ConvWeight<Dtype>(in, kernel, out));
}

//
template <typename Dtype>
void conv_image(
    Dtype* in, const Dtype* kernel, const Dtype* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  conv_loop<ConvImage<Dtype> >(channels, in_h, in_w,
            num_output, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            ConvImage<Dtype>(in, kernel, out));
}

template void conv_filter(
    const float* in, const float* kernel, float* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void conv_filter(
    const double* in, const double* kernel, double* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void conv_weight(
    const float* in, float* kernel, const float* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void conv_weight(
    const double* in, double* kernel, const double* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void conv_image(
    float* in, const float* kernel, const float* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);
template void conv_image(
    double* in, const double* kernel, const double* out,
    int channels, int in_h, int in_w,
    int num_output, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w);

// end of namespace
}
