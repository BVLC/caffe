#include <vector>

#include "caffe/layers/pad_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PadForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index; // Preserve the original value
    int height_out = height_in + pad + pad;
    int width_out = width_in + pad + pad;
    int w = i % width_in;
    i /= width_in;
    int h = i % height_in;
    i /= height_in;
    int c = i % channel;
    i /= channel;

    out[((i * channel + c) * height_out + h + pad) * width_out + pad + w] =
        in[index];
  }
}

template <typename Dtype>
__global__ void PadForwardPadZero(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % width_out;
    int h = (index / width_out) % height_out;
    if (h < pad || h > height_out-1-pad || w < pad || w > width_out-1-pad) {
      out[index] = Dtype(0);
    }
  }
}
// No matching PadBackwardPadZero, since no gradient propagates through zero padding

template <typename Dtype>
__global__ void PadForwardPadLeftAndRightReplicate(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {

  CUDA_KERNEL_LOOP(index, count) {
    int i = index;
    int w = i % width_out;
    i /= width_out;
    int h = i % height_out;

    // Don't do top or bottom padding
    if (h < pad || h > height_out-1-pad) {
      return;
    }

    int off = 0;
    if (w < pad) {
      off = pad - w;
    } else {
      off = width_out - 1 - pad - w;
    }

    if (w < pad || w > width_out-1-pad) {
      out[index] = out[index + off];
    }
  }
}

template <typename Dtype>
__global__ void PadForwardPadTopAndBottomReplicate(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index / width_out;
    int h = i % height_out;

    int off = 0;
    if (h < pad) {
      off = pad - h;
    } else {
      off = height_out - 1 - pad - h;
    }
    off *= width_out;

    if (h < pad || h > height_out-1-pad) {
      out[index] = out[index + off];
    }
  }
}

template <typename Dtype>
__global__ void PadForwardPadLeftAndRightReflect(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {

  CUDA_KERNEL_LOOP(index, count) {
    int i = index;
    int w = i % width_out;
    i /= width_out;
    int h = i % height_out;

    // Don't do top or bottom padding
    if (h < pad || h > height_out-1-pad) {
      return;
    }

    int off = 0;
    if (w < pad) {
      off = 2*(pad - w) - 1;
    } else {
      off = 2*(width_out - pad - w) - 1;
    }

    if (w < pad || w > width_out-1-pad) {
      out[index] = out[index + off];
    }
  }
}

template <typename Dtype>
__global__ void PadForwardPadTopAndBottomReflect(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index / width_out;
    int h = i % height_out;

    int off = 0;
    if (h < pad) {
      off = 2*(pad - h) - 1;
    } else {
      off = 2*(height_out - pad - h) - 1;
    }
    off *= width_out;

    if (h < pad || h > height_out-1-pad) {
      out[index] = out[index + off];
    }
  }
}


template <typename Dtype>
__global__ void PadForwardPadLeftAndRightReflect101(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {

  CUDA_KERNEL_LOOP(index, count) {
    int i = index;
    int w = i % width_out;
    i /= width_out;
    int h = i % height_out;

    // Don't do top or bottom padding
    if (h < pad || h > height_out-1-pad) {
      return;
    }

    int off = 0;
    if (w < pad) {
      off = 2*(pad - w);
    } else {
      off = 2*(width_out - pad - w) - 2;
    }

    if (w < pad || w > width_out-1-pad) {
      out[index] = out[index + off];
    }
  }
}

template <typename Dtype>
__global__ void PadForwardPadTopAndBottomReflect101(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index / width_out;
    int h = i % height_out;

    int off = 0;
    if (h < pad) {
      off = 2*(pad - h);
    } else {
      off = 2*(height_out - pad - h) - 2;
    }
    off *= width_out;

    if (h < pad || h > height_out-1-pad) {
      out[index] = out[index + off];
    }
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // First, set all data to be zero for the boundary pixels
  // CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * top[0]->count()));
  // Copy the main body (not yet setting the padding)
  // NOLINT_NEXT_LINE(whitespace/operators)
  PadForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
      PAD_);
  CUDA_POST_KERNEL_CHECK;

  // Padding
  switch (PAD_TYPE_) {
  case PadParameter::ZERO:
    PadForwardPadZero<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    break;
  case PadParameter::REPLICATE:
    // Left and right first, only in the "body", i.e., not in the
    // vertical padding, then vertical across entire rows.
    PadForwardPadLeftAndRightReplicate<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    PadForwardPadTopAndBottomReplicate<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    break;
  case PadParameter::REFLECT:
    // Left and right first, only in the "body", i.e., not in the
    // vertical padding, then vertical across entire rows.
    PadForwardPadLeftAndRightReflect<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    PadForwardPadTopAndBottomReflect<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    break;
  case PadParameter::REFLECT_101:
    // Left and right first, only in the "body", i.e., not in the
    // vertical padding, then vertical across entire rows.
    PadForwardPadLeftAndRightReflect101<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    PadForwardPadTopAndBottomReflect101<Dtype>
      <<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
        top[0]->count(), top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
    break;
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PadBackwardPadLeftAndRightReplicate(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index;
    int w = i % width_out;
    i /= width_out;
    int h = i % height_out;

    // Don't do top or bottom padding, and padding only comes from the
    // left and right edges
    if (h < pad || h > height_out-1-pad || (w != pad && w != width_out - 1 - pad)) {
      return;
    }

    int off;
    if (w == pad) {
      off = -pad;
    } else { // w == width_out - 1 - pad
      off = 1; // To the first pad pixel
    }

    for (int dw=0; dw < pad; ++dw) {
      out[index] += out[index + off + dw];
    }
  }
}

template <typename Dtype>
__global__ void PadBackwardPadTopAndBottomReplicate(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index / width_out;
    int h = i % height_out;

    // Padding only comes from the top and bottom edges
    if (h != pad && h != height_out - 1 - pad) {
      return;
    }

    int off;
    if (h == pad) {
      off = -pad*width_out;
    } else { // h == height_out - 1 - pad
      off = width_out; // One row, i.e., to the first pad row
    }

    for (int dh=0; dh < pad; ++dh) {
      out[index] += out[index + off + dh*width_out];
    }
  }
}

template <typename Dtype>
__global__ void PadBackwardPadLeftAndRightReflect(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index;
    int w = i % width_out;
    i /= width_out;
    int h = i % height_out;

    // Don't do top or bottom padding
    if (h < pad || h > height_out-1-pad) {
      return;
    }

    // Padding comes from a border of width pad within the main image body
    if ((w < pad || w > 2*pad-1) &&
	(w > width_out - 1 - pad || w < width_out - 2*pad)) {
      return;
    }

    int off;
    if (w < 2*pad) {
      off = 2*(pad - w) - 1;
    } else {
      off = 2*(width_out - pad - w) - 1;
    }

    out[index] += out[index + off];
  }
}

template <typename Dtype>
__global__ void PadBackwardPadTopAndBottomReflect(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index / width_out;
    int h = i % height_out;

    // Padding comes from a border of width pad within the main image body
    if ((h < pad || h > 2*pad-1) &&
	(h > height_out - 1 - pad || h < height_out - 2*pad)) {
      return;
    }

    int off;
    if (h < 2*pad) {
      off = 2*(pad - h) - 1;
    } else {
      off = 2*(height_out - pad - h) - 1;
    }
    off *= width_out;

    out[index] += out[index + off];
  }
}

template <typename Dtype>
__global__ void PadBackwardPadLeftAndRightReflect101(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index;
    int w = i % width_out;
    i /= width_out;
    int h = i % height_out;

    // Don't do top or bottom padding
    if (h < pad || h > height_out-1-pad) {
      return;
    }

    // Padding comes from a border of width pad within the main image body
    if ((w < pad+1 || w > 2*pad) &&
	(w > width_out - 2 - pad || w < width_out - 1 - 2*pad)) {
      return;
    }

    int off;
    if (w < 2*pad+1) {
      off = 2*(pad - w);
    } else {
      off = 2*(width_out - pad - w) - 2;
    }

    out[index] += out[index + off];
  }
}

template <typename Dtype>
__global__ void PadBackwardPadTopAndBottomReflect101(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index / width_out;
    int h = i % height_out;

    // Padding comes from a border of width pad+1 within the main
    // image body, but not right on the edge of the image.
    if ((h < pad+1 || h > 2*pad) &&
	(h > height_out - 2 - pad || h < height_out - 1 - 2*pad)) {
      return;
    }

    int off;
    if (h < 2*pad+1) {
      off = 2*(pad - h);
    } else {
      off = 2*(height_out - pad - h) - 2;
    }
    off *= width_out;

    out[index] += out[index + off];
  }
}

template <typename Dtype>
__global__ void PadBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index; // Preserve original value
    int height_out = height_in + pad + pad;
    int width_out = width_in + pad + pad;
    int w = i % width_in;
    i /= width_in;
    int h = i % height_in;
    i /= height_in;
    int c = i % channel;
    i /= channel;
    out[index] = in[((i * channel + c) * height_out + h + pad) *
		    width_out + pad + w];
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bcount = bottom[0]->count();
    const int tcount = top[0]->count();
    caffe_gpu_set(bcount, static_cast<Dtype>(0), bottom_diff);
    // In reverse order from Forward_gpu, so ...
    // Padding first. Operate within top to set the gradient in the
    // part to be copied to bottom.
    switch (PAD_TYPE_) {
    case PadParameter::ZERO:
      break; // No gradient in the padding; it's constant
    case PadParameter::REPLICATE:
      PadBackwardPadTopAndBottomReplicate<Dtype>
	<<<CAFFE_GET_BLOCKS(tcount), CAFFE_CUDA_NUM_THREADS>>>(
            tcount, top_diff, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
      PadBackwardPadLeftAndRightReplicate<Dtype>
	<<<CAFFE_GET_BLOCKS(tcount), CAFFE_CUDA_NUM_THREADS>>>(
            tcount, top_diff, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
      break;
    case PadParameter::REFLECT:
      PadBackwardPadTopAndBottomReflect<Dtype>
	<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            tcount, top_diff, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
      PadBackwardPadLeftAndRightReflect<Dtype>
	<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            tcount, top_diff, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
      break;
    case PadParameter::REFLECT_101:
      PadBackwardPadTopAndBottomReflect101<Dtype>
	<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            tcount, top_diff, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
      PadBackwardPadLeftAndRightReflect101<Dtype>
	<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            tcount, top_diff, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_, PAD_);
      break;
    }
    // Copy into place
    // NOLINT_NEXT_LINE(whitespace/operators)
    PadBackward<Dtype><<<CAFFE_GET_BLOCKS(bcount), CAFFE_CUDA_NUM_THREADS>>>(
        bcount, top_diff, bottom_diff, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
        PAD_);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PadLayer);

}  // namespace caffe
