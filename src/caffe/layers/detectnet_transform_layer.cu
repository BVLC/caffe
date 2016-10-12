#ifdef USE_OPENCV

#include <math.h>
#include <math_constants.h>
#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/layers/detectnet_transform_layer.hpp"
#include "caffe/util/detectnet_coverage.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Calculate the location in the image from the loop index
__device__ void get_pixel_indices(const int loop_index,
    const uint4 shape, int* x, int* y, int* n
) {
  int idx = loop_index;
  *n = idx / (shape.y * shape.x);
  idx -= *n * shape.y * shape.x;
  *y = idx / shape.x;
  idx -= *y * shape.x;
  *x = idx;
}

// https://www.cs.rit.edu/~ncs/color/t_convert.html
template <typename Dtype>
__device__ void convert_rgb_to_hsv(
    Dtype r, Dtype g, Dtype b,
    Dtype* h, Dtype* s, Dtype* v
) {
  Dtype min_v = min(min(r, g), b);
  Dtype max_v = max(max(r, g), b);  // NOLINT(build/include_what_you_use)
  Dtype delta = max_v - min_v;

  if (max_v == 0 || delta == 0) {
    *h = 0;
    *s = 0;
    *v = max_v;
    return;
  }

  if (r == max_v) {
    *h = (g - b) / delta;
  } else if (g == max_v) {
    *h = 2 + (b - r) / delta;
  } else {
    *h = 4 + (r - g) / delta;
  }

  *h *= 60;
  if (h < 0) {
    *h += 360;
  }
  *s = delta / max_v;
  *v = max_v;
}


// https://www.cs.rit.edu/~ncs/color/t_convert.html
template <typename Dtype>
__device__ void convert_hsv_to_rgb(
    Dtype h, Dtype s, Dtype v,
    Dtype* r, Dtype* g, Dtype* b
) {
  int i;
  Dtype f, p, q, t;

  if (s == 0) {
    *r = v;
    *g = v;
    *b = v;
    return;
  }

  h /= 60;  // sector 0 to 5
  i = floor(h);
  f = h - i;  // factorial part of h
  p = v * (1 - s);
  q = v * (1 - s * f);
  t = v * (1 - s * (1 - f));

  switch (i) {
    case 0:
      *r = v;
      *g = t;
      *b = p;
      break;
    case 1:
      *r = q;
      *g = v;
      *b = p;
      break;
    case 2:
      *r = p;
      *g = v;
      *b = t;
      break;
    case 3:
      *r = p;
      *g = q;
      *b = v;
      break;
    case 4:
      *r = t;
      *g = p;
      *b = v;
      break;
    default:  // case 5:
      *r = v;
      *g = p;
      *b = q;
      break;
  }
}


template <typename Dtype>
__global__ void color_transformations(
    const Dtype* src_data, Dtype* dst_data,
    const uint4 shape, const AugmentSelection* aug_data
) {
  CUDA_KERNEL_LOOP(loop_index, shape.x * shape.y * shape.w) {
    int x, y, n;
    get_pixel_indices(loop_index, shape, &x, &y, &n);

    // check what needs doing
    const AugmentSelection& as = aug_data[n];
    const bool doHueRotation = (abs(as.hue_rotation) > FLT_EPSILON);
    const bool doDesaturation = (as.saturation < (1.0 - 1.0/UINT8_MAX));

    // N*cs*hs*ws + H*ws + W
    int index = n * shape.z * shape.y * shape.x + y * shape.x + x;
    // hs*ws
    const int channel_stride = shape.y * shape.x;

    // read
    Dtype r = src_data[index + 0 * channel_stride];
    Dtype g = src_data[index + 1 * channel_stride];
    Dtype b = src_data[index + 2 * channel_stride];

    if (doHueRotation || doDesaturation) {
      // transform
      Dtype h, s, v;
      convert_rgb_to_hsv(r, g, b, &h, &s, &v);
      if (doHueRotation) {
        h -= aug_data[n].hue_rotation;
      }
      if (doDesaturation) {
        s *= aug_data[n].saturation;
      }
      convert_hsv_to_rgb(h, s, v, &r, &g, &b);
    }

    // write
    dst_data[index + 0 * channel_stride] = r;
    dst_data[index + 1 * channel_stride] = g;
    dst_data[index + 2 * channel_stride] = b;
  }
}


// Mean is WxHxC
// For each pixel in the current image, subtract the corresponding pixel
// from the mean image
template <typename Dtype>
__global__ void pixel_mean_subtraction(
    Dtype* data, const Dtype* mean_data, const uint4 shape
) {
  CUDA_KERNEL_LOOP(loop_index, shape.x * shape.y * shape.w) {
    int x, y, n;
    get_pixel_indices(loop_index, shape, &x, &y, &n);

    for (int c = 0; c < shape.z; c++) {
      // N*cs*hs*ws + C*hs*ws + H*ws + W
      const int data_idx = (n * shape.z * shape.y * shape.x) +
        (c * shape.y * shape.x) + (y * shape.x) + x;
      // C*hs*ws + H*ws + W
      const int mean_idx = (c * shape.y * shape.x) + (y * shape.x) + x;
      data[data_idx] -= mean_data[mean_idx];
    }
  }
}


// Mean is 1x1xC
// For each pixel in the current image, subtract the mean pixel
template <typename Dtype>
__global__ void channel_mean_subtraction(
    Dtype* data, const uint4 shape,
    const Dtype mean_value1, const Dtype mean_value2, const Dtype mean_value3
) {
  CUDA_KERNEL_LOOP(loop_index, shape.x * shape.y * shape.w) {
    int x, y, n;
    get_pixel_indices(loop_index, shape, &x, &y, &n);

    // N*cs*hs*ws + C*hs*ws + H*ws + W
    const int data_idx = (n * shape.z * shape.y * shape.x) +(y * shape.x) + x;
    // hs*ws
    const int channel_stride = shape.y * shape.x;

    data[data_idx + 0 * channel_stride] -= mean_value1;
    data[data_idx + 1 * channel_stride] -= mean_value2;
    data[data_idx + 2 * channel_stride] -= mean_value3;
  }
}


template <typename Dtype>
__device__ void rotate_point(
    const Dtype ax, const Dtype ay,  // original point
    const Dtype cx, const Dtype cy,  // center point
    float angle,
    Dtype* bx, Dtype* by  // destination point
) {
  const Dtype s = sin(angle);
  const Dtype c = cos(angle);

  // translate to origin
  const Dtype tx = ax - cx;
  const Dtype ty = ay - cy;

  *bx = (tx * c) - (ty * s) + cx;
  *by = (tx * s) + (ty * c) + cy;
}


template <typename Dtype>
__device__ Dtype get_value(
    const Dtype* data, const uint4& shape,
    const unsigned int n, const unsigned int c,
    int y, int x
) {
  // Replicate border for 1 pixel
  if (x == -1) x = 0;
  if (x == shape.x) x = shape.x - 1;
  if (y == -1) y = 0;
  if (y == shape.y) y = shape.y - 1;

  if (x >= 0 && x < shape.x && y >= 0 && y < shape.y) {
    // N*cs*hs*ws + C*hs*ws + H*ws + W
    return data[(n * shape.z * shape.y * shape.x) +
      (c * shape.y * shape.x) + (y * shape.x) + x];
  } else {
    return 0;
  }
}


template <typename Dtype>
__device__ Dtype cubic_interpolation(const Dtype& d,
    const Dtype& v1, const Dtype& v2, const Dtype& v3, const Dtype& v4
) {
  // d is [0,1], marking the distance from v2 towards v3
  return v2 + d * (
      -2.0 * v1 - 3.0 * v2 + 6.0 * v3 - 1.0 * v4 + d * (
       3.0 * v1 - 6.0 * v2 + 3.0 * v3 + 0.0 * v4 + d * (
      -1.0 * v1 + 3.0 * v2 - 3.0 * v3 + 1.0 * v4))) / 6.0;
}


// Interpolate in 1D space
template <typename Dtype>
__device__ Dtype interpolate_x(
    const Dtype* data, const uint4& shape,
    const unsigned int n, const unsigned int c,
    const int y, const Dtype x
) {
  Dtype dx = x - floor(x);
  return cubic_interpolation(dx,
      get_value(data, shape, n, c, y, floor(x) - 1),
      get_value(data, shape, n, c, y, floor(x)),
      get_value(data, shape, n, c, y, ceil(x)),
      get_value(data, shape, n, c, y, ceil(x) + 1));
}


// Interpolate in 2D space
template <typename Dtype>
__device__ Dtype interpolate_xy(
    const Dtype* data, const uint4& shape,
    const unsigned int n, const unsigned int c,
    const Dtype y, const Dtype x
) {
  Dtype dy = y - floor(y);
  return cubic_interpolation(dy,
      interpolate_x(data, shape, n, c, floor(y) - 1, x),
      interpolate_x(data, shape, n, c, floor(y), x),
      interpolate_x(data, shape, n, c, ceil(y), x),
      interpolate_x(data, shape, n, c, ceil(y) + 1, x));
}


template <typename Dtype>
__global__ void spatial_transformations(
    const Dtype* src_data, const uint4 src_shape,
    const AugmentSelection* aug_data,
    Dtype* dst_data, const uint4 dst_shape
) {
  CUDA_KERNEL_LOOP(loop_index, dst_shape.x * dst_shape.y * dst_shape.w) {
    int dst_x, dst_y, n;
    get_pixel_indices(loop_index, dst_shape, &dst_x, &dst_y, &n);
    const AugmentSelection& as = aug_data[n];

    // calculate src pixel indices for this thread
    Dtype x = dst_x;
    Dtype y = dst_y;
    // crop
    x += as.crop_offset.x;
    y += as.crop_offset.y;
    // rotate
    if (abs(as.rotation) > FLT_EPSILON) {
      const Dtype w_before = as.scale.width - 1;
      const Dtype h_before = as.scale.height - 1;
      const float angle = as.rotation * CUDART_PI_F / 180.0f;
      const Dtype w_after = abs(w_before * cos(angle)) +
        abs(h_before * sin(angle));
      const Dtype h_after = abs(w_before * sin(angle)) +
        abs(h_before * cos(angle));
      rotate_point(x, y, w_after / 2.0f, h_after / 2.0f,
          -angle, &x, &y);
      x -= (w_after - w_before) / 2.0f;
      y -= (h_after - h_before) / 2.0f;
    }
    // scale
    if (src_shape.x != as.scale.width) {
      x *= Dtype(src_shape.x - 1) / (as.scale.width - 1);
    }
    if (src_shape.y != as.scale.height) {
      y *= Dtype(src_shape.y - 1) / (as.scale.height - 1);
    }
    // flip
    if (as.flip) {
      x = (src_shape.x - x - 1.0);
    }

    for (int c = 0; c < dst_shape.z; c++) {
      // N*cs*hs*ws + C*hs*ws + H*ws + W
      const int dst_idx = (n * dst_shape.z * dst_shape.y * dst_shape.x) +
        (c * dst_shape.y * dst_shape.x) + (dst_y * dst_shape.x) + dst_x;
      dst_data[dst_idx] = interpolate_xy(src_data, src_shape, n, c, y, x);
    }
  }
}


template <typename Dtype>
void DetectNetTransformationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  AugmentSelection* aug_data = reinterpret_cast<AugmentSelection*>(
      gpu_workspace_augmentations_.data());
  Dtype* tmp_data = reinterpret_cast<Dtype*>(
      gpu_workspace_tmpdata_.data());

  const uint4 bottom_shape = make_uint4(
      bottom[0]->shape(3),   // x = W
      bottom[0]->shape(2),   // y = H
      bottom[0]->shape(1),   // z = C
      bottom[0]->shape(0));  // w = N
  const int bottom_count = bottom[0]->count();
  const int bottom_pixels = bottom_shape.x * bottom_shape.y * bottom_shape.w;
  const uint4 top_shape = make_uint4(
      top[0]->shape(3),   // x = W
      top[0]->shape(2),   // y = H
      top[0]->shape(1),   // z = C
      top[0]->shape(0));  // w = N
  const int top_count = top[0]->count();
  const int top_pixels = top_shape.x * top_shape.y * top_shape.w;

  // Get current stream
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = GPUMemory::device_stream(device);

  // Make augmentation selections for each image
  vector<AugmentSelection> augmentations;
  for (int i = 0; i < bottom_shape.w; i++) {
    augmentations.push_back(get_augmentations(
        cv::Point(bottom_shape.x, bottom_shape.y)));
  }
  // Copy augmentation selections to GPU
  size_t aug_data_sz = sizeof(AugmentSelection) * augmentations.size();
  caffe_gpu_memcpy(aug_data_sz, &augmentations[0], aug_data);


  // Color transformations
  // NOLINT_NEXT_LINE(whitespace/operators)
  color_transformations<<<CAFFE_GET_BLOCKS(bottom_pixels),
    CAFFE_CUDA_NUM_THREADS>>>(bottom_data, tmp_data, bottom_shape, aug_data);

  // Mean subtraction
  if (t_param_.has_mean_file()) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    pixel_mean_subtraction<<<CAFFE_GET_BLOCKS(bottom_pixels),
      CAFFE_CUDA_NUM_THREADS>>>(tmp_data, mean_blob_.gpu_data(), bottom_shape);
  } else if (t_param_.mean_value_size() != 0) {
    CHECK_EQ(bottom_shape.z, 3) << "Data must have 3 channels when "
      "using transform_param.mean_value.";
    // NOLINT_NEXT_LINE(whitespace/operators)
    channel_mean_subtraction<<<CAFFE_GET_BLOCKS(bottom_pixels),
      CAFFE_CUDA_NUM_THREADS>>>(tmp_data, bottom_shape,
        mean_values_[0] * UINT8_MAX,
        mean_values_[1] * UINT8_MAX,
        mean_values_[2] * UINT8_MAX);
  }

  // Spatial transformations
  // NOLINT_NEXT_LINE(whitespace/operators)
  spatial_transformations<<<CAFFE_GET_BLOCKS(top_pixels),
    CAFFE_CUDA_NUM_THREADS>>>(tmp_data, bottom_shape, aug_data,
        top_data, top_shape);

  // Use CPU to transform labels
  const vector<vector<BboxLabel> > list_list_bboxes = blobToLabels(*bottom[1]);
  for (size_t i = 0; i < bottom[1]->num(); i++) {
    const vector<BboxLabel>& list_bboxes = list_list_bboxes[i];
    Dtype* output_label = &top[1]->mutable_cpu_data()[
      top[1]->offset(i, 0, 0, 0)
      ];
    transform_label_cpu(list_bboxes, output_label, augmentations[i],
        cv::Size(bottom_shape.x, bottom_shape.y));
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DetectNetTransformationLayer);


}  // namespace caffe

#endif
