#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
    const bool normalized) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template __host__ __device__ float BBoxSizeGPU(const float* bbox,
    const bool normalized);
template __host__ __device__ double BBoxSizeGPU(const double* bbox,
    const bool normalized);

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
    const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSizeGPU(bbox1);
    const Dtype bbox2_size = BBoxSizeGPU(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template __host__ __device__ float JaccardOverlapGPU(const float* bbox1,
    const float* bbox2);
template __host__ __device__ double JaccardOverlapGPU(const double* bbox1,
    const double* bbox2);

template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const int num_priors,
          const bool share_location, const int num_loc_classes,
          const int background_label_id, Dtype* bbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;
    const int c = (index / 4) % num_loc_classes;
    const int d = (index / 4 / num_loc_classes) % num_priors;
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    int pi = d * 4;
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
      pi += i;
      const int vi = pi + num_priors * 4;
      bbox_data[index] = prior_data[pi] + loc_data[index] * prior_data[vi];
    } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      const Dtype decode_bbox_center_x = xmin * prior_width + prior_center_x;
      const Dtype decode_bbox_center_y = ymin * prior_height + prior_center_y;
      const Dtype decode_bbox_width = exp(xmax) * prior_width;
      const Dtype decode_bbox_height = exp(ymax) * prior_height;

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else {
      // Unknown code type.
    }
  }
}

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const int num_priors,
          const bool share_location, const int num_loc_classes,
          const int background_label_id, Dtype* bbox_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  DecodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, loc_data, prior_data, code_type,
      num_priors, share_location, num_loc_classes, background_label_id,
      bbox_data);
  CUDA_POST_KERNEL_CHECK;
}

template void DecodeBBoxesGPU(const int nthreads,
          const float* loc_data, const float* prior_data,
          const CodeType code_type, const int num_priors,
          const bool share_location, const int num_loc_classes,
          const int background_label_id, float* bbox_data);
template void DecodeBBoxesGPU(const int nthreads,
          const double* loc_data, const double* prior_data,
          const CodeType code_type, const int num_priors,
          const bool share_location, const int num_loc_classes,
          const int background_label_id, double* bbox_data);

template <typename Dtype>
__global__ void ComputeOverlappedKernel(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_bboxes;
    const int i = (index / num_bboxes) % num_bboxes;
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    const int c = (index / num_bboxes / num_bboxes) % num_classes;
    const int n = index / num_bboxes / num_bboxes / num_classes;
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = ((n * num_bboxes + i) * num_classes + c) * 4;
    const int start_loc_j = ((n * num_bboxes + j) * num_classes + c) * 4;
    const Dtype overlap = JaccardOverlapGPU(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, num_bboxes, num_classes,
      overlap_threshold, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedGPU(const int nthreads,
          const float* bbox_data, const int num_bboxes, const int num_classes,
          const float overlap_threshold, bool* overlapped_data);
template void ComputeOverlappedGPU(const int nthreads,
          const double* bbox_data, const int num_bboxes, const int num_classes,
          const double overlap_threshold, bool* overlapped_data);

}  // namespace caffe
