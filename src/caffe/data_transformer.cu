#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/gpu_memory.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
__global__
void transform_kernel(int N, int C,
                      int H, int W,  // original size
                      int Hc, int Wc,  // cropped size
                      bool param_mirror,
                      int datum_height, int datum_width,  // offsets
                      int crop_size, Phase phase,
                      const Dtype *in,
                      Dtype *out,  // buffers
                      const Dtype scale,
                      int has_mean_file,
                      int has_mean_values,
                      Dtype *mean,
                      int *random_numbers) {
  const int c = blockIdx.y;

  // loop over images
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    // get mirror and offsets
    int rand1 = random_numbers[n*3    ];
    int rand2 = random_numbers[n*3 + 1];
    int rand3 = random_numbers[n*3 + 2];

    bool mirror = param_mirror && (rand1 % 2);
    int h_off = 0, w_off = 0;
    if (crop_size) {
      if (phase == TRAIN) {
        h_off = rand2 % (datum_height - crop_size + 1);
        w_off = rand3 % (datum_width - crop_size + 1);
      } else {
        h_off = (datum_height - crop_size) / 2;
        w_off = (datum_width - crop_size) / 2;
      }
    }

    // channel is handled by blockIdx.y

    // offsets into start of (image, channel) = (n, c)
    const Dtype *in_ptr  = &in[n*C*H*W + c*H*W];
    Dtype *out_ptr = &out[n*C*Hc*Wc + c*Hc*Wc];

    // loop over pixels using threads
    for (int h = threadIdx.y; h < Hc; h += blockDim.y) {
      for (int w = threadIdx.x; w < Wc; w += blockDim.x) {
        // get the indices for in, out buffers
        int in_idx  = (h_off + h)*W + w_off + w;
        int out_idx;
        if (mirror) {
          out_idx = h*Wc + (Wc - 1 - w);
        } else {
          out_idx = h*Wc + w;
        }
        const Dtype element = in_ptr[in_idx];

        // perform the transform
        if (has_mean_file) {
          out_ptr[out_idx] =
            (element - mean[c*H*W + in_idx]) * scale;
        } else {
          if (has_mean_values) {
            out_ptr[out_idx] =
              (element - mean[c]) * scale;
          } else {
            out_ptr[out_idx] = element * scale;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::TransformGPU(int N, int C, int H, int W,
                                          const Dtype *in, Dtype *out,
                                          int *random_numbers) {
  const int datum_channels = C;
  const int datum_height = H;
  const int datum_width = W;

  const int crop_size = param_.crop_size();
  Dtype scale = param_.scale();
  const bool mirror = param_.mirror();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    // no need to check equality anymore
    // datum_{height, width} are _output_ not input
    mean = data_mean_.mutable_gpu_data();
  }
  // will send
  if (has_mean_values) {
    if (!mean_values_gpu_ptr_) {
      CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << datum_channels;
      if (datum_channels > 1 && mean_values_.size() == 1) {
        // Replicate the mean_value for simplicity
        for (int c = 1; c < datum_channels; ++c) {
          mean_values_.push_back(mean_values_[0]);
        }
      }
      int device;
      CUDA_CHECK(cudaGetDevice(&device));
      GPUMemory::allocate(reinterpret_cast<void **>(&mean_values_gpu_ptr_),
          sizeof(Dtype)*mean_values_.size(), device, cudaStreamDefault);
      caffe_copy(static_cast<int>(mean_values_.size()),
                 reinterpret_cast<Dtype *>(&mean_values_[0]),
                 mean_values_gpu_ptr_);
    }

    mean = mean_values_gpu_ptr_;
  }

  int height = datum_height;
  int width = datum_width;

  if (crop_size) {
    height = crop_size;
    width = crop_size;
  }

  dim3 grid(N, C);
  dim3 block(16, 16);

  transform_kernel<Dtype>
  <<< grid, block, 0, cudaStreamDefault >>>(N, C, H, W,
                                            height, width,
                                            param_.mirror(),
                                            datum_height, datum_width,
                                            crop_size, phase_,
                                            in, out,
                                            scale,
                                            static_cast<int>(has_mean_file),
                                            static_cast<int>(has_mean_values),
                                            mean, random_numbers);
  CUDA_POST_KERNEL_CHECK;
}

template void DataTransformer<float>::TransformGPU(int, int, int, int,
                                        const float*, float*, int*);
template void DataTransformer<double>::TransformGPU(int, int, int, int,
                                        const double*, double*, int*);
}  // namespace caffe
