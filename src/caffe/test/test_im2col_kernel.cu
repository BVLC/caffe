#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/im2col_layer.hpp"
#include "caffe/util/im2col.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// Forward declare kernel functions
template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col);

template <typename Dtype, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const Dtype* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col);

template <typename Dtype>
class Im2colKernelTest : public GPUDeviceTest<Dtype> {
 protected:
  Im2colKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(5, 500, 15, 15)),
        blob_kernel_shape_(new Blob<int>()),
        blob_stride_(new Blob<int>()),
        blob_pad_(new Blob<int>()),
        blob_dilation_(new Blob<int>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    vector<int> dim_blob_shape(1, 2);
    blob_kernel_shape_->Reshape(dim_blob_shape);
    blob_stride_->Reshape(dim_blob_shape);
    blob_pad_->Reshape(dim_blob_shape);
    blob_dilation_->Reshape(dim_blob_shape);

    height_ = blob_bottom_->height();
    width_ = blob_bottom_->width();
    channels_ = blob_bottom_->channels();
    pad_ = 0;
    stride_ = 2;
    dilation_ = 3;
    kernel_size_ = 3;
    height_col_ = (height_ + 2 * pad_ -
        (dilation_ * (kernel_size_ - 1) + 1)) / stride_ + 1;
    width_col_ = (width_ + 2 * pad_ -
        (dilation_ * (kernel_size_ - 1) + 1)) / stride_ + 1;

    for (int i = 0; i < 2; ++i) {
      blob_kernel_shape_->mutable_cpu_data()[i] = kernel_size_;
      blob_stride_->mutable_cpu_data()[i] = stride_;
      blob_pad_->mutable_cpu_data()[i] = pad_;
      blob_dilation_->mutable_cpu_data()[i] = dilation_;
    }
  }

  virtual ~Im2colKernelTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_cpu_;
    delete blob_kernel_shape_;
    delete blob_stride_;
    delete blob_pad_;
    delete blob_dilation_;
  }

  Blob<int>* const blob_kernel_shape_;
  Blob<int>* const blob_stride_;
  Blob<int>* const blob_pad_;
  Blob<int>* const blob_dilation_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int height_;
  int width_;
  int channels_;
  int pad_;
  int stride_;
  int dilation_;
  int kernel_size_;
  int height_col_;
  int width_col_;
};

TYPED_TEST_CASE(Im2colKernelTest, TestDtypes);

TYPED_TEST(Im2colKernelTest, Test2D) {
  // Reshape the blobs to correct size for im2col output
  this->blob_top_->Reshape(this->blob_bottom_->num(),
          this->channels_ * this->kernel_size_ * this->kernel_size_,
          this->height_col_,
          this->width_col_);

  this->blob_top_cpu_->Reshape(this->blob_bottom_->num(),
          this->channels_ * this->kernel_size_ * this->kernel_size_,
          this->height_col_,
          this->width_col_);

  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  TypeParam* cpu_data = this->blob_top_cpu_->mutable_cpu_data();

  // CPU Version
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    im2col_cpu(this->blob_bottom_->cpu_data() + this->blob_bottom_->offset(n),
      this->channels_, this->height_, this->width_,
      this->kernel_size_, this->kernel_size_, this->pad_, this->pad_,
      this->stride_, this->stride_, this->dilation_, this->dilation_,
      cpu_data + this->blob_top_cpu_->offset(n));
  }

  // GPU version
  int num_kernels = this->channels_ * this->height_col_ * this->width_col_;
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);

  // Launch with different grid sizes
  for (int grid_div = 2; grid_div <= 8; grid_div++) {
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      int grid_dim = default_grid_dim/grid_div;
      // NOLINT_NEXT_LINE(whitespace/operators)
      im2col_gpu_kernel<TypeParam><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data + this->blob_bottom_->offset(n),
        this->height_, this->width_, this->kernel_size_, this->kernel_size_,
        this->pad_, this->pad_, this->stride_, this->stride_,
        this->dilation_, this->dilation_,
        this->height_col_, this->width_col_,
        top_data + this->blob_top_->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }

    // Compare results against CPU version
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      TypeParam cpuval = cpu_data[i];
      TypeParam gpuval = this->blob_top_->cpu_data()[i];
      EXPECT_EQ(cpuval, gpuval);
      if (cpuval != gpuval) {
        break;
      }
    }
  }
}

TYPED_TEST(Im2colKernelTest, TestND) {
  // Reshape the blobs to correct size for im2col output
  this->blob_top_->Reshape(this->blob_bottom_->num(),
      this->channels_ * this->kernel_size_ * this->kernel_size_,
      this->height_col_,
      this->width_col_);

  this->blob_top_cpu_->ReshapeLike(*this->blob_top_);

  const TypeParam* bottom_data_cpu = this->blob_bottom_->cpu_data();
  TypeParam* top_data_cpu = this->blob_top_cpu_->mutable_cpu_data();

  // CPU Version
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    im2col_nd_cpu(bottom_data_cpu + this->blob_bottom_->offset(n), 2,
        this->blob_bottom_->shape().data() + 1,
        this->blob_top_cpu_->shape().data() + 1,
        this->blob_kernel_shape_->cpu_data(),
        this->blob_pad_->cpu_data(), this->blob_stride_->cpu_data(),
        this->blob_dilation_->cpu_data(),
        top_data_cpu + this->blob_top_cpu_->offset(n));
  }

  // GPU version
  int num_kernels = this->channels_ * this->height_col_ * this->width_col_;
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);
  const TypeParam* bottom_data_gpu = this->blob_bottom_->gpu_data();

  // Launch with different grid sizes
  for (int grid_div = 2; grid_div <= 8; grid_div++) {
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      const int grid_dim = default_grid_dim / grid_div;
      TypeParam* top_data_gpu = this->blob_top_->mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      im2col_nd_gpu_kernel<TypeParam, 2><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels, bottom_data_gpu + this->blob_bottom_->offset(n),
          this->blob_bottom_->gpu_shape() + 1, this->blob_top_->gpu_shape() + 1,
          this->blob_kernel_shape_->gpu_data(), this->blob_pad_->gpu_data(),
          this->blob_stride_->gpu_data(), this->blob_dilation_->gpu_data(),
          top_data_gpu + this->blob_top_->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }

    // Compare results against CPU version
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      TypeParam cpuval = top_data_cpu[i];
      TypeParam gpuval = this->blob_top_->cpu_data()[i];
      EXPECT_EQ(cpuval, gpuval);
      if (cpuval != gpuval) {
        break;
      }
    }
  }
}

}  // namespace caffe
