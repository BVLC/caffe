#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// Forward declare kernel functions
template <typename Dtype>
__global__ void vol2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int depth, const int kernel_h,
    const int kernel_w, const int kernel_d, const int pad_h, const int pad_w,
    const int pad_d, const int stride_h, const int stride_w, const int stride_d,
    const int height_col, const int width_col, const int depth_col,
    Dtype* data_col);

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class Vol2colKernelTest : public ::testing::Test {
 protected:
  Vol2colKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(vector<int>())),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    // update the blob shape
    int bot_shape_arr[] = {5, 500, 10, 10, 10};
    vector<int> bot_shape(bot_shape_arr, bot_shape_arr +
        sizeof(bot_shape_arr) / sizeof(int));
    blob_bottom_->Reshape(bot_shape);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    vector<int> shape = blob_bottom_->shape();
    channels_ = shape[1];
    height_ = shape[2];
    width_ = shape[3];
    depth_ = shape[4];
    pad_ = 0;
    stride_ = 2;
    kernel_size_ = 3;
    height_col_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    width_col_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    depth_col_ = (depth_ + 2 * pad_ - kernel_size_) / stride_ + 1;
  }

  virtual ~Vol2colKernelTest() {
      delete blob_bottom_;
      delete blob_top_;
      delete blob_top_cpu_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int height_;
  int width_;
  int depth_;
  int channels_;
  int pad_;
  int stride_;
  int kernel_size_;
  int height_col_;
  int width_col_;
  int depth_col_;
};

TYPED_TEST_CASE(Vol2colKernelTest, TestDtypes);

TYPED_TEST(Vol2colKernelTest, TestGPU) {
  Caffe::set_mode(Caffe::GPU);

  // Reshape the blobs to correct size for vol2col output
  vector<int> top_shape(5);
  top_shape[0] = this->blob_bottom_->shape(0);
  top_shape[1] = this->channels_ * this->kernel_size_ * this->kernel_size_ *
      this->kernel_size_;
  top_shape[2] = this->height_col_;
  top_shape[3] = this->width_col_;
  top_shape[4] = this->depth_col_;
  this->blob_top_->Reshape(top_shape);
  this->blob_top_cpu_->Reshape(top_shape);

  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  TypeParam* cpu_data = this->blob_top_cpu_->mutable_cpu_data();

  // CPU Version
  for (int n = 0; n < this->blob_bottom_->shape(0); ++n) {
    vol2col_cpu(this->blob_bottom_->cpu_data() +
      this->blob_bottom_->offset(vector<int>(1, n)), this->channels_,
      this->height_, this->width_, this->depth_, this->kernel_size_,
      this->kernel_size_, this->kernel_size_, this->pad_, this->pad_,
      this->pad_, this->stride_, this->stride_, this->stride_, cpu_data
      + this->blob_top_cpu_->offset(vector<int>(1, n)));
  }

  // GPU version
  int num_kernels = this->channels_ * this->height_col_ * this->width_col_ *
      this->depth_col_;
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);

  // Launch with different grid sizes
  for (int grid_div = 2; grid_div <= 8; grid_div++) {
    for (int n = 0; n < this->blob_bottom_->shape(0); ++n) {
      int grid_dim = default_grid_dim/grid_div;
      // NOLINT_NEXT_LINE(whitespace/operators)
      vol2col_gpu_kernel<TypeParam><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data +
        this->blob_bottom_->offset(vector<int>(1, n)), this->height_,
        this->width_, this->depth_, this->kernel_size_, this->kernel_size_,
        this->kernel_size_, this->pad_, this->pad_, this->pad_, this->stride_,
        this->stride_, this->stride_, this->height_col_, this->width_col_,
        this->depth_col_, top_data +
        this->blob_top_->offset(vector<int>(1, n)));
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

}  // namespace caffe
