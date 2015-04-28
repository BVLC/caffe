#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// Forward declare kernel functions
template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col);

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class Im2colKernelTest : public ::testing::Test {
 protected:
  Im2colKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(5, 500, 10, 10)),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    height_ = blob_bottom_->height();
    width_ = blob_bottom_->width();
    channels_ = blob_bottom_->channels();
    pad_ = 0;
    hole_ = 1;
    stride_ = 2;
    kernel_size_ = 3;
    const int kernel_size_eff = kernel_size_ + (kernel_size_ - 1) * (hole_ - 1);
    height_col_ = (height_ + 2 * pad_ - kernel_size_eff) / stride_ + 1;
    width_col_ = (width_ + 2 * pad_ - kernel_size_eff) / stride_ + 1;
  }

  virtual ~Im2colKernelTest() {
      delete blob_bottom_;
      delete blob_top_;
      delete blob_top_cpu_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int height_;
  int width_;
  int channels_;
  int pad_;
  int hole_;
  int stride_;
  int kernel_size_;
  int height_col_;
  int width_col_;
};

TYPED_TEST_CASE(Im2colKernelTest, TestDtypes);

TYPED_TEST(Im2colKernelTest, TestGPU) {
  Caffe::set_mode(Caffe::GPU);

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
  im2col_cpu(this->blob_bottom_->cpu_data(),
	     this->blob_bottom_->num(), this->channels_, this->height_, this->width_,
	     this->kernel_size_, this->kernel_size_, this->pad_, this->pad_,
	     this->stride_, this->stride_, this->hole_, this->hole_,
	     cpu_data);

  // GPU version
  im2col_gpu(this->blob_bottom_->gpu_data(),
	     this->blob_bottom_->num(), this->channels_, this->height_, this->width_,
	     this->kernel_size_, this->kernel_size_, this->pad_, this->pad_,
	     this->stride_, this->stride_, this->hole_, this->hole_,
	     top_data);

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

}  // namespace caffe
