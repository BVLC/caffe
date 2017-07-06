#ifdef CMAKE_BUILD
#include "caffe_config.h"
#endif
#ifdef USE_INTEL_SPATIAL
#include <sstream>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/conv_spatial_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/cl_kernels.hpp"
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#include "viennacl/tools/sha1.hpp"
#endif

#include <boost/filesystem.hpp>

// #define TEST_ALL_KERNELS

namespace caffe {

#define ALIGN(val, N) (((val) + (N) - 1) & ~((N) - 1))

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int_tp input_dim = this->input_shape(i + 1);
    const int_tp kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1)
        + 1;
    const int_tp output_dim = (input_dim + 2 * pad_data[i]
        - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  tuned_ = 0;
  // Calculate variables used for kernel generation
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  kernel_h_ = kernel_shape_data[0];
  kernel_w_ = kernel_shape_data[1];
  const int_tp* pad_data = this->pad_.cpu_data();
  pad_h_ = pad_data[0];
  pad_w_ = pad_data[1];
  const int_tp* stride_data = this->stride_.cpu_data();
  stride_h_ = stride_data[0];
  stride_w_ = stride_data[1];
  const int_tp* dilation_data = this->dilation_.cpu_data();
  dilation_h_ = dilation_data[0];
  dilation_w_ = dilation_data[1];
  M_ = this->num_output_ / this->group_;
  K_ = this->channels_ * kernel_h_ * kernel_w_ / this->group_;
  swizzled_weights_blob_.Reshape(ALIGN(this->num_output_, 16),
                            this->channels_,
                            kernel_h_, ALIGN(kernel_w_, 2));
  swizzled_weights_ = NULL;
  bias_ = NULL;

  if (IsFusedWithEltwiseReLU()) {
    CHECK_EQ(
      this->layer_param().convolution_param().eltwise_param().coeff_size(),
      0);
    CHECK_EQ(bottom.size(), 2);
    op_ = this->layer_param_.eltwise_param().operation();
    CHECK_EQ(op_, EltwiseParameter_EltwiseOp_SUM);
  }

  if (IsFusedWithReLU())
    negative_slope_ =
      this->layer_param_.convolution_param().relu_param().negative_slope();
  else
    negative_slope_ = 0;

  if (std::getenv("CLCAFFE_CACHE_PATH"))
    cache_path_ << std::getenv("CLCAFFE_CACHE_PATH");
  else if (std::getenv("VIENNACL_CACHE_PATH"))
    cache_path_ << std::getenv("VIENNACL_CACHE_PATH") << "/clCaffe";
  else if (std::getenv("HOME")) {
    cache_path_ << std::getenv("HOME") << "/.cache/clCaffe";
  }
  cache_path_ << "/spatialkernels/";
  const boost::filesystem::path& path = cache_path_.str();
  const boost::filesystem::path& dir =
                 boost::filesystem::unique_path(path).string();
  bool hasCacheDir = false;
  if (!boost::filesystem::exists(dir))
    hasCacheDir = boost::filesystem::create_directories(dir);
  else
    hasCacheDir = boost::filesystem::is_directory(dir);

  if (hasCacheDir != true) {
    std::cout << "Failed to create cache directory,"
              << "will tune again for next running" << std::endl;
    return;
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (IsFusedWithEltwiseReLU()) {
    const vector<Blob<Dtype>*> bottom_image(bottom.begin(), bottom.end() - 1);
    BaseConvolutionLayer<Dtype>::Reshape(bottom_image, top);
  } else {
    BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  }
  height_ = bottom[0]->shape(this->channel_axis_ + 1);
  width_ = bottom[0]->shape(this->channel_axis_ + 2);
  const int_tp kernel_extent_h = dilation_h_ * (kernel_h_ - 1) + 1;
  const int_tp kernel_extent_w = dilation_w_ * (kernel_w_ - 1) + 1;
  output_h_ = (height_ + 2 * pad_h_ - kernel_extent_h) / stride_h_ + 1;
  output_w_ = (width_ + 2 * pad_w_ - kernel_extent_w) / stride_w_ + 1;

  // Shape the tops.
  vector<int_tp> top_shape(bottom[0]->shape().begin(),
                           bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }

  for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }

  CHECK_EQ(2, this->num_spatial_axes_)
    << "ConvolutionSpatial input must have 2 spatial axes "
    << "(e.g., height and width). ";

  const int_tp height_out = top[0]->shape(this->channel_axis_ + 1);
  const int_tp width_out = top[0]->shape(this->channel_axis_ + 2);
  N_ = height_out * width_out;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (this->bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  if (!std::is_same<Dtype, double>::value) {
    this->num_ = bottom[0]->count(0, this->channel_axis_);
    SetUp(bottom, top, Caffe::GetDefaultDevice()->backend());
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  CHECK_EQ(IsFusedWithEltwiseReLU() == false && IsFusedWithReLU() == false,
           true);
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifndef CPU_ONLY
#ifdef USE_GREENTEA

  #define dbg
#ifdef dbg
#define dbgPrint(x) (x)
#else
#define dbgPrint(x)
#endif

// For large enough input size, we do not need to tune kernels for different
// size. The reason is with large input size, there will be enough work items
// to feed al the EUs.

#define TUNING_SIZE(x) ((x) > 256 ? 256 : (ALIGN(x, 16)))


template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::generate_key() {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  std::stringstream keyBuilder;
  if (std::is_same<Dtype, float>::value)
    keyBuilder << "float_";
  else
    keyBuilder << "half_";
  keyBuilder << this->layer_param_.convolution_param().fuse_type() << "_"
             << kernel_w_ << "_"
             << kernel_h_ << "_"
             << this->channels_ << "_"
             << this->group_ << "_"
             << stride_h_ << "_"
             << stride_w_ << "_"
             << dilation_h_ << "_"
             << dilation_w_ << "_"
             << this->bias_term_ << "_"
             << TUNING_SIZE(width_) << "_"
             << TUNING_SIZE(height_) << "_"
             << pad_w_ << "_"
             << pad_h_ << "_"
             << this->num_ << "_"
             << M_;

  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                (this->device_->id());
  std::string prefix = ctx.current_device().name()
                  + ctx.current_device().vendor()
                  + ctx.current_device().driver_version()
                  + std::to_string(ctx.current_device().max_compute_units());
  key_ = viennacl::tools::sha1(prefix + keyBuilder.str());
  short_key_ = keyBuilder.str();
}

template<typename Dtype>
std::string ConvolutionLayerSpatial<Dtype>::generate_specific_key(
    int_tp type, int_tp blockWidth, int_tp blockHeight, int_tp blockDepth) {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  std::stringstream keyBuilder;
  keyBuilder << short_key_
             << "_" << type
             << "_" << blockWidth
             << "_" << blockHeight
             << "_" << blockDepth;
  return keyBuilder.str();
}

template<typename Dtype>
void interleaveMatrix(
         Dtype* mem_dst, const Dtype *mem,
         int r, int c, int interleavedRows, int nonInterleavedRows,
         int blockWidth, int rowAlignment ) {
  CHECK_EQ(interleavedRows % 2, 0) <<
      "interleaveMatrix only supports even values for interleavedRows.";

  size_t memSize = r * c * sizeof(Dtype);
  size_t dstSize = memSize *
            (interleavedRows + nonInterleavedRows * 2) /
            (interleavedRows + nonInterleavedRows);
  memset(mem_dst, 0, dstSize);    // NOLINT

  const int xStride = blockWidth;
  const int yStride = c * 2;
  const Dtype *pSrc = mem;
  Dtype* pDst = mem_dst;
  for (int y = 0; y < r;) {
    for (int rows = 0; rows < interleavedRows; rows += 2) {
      if ( y >= r ) break;
      if ((c % xStride) == 0) {
        for (int x = 0; x < c / xStride; x++) {
          memcpy( pDst + x * xStride * 2,                         // NOLINT
                  pSrc + x * xStride,     xStride * sizeof(Dtype));
          memcpy( pDst + x * xStride * 2 + xStride,               // NOLINT
                  pSrc + x * xStride + c, xStride * sizeof(Dtype));
        }
      } else {
        const int count = c / xStride;
        int x = 0;
        for (; x < count - 1; x++) {
          memcpy(pDst + x * xStride * 2,                          // NOLINT
                 pSrc + x * xStride, xStride * sizeof(Dtype));
          memcpy(pDst + x * xStride * 2 + xStride,                // NOLINT
                 pSrc + x * xStride + c, xStride * sizeof(Dtype));
        }
        memcpy(pDst + x * xStride * 2,                            // NOLINT
               pSrc + x * xStride, xStride * sizeof(Dtype));
      }
      pSrc += yStride;
      pDst += yStride;
      y += 2;
    }

    for (int rows = 0; rows < nonInterleavedRows; rows++) {
      if (y >= r) break;
      const int stride = rowAlignment;
      int remaining = c;
      for (int x = 0; x < c; x += stride) {
        if (remaining >= stride) {
          memcpy( pDst + x * 2, pSrc + x, stride * sizeof(Dtype));    // NOLINT
          remaining -=stride;
        } else {
          memcpy(pDst + x * 2, pSrc + x, remaining * sizeof(Dtype));  // NOLINT
        }
      }
      pSrc += yStride / 2;
      pDst += yStride;
      y++;
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::swizzleWeights(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp swizzled_factor,
    bool interleave) {

  // Simply skip the weight swizzle if we already got a swizzled_weights_
  // in test phase and not in auto tuning
  // This requires we always call convolve again with the winner configuration
  // during the auto tuning stage.
  if (tuned_ &&
      swizzled_weights_ != NULL &&
      this->phase_ == TEST)
    return;

  swizzled_weights_ = swizzled_weights_blob_.mutable_gpu_data();

  if (!interleave) {
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();
    viennacl::ocl::kernel &oclk_copy_weight = program.get_kernel(
        CL_KERNEL_SELECT("copyWeightsSwizzled"));
    cl_uint argIdx = 0;

    int_tp channels = this->channels_ / this->group_;
    oclk_copy_weight.arg(argIdx++, WrapHandle((cl_mem) weight, &ctx));
    oclk_copy_weight.arg(argIdx++, WrapHandle((cl_mem) swizzled_weights_,
                         &ctx));
    oclk_copy_weight.arg(argIdx++, kernel_w_);
    oclk_copy_weight.arg(argIdx++, kernel_h_);
    oclk_copy_weight.arg(argIdx++, channels);
    oclk_copy_weight.arg(argIdx++, this->num_output_);
    oclk_copy_weight.arg(argIdx++, swizzled_factor);
    const size_t global_work_size_Copy[3] = {
        (size_t) (ALIGN(this->num_output_, swizzled_factor)
        * channels * kernel_w_ * kernel_h_), 1, 1 };

    OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     oclk_copy_weight.handle().get(), 3, NULL,
                                     global_work_size_Copy, NULL, 0, NULL,
                                     NULL));
  } else {
    Dtype *cpu_swizzled_weight = swizzled_weights_blob_.mutable_cpu_data();
    int interleavedRows = (kernel_w_ / 2) * 2;
    int nonInterleavedRows = kernel_w_ % 2;
    int blockWidth = swizzled_factor;  // should equal to SIMD size.
    int rowAlignment = 32;
    size_t interleaved_filter_size = M_ * kernel_w_ * kernel_h_ *
                                     this->channels_ * sizeof(Dtype);
    Dtype * tmpSwizzledWeight = static_cast<Dtype*>(
                                  malloc(interleaved_filter_size));
    CHECK_EQ(tmpSwizzledWeight != NULL, true)
      << "Failed to allocate temporary swizzled weight";
    for ( int od = 0; od < M_; od++)
      for ( int id = 0; id < this->channels_; id++)
        for ( int r = 0; r < kernel_h_; r++)
          for ( int c = 0; c < kernel_w_; c++)
            tmpSwizzledWeight[((id * kernel_h_ + r)
                * kernel_w_ + c) * M_ + od]
                = weight_cpu[((od * this->channels_ + id)
                * kernel_h_ + r) * kernel_w_ + c ];
    interleaveMatrix(cpu_swizzled_weight, tmpSwizzledWeight,
              kernel_w_ * kernel_h_ * this->channels_, M_,
              interleavedRows, nonInterleavedRows, blockWidth, rowAlignment);
    free(tmpSwizzledWeight);
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::calculate_global_size(int_tp batch,
                                  int_tp* wio,  // work item output size
                                  size_t* lSize,  // local size
                                  size_t* gSize) {  // global size
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  gSize[0] = ceil(
      (fmax(static_cast<float>(output_w_) / wio[0], 1.0)) / lSize[0])
      * lSize[0];
  gSize[1] = ceil(
      (fmax(static_cast<float>(output_h_) / wio[1], 1.0)) / lSize[1])
      * lSize[1];
  gSize[2] = ceil(
      static_cast<float>((ceil(static_cast<float>(M_) * batch / wio[2])))
          / lSize[2]) * lSize[2];
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::create_basic_kernel(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  // Standard spatial setup is done here
  std::stringstream keyBuilder;
  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  std::string kernelDef = "MULTI";
  std::string kernelUKey = generate_specific_key(4, blockWidth, blockHeight,
                                                 blockDepth);
  int_tp workItemOutput[3];
  workItemOutput[0] = 1;
  workItemOutput[1] = 1;
  workItemOutput[2] = 1;

  kernel_name_ = "BASIC_";
  kernel_name_ += kernelUKey.c_str();

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D KERNELSIZE="
                << kernel_w_ * kernel_h_ << " -D KERNEL_W=" << kernel_w_
                << " -D KERNEL_H=" << kernel_h_ << " -D CHANNELS="
                << this->channels_ / this->group_
                << " -D STRIDE_H=" << stride_h_
                << " -DDILATION_X=" << dilation_w_
                << " -DDILATION_Y=" << dilation_h_
                << " -D STRIDE_W=" << stride_w_ << " -D APPLY_BIAS="
                << this->bias_term_ << " -D OUTPUT_Z=" << M_
                << " -D XPAR=" << workItemOutput[0] << " -D YPAR="
                << workItemOutput[1] << " -D ZPAR=" << workItemOutput[2]
                << " -D " << kernelDef.c_str() << " -D CFMultiNoPadding="
                << kernel_name_;

  if (IsFusedWithEltwiseReLU()) {
    optionsString << " -DFUSED_CONV_ELTWISE=1";
  }

  if (IsFusedWithReLU()) {
    optionsString << " -DFUSED_CONV_RELU=1";
  }

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  if (IsBeignet(&ctx))
    optionsString << " -D__BEIGNET__";
  string options = optionsString.str();
  try {
    submit_conv_spatial_program<Dtype>(&ctx, kernel_name_, options);
  } catch (std::exception& e) {
    dbgPrint(std::cout << "Basic kernel generation failed" << std::endl);
    return false;
  }

  size_t localSize[3] = { 1, 1, 1 };
  size_t globalSize[3];
  calculate_global_size(1, workItemOutput, localSize, globalSize);

  kernelQueue.push_back(
      new kernelConfig(kernel_name_, globalSize, localSize, workItemOutput,
                       false, false, true, 4));

  return true;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::setBufferKernelArg(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    viennacl::ocl::kernel *kernel,
    const cl_uint &argIdx,
    viennacl::ocl::context *ctx,
    cl_mem buffer, size_t offset,
    size_t size, bool readOnly,
    bool preserved) {

  if (offset == 0) {
    kernel->arg(argIdx, WrapHandle((cl_mem) buffer, ctx));
    return;
  }

  if (preserved &&
    subBufferMap.find(std::make_tuple(buffer, offset, size))
      != subBufferMap.end()) {
    kernel->arg(argIdx,
      WrapHandle(subBufferMap.find
                   (std::make_tuple(buffer, offset, size))->second, ctx));
    return;
  }
  cl_buffer_region region;
  region.origin = offset * sizeof(Dtype);
  region.size = size * sizeof(Dtype);
  cl_mem_flags memFlags = readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  cl_int error;
  cl_mem sub_buffer = clCreateSubBuffer(buffer, memFlags,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region, &error);
  if (error != CL_SUCCESS) {
    dbgPrint(std::cout << "Failed to create sub buffer ("
                        << error << ")." << std::endl);
    throw(error);
  }
  kernel->arg(argIdx, WrapHandle(sub_buffer, ctx));
  if (preserved)
    subBufferMap.insert(std::make_pair(std::make_tuple(buffer, offset, size),
                        sub_buffer));
  else
    tmpSubBuffers.push_back(sub_buffer);
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::cleanTmpSubBuffers(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (auto &buffer : tmpSubBuffers)
    clReleaseMemObject(buffer);
  tmpSubBuffers.clear();
}

template<typename Dtype>
cl_int ConvolutionLayerSpatial<Dtype>::convolve(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  viennacl::ocl::program & program = ctx.get_program(config->kernelName);
  viennacl::ocl::kernel &kernel = program.get_kernel(config->kernelName);
  cl_int err = CL_SUCCESS;
  if (config->kernelType == 2) {
    swizzleWeights(bottom, top, config->workItem_output[2], false);
    size_t total_bottom_size = this->bottom_dim_ * numImages;
    size_t total_kernel_size = kernel_h_ * kernel_w_ * this->channels_ * M_;
    size_t total_bias_size = M_ * this->group_;
    size_t total_top_size = this->top_dim_ * numImages;
    for (int_tp g = 0; g < this->group_; ++g) {
      bias_offset_ = M_ * g;
      int_tp image_offset = width_ * height_ *
                            (this->channels_ / this->group_) * g;
      int_tp output_image_offset = output_w_ * output_h_ * M_ * g;

      int_tp kernel_offset = kernel_h_ * kernel_w_
                             * (this->channels_ / this->group_) * M_ * g;
      cl_uint argIdx = 0;
      if (IsFusedWithEltwiseReLU())
        kernel.arg(argIdx++, WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
      if (IsFusedWithReLU())
        kernel.arg(argIdx++, fixup_arg_type(negative_slope_));

      try {
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bottom_data,
                           image_offset,
                           total_bottom_size - image_offset,
                           true, false);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) swizzled_weights_,
                           kernel_offset,
                           total_kernel_size - kernel_offset,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bias_,
                           bias_offset_,
                           total_bias_size - bias_offset_,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) top_data,
                           output_image_offset,
                           total_top_size - output_image_offset,
                           false, false);
      } catch (int e) {
        err = e;
      }

      if (err == CL_SUCCESS) {
        kernel.arg(argIdx++, (uint16_t)width_);
        kernel.arg(argIdx++, (uint16_t)height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
        const int_tp output_block_w = config->workItem_output[0];
        const int_tp output_block_h = config->workItem_output[1];
        size_t global_size[3] = { (size_t) (output_w_ + output_block_w - 1)
             / output_block_w, (size_t) (output_h_ + output_block_h - 1)
             / output_block_h, (size_t) config->global_work_size[2]};

        err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     kernel.handle().get(), 3,
                                     NULL,
                                     global_size,
                                     config->local_work_size, 0, NULL,
                                     NULL);
      }
      if (err != CL_SUCCESS)
        break;
    }

    if (this->group_ > 1) {
      cleanTmpSubBuffers(bottom, top);
    }
    if (err != CL_SUCCESS)
      return err;
  } else if (config->kernelType == 5) {
    swizzleWeights(bottom, top, config->workItem_output[1], true);
    size_t total_bottom_size = this->bottom_dim_ * numImages;
    size_t total_kernel_size = kernel_h_ * kernel_w_ * this->channels_ * M_;
    size_t total_bias_size = M_ * this->group_;
    size_t total_top_size = this->top_dim_ * numImages;
    for (int_tp g = 0; g < this->group_; ++g) {
      bias_offset_ = M_ * g;
      int_tp image_offset = width_ * height_ *
                            (this->channels_ / this->group_) * g;
      int_tp output_image_offset = output_w_ * output_h_ * M_ * g;

      cl_uint argIdx = 0;
      if (IsFusedWithEltwiseReLU())
        kernel.arg(argIdx++, WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
      if (IsFusedWithReLU())
        kernel.arg(argIdx++, fixup_arg_type(negative_slope_));

      int_tp kernel_offset = kernel_h_ * kernel_w_
                             * (this->channels_ / this->group_) * M_ * g;
      try {
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bottom_data,
                           image_offset,
                           total_bottom_size - image_offset,
                           true, false);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) swizzled_weights_,
                           kernel_offset,
                           total_kernel_size - kernel_offset,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) bias_,
                           bias_offset_,
                           total_bias_size - bias_offset_,
                           true, true);
        setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                           (cl_mem) top_data,
                           output_image_offset,
                           total_top_size - output_image_offset,
                           false, false);
      } catch (int e) {
        err = e;
      }

      if (err == CL_SUCCESS) {
        kernel.arg(argIdx++, (uint16_t)width_);
        kernel.arg(argIdx++, (uint16_t)height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
        int out_pitch_y = output_w_ * output_h_;
        int out_pitch_z = out_pitch_y * M_;
        int aligned_input_size = height_ * width_ *
                                 this->channels_ / this->group_;
        int slice_pitch = width_ * height_;
        kernel.arg(argIdx++, (uint32_t)out_pitch_y);
        kernel.arg(argIdx++, (uint32_t)out_pitch_z);
        kernel.arg(argIdx++, (uint32_t)aligned_input_size);
        kernel.arg(argIdx++, (uint32_t)slice_pitch);

        int blockM = config->workItem_output[0];
        int blockK = config->workItem_output[1];
        int blockN = config->workItem_output[2];
        int_tp alignedFilterWidth = ALIGN(M_, blockN);
        int_tp alignedExpandHeight = ALIGN(output_w_ * output_h_, blockM);
        int_tp globalWorkSizeDX = blockN;
        int_tp globalWorkSizeDY = blockM;
        size_t sgemm_m = alignedExpandHeight;
        size_t sgemm_n = alignedFilterWidth;
        size_t gx = (size_t) ceil( (float) sgemm_n /
                                   (float) globalWorkSizeDX );
        size_t gy = (size_t) ceil( (float) sgemm_m /
                                   (float) globalWorkSizeDY );
        gy = ALIGN(gy, blockK);
        size_t global_size[3] = { gx, gy, config->global_work_size[2] };

        viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(this->device_->id());
        err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                     kernel.handle().get(), 3,
                                     NULL,
                                     global_size,
                                     config->local_work_size, 0, NULL,
                                     NULL);
        OCL_CHECK(err);
      }
      if (err != CL_SUCCESS)
        break;
    }

    if (this->group_ > 1) {
      cleanTmpSubBuffers(bottom, top);
    }
    if (err != CL_SUCCESS)
      return err;
  } else {
    for (int_tp n = 0; n < numImages; ++n) {
      for (int_tp g = 0; g < this->group_; ++g) {
        bias_offset_ = M_ * g;
        int_tp image_offset = n * this->bottom_dim_
            + width_ * height_ * (this->channels_ / this->group_) * g;
        int_tp output_image_offset = n * this->top_dim_
            + output_w_ * output_h_ * M_ * g;

        cl_uint argIdx = 0;
        if (IsFusedWithEltwiseReLU())
          kernel.arg(argIdx++,
                     WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
        if (IsFusedWithReLU())
          kernel.arg(argIdx++, fixup_arg_type(negative_slope_));

        int_tp kernel_offset = kernel_h_ * kernel_w_ *
                               (this->channels_ / this->group_) * M_
                               * g;

        kernel.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
        kernel.arg(argIdx++, image_offset);
        kernel.arg(argIdx++, WrapHandle((cl_mem) weight, &ctx));
        kernel.arg(argIdx++, kernel_offset);
        kernel.arg(argIdx++, WrapHandle((cl_mem) bias_, &ctx));
        kernel.arg(argIdx++, bias_offset_);
        kernel.arg(argIdx++, WrapHandle((cl_mem) top_data, &ctx));
        kernel.arg(argIdx++, output_image_offset);
        kernel.arg(argIdx++, (uint16_t)width_);
        kernel.arg(argIdx++, (uint16_t)height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
        kernel.arg(argIdx++, (uint16_t)pad_w_);
        kernel.arg(argIdx++, (uint16_t)pad_h_);

        int_tp workItemOutput[3] = { 1, 1, 1 };
        size_t localSize[3] = { 1, 1, 1 };
        size_t globalSize[3];
        calculate_global_size(1, workItemOutput, localSize, globalSize);
        if (config->use_null_local) {
          err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       kernel.handle().get(), 3,
                                       NULL,
                                       config->global_work_size, NULL, 0, NULL,
                                       NULL);
        } else {
          err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       kernel.handle().get(), 3,
                                       NULL,
                                       config->global_work_size,
                                       config->local_work_size, 0, NULL,
                                       NULL);
        }

        if (err != CL_SUCCESS)
          return err;
      }
    }
  }

  return err;
}

template<typename Dtype>
float ConvolutionLayerSpatial<Dtype>::timed_convolve(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  // warm up.
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  bool saved_tuned = tuned_;
  tuned_ = false;
  convolve(bottom, top, index, this->num_, config);
  Timer timer;
  timer.initted();
  timer.Start();
  cl_int err;
  dbgPrint(std::cout << "Bechmarking kernel: " << config->kernelName
           << std::endl);
  tuned_ = true;
  int loop_cnt = 4;
  for (int i = 0; i < loop_cnt; i++) {
    err = convolve(bottom, top, index, this->num_, config);
    if (err != CL_SUCCESS)
      break;
  }
  tuned_ = saved_tuned;
  timer.Stop();
  if (err != CL_SUCCESS) {
    config->tested = true;
    config->verified = false;
    dbgPrint(std::cout << "convolution failed with error code "
             << err << std::endl);
    return 1e5;
  }

  float elapsedTime = timer.MilliSeconds() / loop_cnt;
#ifdef dbg
  double out_w = output_w_;
  double out_h = output_h_;
  double out_z = M_;
  double k_w = kernel_w_;
  double k_h = kernel_h_;
  double k_z = this->channels_;
  double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z) * this->num_;
  std::cout << "\tEstimated Gflops:" << ((totalFlops/1000)/1000)/1000
  << std::endl;
  std::cout << "\tEstimated GFLOPS/S: " <<
  (((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime) << std::endl;
#if 0
  std::cout << "Estimated utilization: " <<
  ((((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime))/880.0
  << std::endl;
#endif
#endif
  return elapsedTime;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::verify_result(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp index,
    int_tp numImages,
    const Blob<Dtype> &verify_blob,
    kernelConfig* config) {

  uint_tp verificationFail = 0;

  if (config->verified)
    return true;
  else if (config->tested)
    return false;
  greentea_memset(this->device_->id(),
                  top[index]->count() * sizeof(Dtype),
                  0xff,
                  (cl_mem)top[index]->mutable_gpu_data(),
                  0);
  config->executionTime = timed_convolve(bottom, top, index, numImages,
                                         config);
  // Currently we can't do verification when conv is fused because the results
  // won't match the results of forward_gpu_gemm. Need more work to fix it.
  // FP16 verification may fail due to the natrue accuracy lost between FP16 and FP32.
  if (IsFused() || std::is_same<Dtype, half_float::half>::value)
    return true;
  const Dtype *verify_data = verify_blob.cpu_data();
  const Dtype *data = top[index]->cpu_data();
  Dtype err_factor = 1;
  if (std::is_same<Dtype, half_float::half>::value)
    err_factor = 8;

  for (int_tp n = 0; n < numImages; ++n) {
    for (int_tp g = 0; g < this->group_; ++g) {
      int_tp output_image_offset = n * this->top_dim_
          + output_w_ * output_h_ * M_ * g;
      for (int out_ch = 0; out_ch < M_ && !verificationFail; out_ch++)
        for (int h = 0; h < output_h_ && !verificationFail; h++)
          for (int w = 0; w < output_w_; w++) {
            size_t offset = output_image_offset + out_ch * output_w_ * output_h_
                            + h * output_w_ + w;
            if (fabs(data[offset] - verify_data[offset]) >
                       0.1 * fabs(verify_data[offset] * err_factor) &&
                !(fabs(verify_data[offset]) < 1e-3 * err_factor
                  && fabs(data[offset] - verify_data[offset]) <
                     1e-4 * err_factor)) {
              dbgPrint(printf("test verification failed @ image %d group %d"
                              "out_ch %d h %d w %d got %G expected %G\n",
                      n, g, out_ch, h, w,
                      static_cast<float>(data[offset]),
                      static_cast<float>(verify_data[offset])));
              verificationFail = 1;
              break;
            }
          }
      if (verificationFail)
        return false;
    }
  }
  return true;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::create_gemm_like_conv_kernel(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp blockM,
    int_tp blockK,
    int_tp blockN) {
  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  std::string kernelUKey = generate_specific_key(5, blockM, blockK,
                                                 blockN);
  int_tp workItemOutput[3] = { blockM, blockK, blockN };

  int_tp simd_size = blockK;
  int_tp num_batches = this->num_;
  int_tp globalWorkSizeDX = blockN;
  int_tp globalWorkSizeDY = blockM;

  kernel_name_ = "U_GEMM_LIKE_CONV_";
  kernel_name_ += kernelUKey.c_str();
  if (blockK == 8)
    kernel_name_ += "_SIMD8";
  else
    kernel_name_ += "_SIMD16";
  std::stringstream kernelDef;
  kernelDef << "GEMM_LIKE_CONV_" << blockN << "_" << blockM;
  if (blockK == 16)
    kernelDef << "_SIMD16";

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D " << kernelDef.str()
                << " -D Conv_Interleaved=" << kernel_name_.c_str();

  optionsString <<
        " -cl-mad-enable" <<
        " -DKERNEL_WIDTH=" << kernel_w_ <<
        " -DKERNEL_HEIGHT=" << kernel_h_ <<
        " -DSTRIDE_X=" << stride_w_ <<
        " -DSTRIDE_Y=" << stride_h_ <<
        " -DDILATION_X=" << dilation_w_ <<
        " -DDILATION_Y=" << dilation_h_ <<
        " -DINPUT_DEPTH=" << this->channels_ <<
        " -DWIDTH1=" << M_ <<
        " -DOUT_PADDING_LEFT=" << 0 <<
        " -DOUT_PADDING_HEIGHT=" << 0 <<
        " -DOUT_DEPTH=" << M_ <<
        " -DNUM_BATCHES=" << this->num_ <<
        " -DDY=" << globalWorkSizeDY <<
        " -DDX=" << globalWorkSizeDX <<
        " -DKERNEL_WIDTH_DIV2=" << kernel_w_ / 2 <<
        " -DKERNEL_SLICE_DIV2=" << (kernel_w_ * kernel_h_) / 2 <<
        " -DTILE_N_LAST=" << M_ % 32 <<
        " -DTILE_N_LAST_DIV8=" << (M_ % 32) / 8;

  if (IsFusedWithEltwiseReLU()) {
    optionsString << " -DFUSED_CONV_ELTWISE=1";
  }

  if (IsFusedWithReLU()) {
    optionsString << " -DFUSED_CONV_RELU=1";
  }
  optionsString << " -DINPUT_PAD_W=" << pad_w_ << " -DINPUT_PAD_H=" << pad_h_;

  size_t gz = num_batches;
  size_t global_size[3] = { 0, 0, gz };

  size_t local_size[3] = { 1, static_cast<size_t>(simd_size), 1 };
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  if (IsBeignet(&ctx))
    optionsString << " -D__BEIGNET__";
  string options = optionsString.str();

  viennacl::ocl::program & program = submit_conv_spatial_program<Dtype>(&ctx,
                                                                 kernel_name_,
                                                                 options);
  size_t workgroupSize_used;
  viennacl::ocl::kernel & kernel = program.get_kernel(kernel_name_);
  cl_int err = clGetKernelWorkGroupInfo(
      kernel.handle().get(), viennacl::ocl::current_device().id(),
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(size_t), &workgroupSize_used,
      NULL);

  if (workgroupSize_used != simd_size) {
    ctx.delete_program(kernel_name_);
    return false;
  }

  if (err == CL_SUCCESS || err == true) {
    kernelQueue.push_back(
        new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                         false, true, false, 5));
    return true;
  } else {
    ctx.delete_program(kernel_name_);
    return false;
  }
}


template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::setup_IDLF(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp blockWidth,
    int_tp blockHeight,
    int_tp simd_size) {
  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  const int_tp blockDepth = 1;
  std::string kernelUKey = generate_specific_key(2, blockWidth, blockHeight,
                                                 blockDepth);
  int_tp workItemOutput[3] = { blockWidth, blockHeight, simd_size };
  const int_tp num_output_maps = M_;
  int_tp output_block_width = blockWidth;
  int_tp output_block_height = blockHeight;
  int_tp num_batches = this->num_;

  kernel_name_ = "IDLF_";
  kernel_name_ += kernelUKey.c_str();

  if (simd_size == 16)
    kernel_name_ += "_SIMD16";
  else
    kernel_name_ += "_SIMD8";

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D IDLF"
                << " -D convolve_simd="
                << kernel_name_;

  size_t global_size[3] = { 0, 0,
                (size_t) num_batches * ALIGN(num_output_maps, simd_size) };

  size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };
  int tile_x = (((output_block_width - 1) * stride_w_
               + kernel_w_ * dilation_w_) + 3) & ~3;
  int tile_y = (output_block_height -1) * stride_h_ + kernel_h_ * dilation_h_;
  int tile_y_stride = (4 * simd_size) / tile_x;
  int invec_size = (tile_y + tile_y_stride - 1) / tile_y_stride;

  optionsString << " -D SIMD_SIZE=" << simd_size
                << " -D filter_qualifier=__global" << " -D OUT_BLOCK_WIDTH="
                << output_block_width << " -D OUT_BLOCK_HEIGHT="
                << output_block_height
                << " -D INPUT_DEPTH=" << this->channels_ / this->group_
                << " -DTOTAL_INPUT_DEPTH_SIZE=" << this->channels_
                << " -DTOTAL_OUTPUT_DEPTH=" << this->num_output_
                << " -DINPUT_START_X=" << 0 << " -DINPUT_START_Y=" << 0
                << " -DINPUT_START_Z=" << 0
                << " -DKERNEL_WIDTH=" << kernel_w_
                << " -DKERNEL_HEIGHT=" << kernel_h_
                << " -DNUM_FILTERS=" << M_ << " -DSTRIDEX=" << stride_w_
                << " -DSTRIDEY=" << stride_h_ << " -DDILATION_X=" << dilation_w_
                << " -DDILATION_Y=" << dilation_h_
                << " -DOWPAD=" << 0 << " -DOHPAD="
                << 0 << " -DOUT_BUFF_OFFSET=" << 0
                << " -DTILE_X=" << tile_x
                << " -DTILE_Y=" << tile_y
                << " -DTILE_Y_STRIDE=" << tile_y_stride
                << " -DINVEC_SIZE=" << invec_size
                << " -DALIGNED_NUM_FILTERS=" << ALIGN(M_, simd_size);

  optionsString << " -DINPUT_PAD_W=" << pad_w_ << " -DINPUT_PAD_H=" << pad_h_;

  if (IsFusedWithEltwiseReLU()) {
    optionsString << " -DFUSED_CONV_ELTWISE=1";
  }

  if (IsFusedWithReLU()) {
    optionsString << " -DFUSED_CONV_RELU=1";
  }

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  if (IsBeignet(&ctx))
    optionsString << " -D__BEIGNET__";
  string options = optionsString.str();
  viennacl::ocl::program & program = submit_conv_spatial_program<Dtype>(&ctx,
                                                                 kernel_name_,
                                                                 options);

  // ClKernel kernel;
  size_t workgroupSize_used;
  viennacl::ocl::kernel & kernel = program.get_kernel(kernel_name_);
  cl_int err = clGetKernelWorkGroupInfo(
      kernel.handle().get(), viennacl::ocl::current_device().id(),
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(size_t), &workgroupSize_used,
      NULL);

  if (workgroupSize_used != simd_size) {
    ctx.delete_program(kernel_name_);
    return false;
  }

  if (err == CL_SUCCESS || err == true) {
    kernelQueue.push_back(
        new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                         false, true, false, 2));
    return true;
  } else {
    ctx.delete_program(kernel_name_);
    return false;
  }
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::tune_local_size(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    kernelConfig* config) {
  if (config->use_null_local || !config->autoTune)
    return true;

  float fastestTime = 999999990000000000000000000.0f;
  uint_tp multiplier = 4;
  uint_tp localSize[3] = { 1, 1, 1 };

  int_tp skip = 0;
  Timer timer;
  timer.initted();
  bool allFailed = true;
  for (int_tp z = 0; z <= 16; z++) {
    for (int_tp y = 0; y <= 16; y++) {
      for (int_tp x = 1; x <= 16; x++) {
        timer.Start();
        skip = 0;

        if (config->autoTune) {
          config->local_work_size[0] =
              (multiplier * x == 0) ? 1 : multiplier * x;
          config->local_work_size[1] =
              (multiplier * y == 0) ? 1 : multiplier * y;
          config->local_work_size[2] =
              (multiplier * z == 0) ? 1 : multiplier * z;

          calculate_global_size(1, config->workItem_output,
                                config->local_work_size,
                                config->global_work_size);
        }
        if (config->workItem_output[2] *
            config->global_work_size[2] != M_)
          break;

        if (config->swizzle_weights)
          z = 32;

        int_tp err = 0;
        err = convolve(bottom, top, 0, 1, config);

        if (err != CL_SUCCESS)
          skip = 1;

        if (skip) {
          timer.Stop();
          break;
        }
        timer.Stop();
        allFailed = false;
        float elapsedTime = timer.MilliSeconds();

        if (elapsedTime < fastestTime) {
          fastestTime = elapsedTime;
          localSize[0] = config->local_work_size[0];
          localSize[1] = config->local_work_size[1];
          localSize[2] = config->local_work_size[2];
        }
      }
    }
  }
  if (allFailed) {
    // 1,1,1 is never a good local size and no need to test at all.
    dbgPrint(std::cout << "Can't find good local size for "
                       << config->kernelName << std::endl);
    return false;
  }

  dbgPrint(std::cout << "Best local size[" << localSize[0] << "][" <<
      localSize[1] << "]["<< localSize[2] << "]: " << fastestTime <<
      " Kernel_h: " << kernel_h_ << " kernel_w_: " << kernel_w_ <<
      " stride_w: " << stride_w_ << " pad_w_: " << pad_w_ << std::endl);

  if (config->autoTune) {
    for (int_tp li = 0; li < 3; li++)
      config->local_work_size[li] = localSize[li];

    calculate_global_size(1, config->workItem_output, config->local_work_size,
                          config->global_work_size);
  }
  return true;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::create_convolution_kernel(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp kernelType,
    int_tp blockWidth, int_tp blockHeight,
    int_tp blockDepth) {
  if (kernelType == 2)
    setup_IDLF(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == 4)
    create_basic_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == 5)
    create_gemm_like_conv_kernel(
        bottom, top, blockWidth, blockHeight, blockDepth);
  else
    assert(0);
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::setup_convolution(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    const Blob<Dtype> &verify_blob) {
  // Initializes unique kernel ID
  kernel_uid_ = 0;
  std::string viennacl_cache_path;

  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                    (this->device_->id());
  if (std::getenv("VIENNACL_CACHE_PATH")) {
    viennacl_cache_path = std::getenv("VIENNACL_CACHE_PATH");
    // Disable viennacl cache mechanism during tuning phase.
    ctx.cache_path("");
  } else {
    viennacl_cache_path = "";
  }

  if (this->device_->CheckCapability("cl_intel_subgroups")) {
    /* IDLF kernels are using Intel specific extension which make
       them intel only. */
    // Generates static key_
    int max_compute_units = ctx.current_device().max_compute_units();
    int kernelCnt = 0;
    if (this->group_ == 1 && ((M_ % 8 == 0) && (M_ % 32 != 24))) {
      create_convolution_kernel(bottom, top, 5, 1, 8, 32);
      create_convolution_kernel(bottom, top, 5, 2, 8, 32);
      if ((kernel_w_ < 4 || (!std::is_same<Dtype, float>::value))
          && M_ % 32 == 0)
        create_convolution_kernel(bottom, top, 5, 1, 16, 32);
      if (kernel_w_ < 4 && (!std::is_same<Dtype, float>::value))
        create_convolution_kernel(bottom, top, 5, 2, 16, 32);
    }

    for (int simd_size = 8; simd_size <= 16; simd_size += 8) {
      if (simd_size == 8
          && !((this->group_ == 1 || M_ % 8 == 0)))
        continue;
      if (simd_size == 16
          && !(this->group_ == 1 || M_ % 16 == 0))
        continue;
      int width_max, height_max, block_size_max;
      if (simd_size == 8) {
        width_max = 16;
        height_max = 16;
        block_size_max = 48;
      } else {
        width_max = 14;
        height_max = 14;
        block_size_max = 32;
      }
      for (uint32_t width = width_max; width > 0; width--) {
        int candidate = 0;
        if (width > output_w_)
          continue;
        for (uint32_t height = height_max; height > 0; height--) {
          if (width * height > block_size_max || height > output_h_)
            continue;
          // Only when the work items count is less than the device
          // max work items or the M_ is less than 16, we will tune
          // for simd 8.
          if (simd_size == 8
              && M_ >= 16
              && ((this->num_ * M_ * output_w_ * output_h_ /
                   static_cast<float>(width * height))
                 >= max_compute_units * 7 * 16))
            continue;
          int tile_x = (kernel_w_ * dilation_w_
                       + (width - 1) * stride_w_ + 3) & ~3;
          int tile_y = kernel_h_ * dilation_h_ + (height - 1) * stride_h_;
          if (tile_x > (4 * simd_size))
            continue;
          int tile_y_stride = (4 * simd_size) / tile_x;

          if ((tile_y + tile_y_stride - 1) / tile_y_stride < 4) {
            create_convolution_kernel(bottom, top, 2, width, height, simd_size);
            candidate++;
          }
          if (candidate >= 4 && height == 2)
            break;
        }
        kernelCnt += candidate;
        if (kernelCnt >= 12 && width == 2)
          break;
      }
    }
  }
  for (int_tp x = 0; x < kernelQueue.size(); x++) {
    if (tune_local_size(bottom, top, kernelQueue[x])) {
      kernelQueue[x]->executionTime = timed_convolve(bottom, top, bottom_index_,
                                                   this->num_, kernelQueue[x]);
    } else {
      // skip those kernels without a good local size.
      kernelQueue[x]->verified = false;
      kernelQueue[x]->tested = true;
    }
#ifdef TEST_ALL_KERNELS
    if (kernelQueue[x]->tested == false) {
      bool verified = verify_result(bottom, top, bottom_index_, this->num_,
                                      verify_blob, kernelQueue[x]);
      if (verified == false) {
        dbgPrint(std::cout << "Kernel "
                             << kernelQueue[x]->kernelName
                             << " failed verification" << std::endl);
        dbgPrint(std::cout << "kernelQueue[x]->workItem_output[0]: "
                       << kernelQueue[x]->workItem_output[0] << " "
                       << "kernelQueue[x]->workItem_output[1]: "
                       << kernelQueue[x]->workItem_output[1] << " "
                       << "kernelQueue[x]->workItem_output[2]: "
                       << kernelQueue[x]->workItem_output[2] << " "
                       << "kernelQueue[x]->kernelType: "
                       << kernelQueue[x]->kernelType << " "
                       << "kernelQueue[x]->global_work_size[0]: "
                       << kernelQueue[x]->global_work_size[0] << " "
                       << "kernelQueue[x]->global_work_size[1]: "
                       << kernelQueue[x]->global_work_size[1] << " "
                       << "kernelQueue[x]->global_work_size[2]: "
                       << kernelQueue[x]->global_work_size[2] << " "
                       << "kernelQueue[x]->local_work_size[0]: "
                       << kernelQueue[x]->local_work_size[0] << " "
                       << "kernelQueue[x]->local_work_size[1]: "
                       << kernelQueue[x]->local_work_size[1] << " "
                       << "kernelQueue[x]->local_work_size[2]: "
                       << kernelQueue[x]->local_work_size[2] << " "
                       << kernelQueue[x]->swizzle_weights << " "
                       << kernelQueue[x]->use_null_local << std::endl);
      } else {
        dbgPrint(std::cout << "Kernel "
                           << kernelQueue[x]->kernelName
                           << " pass verification" << std::endl);
      }
    }
#endif
  }
  int_tp failures = 0;
  bool verification = false;
  if (kernelQueue.size()) {
    while (failures < kernelQueue.size()) {
      int_tp fastestKernel = -1;
      float fastestTime = 999999990000000000000000000.0f;

      for (int_tp x = 0; x < kernelQueue.size(); x++) {
        if (kernelQueue[x]->executionTime < fastestTime
            && kernelQueue[x]->tested == false) {
          fastestKernel = x;
          fastestTime = kernelQueue[x]->executionTime;
        }
      }
      if (fastestKernel < 0) break;
      // Test fastest kernel
      bool verified = verify_result(bottom, top, bottom_index_, this->num_,
                                    verify_blob, kernelQueue[fastestKernel]);
      if (verified == true) {
        kernelQueue[fastestKernel]->verified = true;
        kernel_index_ = fastestKernel;
        verification = true;
        break;
      } else {
        kernelQueue[fastestKernel]->tested = true;
        dbgPrint(std::cout << "Kernel "
                           << kernelQueue[fastestKernel]->kernelName
                           << " failed verification" << std::endl);
        failures++;
      }
    }
  }
  if (verification) {
    dbgPrint(std::cout << "Kernel <" << kernelQueue[kernel_index_]->kernelName
                       << "> passed verification" << std::endl);
  } else {
    dbgPrint(std::cout << "Verification was not successful, "
                       << "fallback to basic kernel" << std::endl);
    create_basic_kernel(bottom, top, 1, 1, 1);
    kernel_index_ = kernelQueue.size() - 1;
    verification = verify_result(bottom, top, bottom_index_, this->num_,
                                 verify_blob, kernelQueue[kernel_index_]);
    CHECK_EQ(verification, true) << "Basic kernel failed verification."
                                 << std::endl;
  }
  this->bestKernelConfig = kernelQueue[kernel_index_];

  dbgPrint(std::cout << "Convolution Time:"
                     << kernelQueue[kernel_index_]->executionTime << std::endl);

  if (bestKernelConfig->kernelType != 2 && bestKernelConfig->kernelType != 5)
    swizzled_weights_ = NULL;

  for (int_tp x = 0; x < kernelQueue.size(); x++) {
    if (x != kernel_index_) {
      viennacl::ocl::current_context().delete_program(
          kernelQueue[x]->kernelName);
      delete kernelQueue[x];
    }
  }
  kernelQueue.clear();

  tuned_ = true;

  string outputFile;
  outputFile = cache_path_.str() + key_;
  std::ifstream cachedKernel(outputFile.c_str());
  std::ofstream outputKernel;
  outputKernel.open(outputFile.c_str());
  outputKernel << bestKernelConfig->workItem_output[0] << " "
               << bestKernelConfig->workItem_output[1] << " "
               << bestKernelConfig->workItem_output[2] << " "
               << bestKernelConfig->kernelType << " "
               << bestKernelConfig->global_work_size[0] << " "
               << bestKernelConfig->global_work_size[1] << " "
               << bestKernelConfig->global_work_size[2] << " "
               << bestKernelConfig->local_work_size[0] << " "
               << bestKernelConfig->local_work_size[1] << " "
               << bestKernelConfig->local_work_size[2] << " "
               << bestKernelConfig->swizzle_weights << " "
               << 0 << " "  // deprecated
               << bestKernelConfig->use_null_local << " ";
  outputKernel.close();
  ctx.cache_path(viennacl_cache_path);
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  weight = this->blobs_[0]->gpu_data();
  weight_cpu = static_cast<const Dtype*>(this->blobs_[0]->cpu_data());
  if (this->bias_term_)
    bias_ = this->blobs_[1]->gpu_data();

  int bottom_size = bottom.size();
  if (IsFusedWithEltwiseReLU())
    bottom_size = 1;
  for (int_tp i = 0; i < bottom_size; ++i) {
    bottom_index_ = i;
    bottom_data = bottom[i]->gpu_data();
    top_data = top[i]->mutable_gpu_data();
    weight_offset = M_ * K_;
    col_offset = K_ * N_;
    top_offset = M_ * N_;
    bias_offset_ = 0;

    if (!tuned_) {
      Blob<Dtype> verify_blob;
      verify_blob.ReshapeLike(*top[i]);
      Dtype *verify_data = verify_blob.mutable_gpu_data();
      const Dtype *weight_gpu_data = this->blobs_[0]->gpu_data();
      const Dtype *bottom_gpu_data = bottom[i]->gpu_data();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->forward_gpu_gemm(bottom_gpu_data, n * this->bottom_dim_,
                               weight_gpu_data, verify_data,
                               n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(verify_data, n * this->top_dim_, bias);
        }
      }
      generate_key();
      setup_convolution(bottom, top, verify_blob);
      CHECK_EQ(tuned_, true) << "Spatial convolution auto-tuning failed.";
    }

    convolve(bottom, top, i, this->num_, bestKernelConfig);
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, n * this->bottom_dim_,
              top_diff, n * this->top_dim_, weight_diff);
        }
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
        // Multi queue execution, all previous work needs to be done first
        this->device_->FinishQueues();
        for (int_tp n = 0; n < this->num_; ++n) {
          // Multi queue execution, go through work queues
          this->device_->SwitchQueue(n);
          this->backward_gpu_gemm(top_diff, n * this->top_dim_, weight,
                                  bottom_diff, n * this->bottom_dim_);
        }
        // Multi queue execution, finish all queues
        this->device_->FinishQueues();
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::load_cached_kernels(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Generates static key_
  std::string previous_key = key_;
  generate_key();
  int prev_kernel_type = 0;
  if (tuned_) {
    if (key_.compare(previous_key) == 0)
      return;
    tuned_ = false;
    prev_kernel_type = bestKernelConfig->kernelType;
    viennacl::ocl::current_context().
      delete_program(bestKernelConfig->kernelName);
    delete bestKernelConfig;
    bestKernelConfig = NULL;
  }
  // Initializes unique kernel ID
  kernel_uid_ = 0;

  // Find cached kernel configuration
  string outputFile;
  outputFile = cache_path_.str() + key_;
  std::ifstream cachedKernel(outputFile.c_str());
  if (cachedKernel) {
    int_tp x, y, z, type;
    cachedKernel >> x;
    cachedKernel >> y;
    cachedKernel >> z;
    cachedKernel >> type;
    if (type == 2) {
      if (z == 1)
        z = 16;
      CHECK_EQ(z == 16 || z == 8, true) << "invalid SIMD size" << std::endl;
    }
    create_convolution_kernel(bottom, top, type, x, y, z);
    kernel_index_ = kernelQueue.size() - 1;
    if (kernel_index_ == -1) {
      std::cerr << "Failed to get kernel from cached configurations."
                << std::endl;
      std::cerr << "Deleting broken cache file and try tuning again..."
                << std::endl;
      string bakFile = outputFile + ".bak";
      std::rename(outputFile.c_str(), bakFile.c_str());
      return;
    }
    bestKernelConfig = kernelQueue[kernel_index_];
    kernelQueue.clear();
    // As we are using varying image size kernels now, let's skip the
    // cached work group size and local group size here, and we already
    // get correct work/local group size at the create_convolution kernel stage.
    // To not break the previous trained record, for now just skipping them.
    // Will use a totally different cache mechanism in the future.
    size_t foo;  // for deprecated parameters.
    cachedKernel >> foo;
    cachedKernel >> foo;
    cachedKernel >> foo;
    cachedKernel >> bestKernelConfig->local_work_size[0];
    cachedKernel >> bestKernelConfig->local_work_size[1];
    cachedKernel >> bestKernelConfig->local_work_size[2];
    if (bestKernelConfig->kernelType == 1)
      calculate_global_size(1, bestKernelConfig->workItem_output,
                            bestKernelConfig->local_work_size,
                            bestKernelConfig->global_work_size);
    cachedKernel >> bestKernelConfig->swizzle_weights;
    cachedKernel >> foo;
    cachedKernel >> bestKernelConfig->use_null_local;
    tuned_ = true;
    // If kernel type changed to type 2 or 4, we need to reset the swizzled
    // weights pointer to invalidate the previous swizzled weights data.
    if (prev_kernel_type != bestKernelConfig->kernelType &&
        (bestKernelConfig->kernelType == 2 ||
         bestKernelConfig->kernelType == 5))
      swizzled_weights_ = NULL;
  }
  return;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    caffe::Backend backend) {
  if (backend == caffe::BACKEND_OpenCL) {
    load_cached_kernels(bottom, top);
  }
}

#else
template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#endif
INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayerSpatial);
#endif

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayerSpatial);
#endif

INSTANTIATE_CLASS(ConvolutionLayerSpatial);

}  // namespace caffe
#endif  // USE_INTEL_SPATIAL
