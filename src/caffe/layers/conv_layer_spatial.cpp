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
#endif

#include <boost/filesystem.hpp>


namespace caffe {

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int_tp input_dim = this->input_shape(i + 1);
    const int_tp output_dim = (input_dim + 2 * pad_data[i]
        - kernel_shape_data[i]) / stride_data[i] + 1;
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
  M_ = this->num_output_ / this->group_;
  K_ = this->channels_ * kernel_h_ * kernel_w_ / this->group_;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  height_ = bottom[0]->shape(this->channel_axis_ + 1);
  width_ = bottom[0]->shape(this->channel_axis_ + 2);
  output_h_ = (height_ + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  output_w_ = (width_ + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
  padded_width_ = width_ + 2 * pad_w_;
  padded_height_ = height_ + 2 * pad_h_;

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
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.
  spatial_col_buffer_.Reshape(this->num_, this->channels_, height_ + 2 * pad_h_,
                      width_ + 2 * pad_w_);
  swizzled_weights_.Reshape(this->num_output_, this->channels_,
                            kernel_h_ + 2 * pad_h_, kernel_w_ + 2 * pad_w_);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (this->bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  if (std::is_same<Dtype, float>::value) {
    this->num_ = bottom[0]->count(0, this->channel_axis_);
    SetUp(bottom, top, Caffe::GetDefaultDevice()->backend());
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
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

//  #define dbg
#ifdef dbg
#define dbgPrint(x) (x)
#else
#define dbgPrint(x)
#endif

#define CACHE_DIRECTORY ".spatialkernels/"

// For large enough input size, we do not need to tune kernels for different
// size. The reason is with large input size, there will be enough work items
// to feed al the EUs.
#define ADJUST_INPUT_IMAGE_SIZE(x) ((x) > 16 * 16 ? 256 : (x))

template<>
void ConvolutionLayerSpatial<float>::generate_key() {
  std::stringstream keyBuilder;
  int adjusted_width;
  int adjusted_height;
  adjusted_width = ADJUST_INPUT_IMAGE_SIZE(padded_width_);
  adjusted_height = ADJUST_INPUT_IMAGE_SIZE(padded_height_);
  keyBuilder << kernel_w_ << "_" << kernel_h_ << "_" << channels_ << "_"
             << group_ << "_" << stride_h_ << "_" << stride_w_ << "_"
             << bias_term_ << "_" << adjusted_width << "_" << adjusted_height
             << "_" << num_ << "_" << group_ << "_" << M_;
  key_ = keyBuilder.str();
}

template<>
std::string ConvolutionLayerSpatial<float>::generate_unique_key() {
  std::stringstream keyBuilder;
  keyBuilder << key_ << "" << kernel_uid_;
  kernel_uid_++;
  return keyBuilder.str();
}

template<>
std::string ConvolutionLayerSpatial<float>::generate_specific_key(
    int_tp type, int_tp blockWidth, int_tp blockHeight, int_tp blockDepth) {
  std::stringstream keyBuilder;
  keyBuilder << key_ << "_" << type << "_" << blockWidth << "_" << blockHeight
             << "_" << blockDepth;
  return keyBuilder.str();
}

template<>
bool ConvolutionLayerSpatial<float>::generate_kernel(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  // Standard spatial setup is done here
  std::string kernelDef = "MULTI";
  std::string stringBuilder;
  std::stringstream optionsString;

  int_tp workItemOutput[3];
  int_tp yDim = blockHeight;
  int_tp zDim = blockDepth;

  std::string kernelUKey = generate_specific_key(1, blockWidth, blockHeight,
                                                 blockDepth);
  std::stringstream multFunctionBuilder;
  workItemOutput[0] = 4;
  workItemOutput[1] = yDim;
  workItemOutput[2] = zDim;

  std::string multiplication_func = "floatDotV4(V1,V2)=(V1.s0123*V2.s0123)";

  if (kernel_w_ <= 11) {
    multFunctionBuilder << "floatDotV4(V1,V2)=" << "(";
    for (int_tp kw = 0; kw < kernel_w_; kw++) {
      multFunctionBuilder << "V1.s" << std::hex << kw << kw + 1 * stride_w_
                          << kw + 2 * stride_w_ << kw + 3 * stride_w_
                          << std::dec;
      multFunctionBuilder << "*";
      multFunctionBuilder << "V2.s" << std::hex << kw << std::dec;

      if (kw == kernel_w_ - 1)
        multFunctionBuilder << ")";
      else
        multFunctionBuilder << "+";
    }
    multiplication_func = multFunctionBuilder.str();
  }

  int_tp lineSize = kernel_w_ + (workItemOutput[0] - 1) * stride_w_;

  kernel_name_ = "U";
  kernel_name_ += kernelUKey.c_str();
  if (kernel_h_ == 11 && stride_h_ == 4) {
    kernel_name_ += "_1";
    kernelDef = "MULTI_11";
    workItemOutput[1] = 1;
  } else if (kernel_w_ <= 11 && lineSize <= 16 && stride_h_ == 1) {
    kernel_name_ += "_2";
    kernelDef = "MULTI_GEN";
  } else {
    kernel_name_ += "_5";
    kernelDef = "MULTI";
    workItemOutput[1] = 1;
    workItemOutput[0] = 1;
  }

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D KERNELSIZE="
                << kernel_w_ * kernel_h_ << " -D KERNEL_W=" << kernel_w_
                << " -D KERNEL_H=" << kernel_h_ << " -D CHANNELS="
                << channels_ / group_ << " -D STRIDE_H=" << stride_h_
                << " -D STRIDE_W=" << stride_w_ << " -D APPLY_BIAS="
                << bias_term_ << " -D OUTPUT_Z=" << M_
                << " -D " << multiplication_func.c_str() << " -D XPAR="
                << workItemOutput[0] << " -D YPAR=" << workItemOutput[1]
                << " -D ZPAR=" << workItemOutput[2] << " -D "
                << kernelDef.c_str() << " -D CFMulti_11_11_4=U"
                << kernelUKey.c_str() << "_1" << " -D CFMulti_6=U"
                << kernelUKey.c_str() << "_2" << " -D CFMulti=U"
                << kernelUKey.c_str() << "_5";

  if (lineSize <= 4)
    optionsString << " -D DTImage=" << "Dtype4";
  else if (lineSize <= 8)
    optionsString << " -D DTImage=" << "Dtype8";
  else
    optionsString << " -D DTImage=" << "Dtype16";

  if (kernel_w_ <= 4)
    optionsString << " -D DTKernel=" << "Dtype4";
  else if (kernel_w_ <= 8)
    optionsString << " -D DTKernel=" << "Dtype8";
  else
    optionsString << " -D DTKernel=" << "Dtype16";

  string options = optionsString.str();
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
      this->device_->id());

  try {
    viennacl::ocl::program & program = submit_conv_spatial_program(&ctx,
                                                                   kernel_name_,
                                                                   options);
    cl_ulong privateMemUsed;
    viennacl::ocl::kernel & kernel = program.get_kernel(kernel_name_);
    clGetKernelWorkGroupInfo(kernel.handle().get(),
                             viennacl::ocl::current_device().id(),
                             CL_KERNEL_PRIVATE_MEM_SIZE,
                             sizeof(cl_ulong), &privateMemUsed,
                             NULL);
    size_t workSize[3] = { 1, 1, 1 };
    if (privateMemUsed == 0) {
      kernelQueue.push_back(
          new kernelConfig(kernel_name_, workSize, workSize, workItemOutput,
                           true, false, false, 1));
      dbgPrint(std::cout <<
          "successfully generated kernel using generate Kernel"
          << std::endl);
    } else {
      ctx.delete_program(kernel_name_);
    }
  } catch (std::exception & e) {
    dbgPrint(std::cout << e.what() << std::endl);
    return false;
  }

  return true;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::swizzleWeights(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp swizzled_factor) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
      this->device_->id());
  viennacl::ocl::program &program = this->device_->program();
  viennacl::ocl::kernel &oclk_copy_weight = program.get_kernel(
      CL_KERNEL_SELECT("copyWeightsSwizzled"));
  cl_uint argIdx = 0;

  int_tp channels = this->channels_ / this->group_;
  oclk_copy_weight.arg(argIdx++, WrapHandle((cl_mem) weight, &ctx));
  oclk_copy_weight.arg(argIdx++, WrapHandle((cl_mem) swizzled_weights, &ctx));
  oclk_copy_weight.arg(argIdx++, kernel_w_);
  oclk_copy_weight.arg(argIdx++, kernel_h_);
  oclk_copy_weight.arg(argIdx++, channels);
  oclk_copy_weight.arg(argIdx++, this->num_output_);
  oclk_copy_weight.arg(argIdx++, swizzled_factor);
  const size_t global_work_size_Copy[3] = { (size_t) (this->num_output_
      * channels * kernel_w_ * kernel_h_), 1, 1 };

  OCL_CHECK(clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                       oclk_copy_weight.handle().get(), 3, NULL,
                                       global_work_size_Copy, NULL, 0, NULL,
                                       NULL));
}

template<>
void ConvolutionLayerSpatial<float>::calculate_global_size(int_tp batch,
                                  int_tp* wio,  // work item output size
                                  size_t* lSize,  // local size
                                  size_t* gSize) {  // global size
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
void ConvolutionLayerSpatial<Dtype>::pad_image(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp image_offset,
    kernelConfig* config,
    int_tp imgNum) {
#ifdef USE_GREENTEA
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
      this->device_->id());
  // Copy kernel
  viennacl::ocl::program &program = this->device_->program();
  viennacl::ocl::kernel &oclk_copy = program.get_kernel(
                                       CL_KERNEL_SELECT("copyImage"));
  cl_uint argIdx = 0;
  int_tp col_data_offset = 0;
  int_tp channels = this->channels_;

  oclk_copy.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
  oclk_copy.arg(argIdx++, image_offset);
  oclk_copy.arg(argIdx++, channels);
  oclk_copy.arg(argIdx++, height_);
  oclk_copy.arg(argIdx++, width_);
  oclk_copy.arg(argIdx++, padded_height_);
  oclk_copy.arg(argIdx++, padded_width_);
  oclk_copy.arg(argIdx++, pad_h_);
  oclk_copy.arg(argIdx++, pad_w_);
  oclk_copy.arg(argIdx++, WrapHandle((cl_mem) col_data, &ctx));
  oclk_copy.arg(argIdx++, col_data_offset);
  oclk_copy.arg(argIdx++, imgNum);
  const size_t global_work_size_Copy[3] = { (size_t) padded_width_,
      (size_t) padded_height_, (size_t) channels };

  clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                         oclk_copy.handle().get(), 3, NULL,
                         global_work_size_Copy, NULL, 0, NULL, NULL);
#endif
}

template<>
bool ConvolutionLayerSpatial<float>::create_basic_kernel(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  // Standard spatial setup is done here
  std::stringstream keyBuilder;
  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  std::string kernelDef = "MULTI";
  std::string kernelUKey = generate_specific_key(1, blockWidth, blockHeight,
                                                 blockDepth);

  int_tp workItemOutput[3];
  workItemOutput[0] = 1;
  workItemOutput[1] = 1;
  workItemOutput[2] = 1;

  kernel_name_ = "U";
  kernel_name_ += kernelUKey.c_str();
  kernel_name_ += "_BASIC";

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D KERNELSIZE="
                << kernel_w_ * kernel_h_ << " -D KERNEL_W=" << kernel_w_
                << " -D KERNEL_H=" << kernel_h_ << " -D CHANNELS="
                << channels_ / group_ << " -D STRIDE_H=" << stride_h_
                << " -D STRIDE_W=" << stride_w_ << " -D APPLY_BIAS="
                << bias_term_ << " -D OUTPUT_Z=" << M_
                << " -D XPAR=" << workItemOutput[0] << " -D YPAR="
                << workItemOutput[1] << " -D ZPAR=" << workItemOutput[2]
                << " -D " << kernelDef.c_str() << " -D CFMulti=U"
                << kernelUKey.c_str() << "_BASIC";

  string options = optionsString.str();

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  try {
    submit_conv_spatial_program(&ctx, kernel_name_, options);
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
  CHECK_EQ(error, CL_SUCCESS) << "Failed to create sub buffer." << std::endl;
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

template<>
cl_int ConvolutionLayerSpatial<float>::convolve(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  viennacl::ocl::program & program = ctx.get_program(config->kernelName);
  viennacl::ocl::kernel &kernel = program.get_kernel(config->kernelName);
  cl_int err = 0;

  if (config->kernelType != 2) {
    for (int_tp n = 0; n < numImages; ++n) {
      for (int_tp g = 0; g < group_; ++g) {
        bias_offset_ = M_ * g;
        int_tp image_offset = n * this->bottom_dim_
            + width_ * height_ * (channels_ / group_) * g;
        int_tp output_image_offset = n * this->top_dim_
            + output_w_ * output_h_ * M_ * g;

        cl_uint argIdx = 0;
        int_tp kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_
            * g;

        // Copy image
        if (pad_w_ > 0 || pad_h_ > 0) {
          pad_image(bottom, top, image_offset, config, numImages);
          image_offset = 0;
          kernel.arg(argIdx++, WrapHandle((cl_mem) col_data, &ctx));
        } else {
          kernel.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
        }
        kernel.arg(argIdx++, image_offset);
        kernel.arg(argIdx++, WrapHandle((cl_mem) weight, &ctx));
        kernel.arg(argIdx++, kernel_offset);
        kernel.arg(argIdx++, WrapHandle((cl_mem) bias_, &ctx));
        kernel.arg(argIdx++, bias_offset_);
        kernel.arg(argIdx++, WrapHandle((cl_mem) top_data, &ctx));
        kernel.arg(argIdx++, output_image_offset);
        kernel.arg(argIdx++, (uint16_t)padded_width_);
        kernel.arg(argIdx++, (uint16_t)padded_height_);
        kernel.arg(argIdx++, (uint16_t)output_w_);
        kernel.arg(argIdx++, (uint16_t)output_h_);
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
        viennacl::backend::finish();
      }
    }
  } else {
    swizzleWeights(bottom, top, 16);
    size_t total_bottom_size = bottom_dim_ * numImages;
    size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
    size_t total_bias_size = M_ * group_;
    size_t total_top_size = top_dim_ * numImages;
    for (int_tp g = 0; g < group_; ++g) {
      bias_offset_ = M_ * g;
      int_tp image_offset = width_ * height_ * (channels_ / group_) * g;
      int_tp output_image_offset = output_w_ * output_h_ * M_ * g;

      cl_uint argIdx = 0;
      int_tp kernel_offset = kernel_h_ * kernel_w_
                             * (channels_ / group_) * M_ * g;
      // Copy image
      cl_mem input_image;
      if (pad_w_ > 0 || pad_h_ > 0) {
        pad_image(bottom, top, image_offset, config, numImages);
        image_offset = 0;
        input_image = (cl_mem) col_data;
      } else {
        input_image = (cl_mem) bottom_data;
      }
      setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx, input_image,
                         image_offset, total_bottom_size - image_offset,
                         true, false);
      setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                         (cl_mem) swizzled_weights,
                         kernel_offset, total_kernel_size - kernel_offset,
                         true, true);
      setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx, (cl_mem) bias_,
                         bias_offset_, total_bias_size - bias_offset_,
                         true, true);
      setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                         (cl_mem) top_data,
                         output_image_offset,
                         total_top_size - output_image_offset,
                         false, false);
      kernel.arg(argIdx++, (uint16_t)padded_width_);
      kernel.arg(argIdx++, (uint16_t)padded_height_);
      kernel.arg(argIdx++, (uint16_t)output_w_);
      kernel.arg(argIdx++, (uint16_t)output_h_);
      err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                   kernel.handle().get(), 3,
                                   NULL,
                                   config->global_work_size,
                                   config->local_work_size, 0, NULL,
                                   NULL);
      if (err != CL_SUCCESS)
        return err;
      viennacl::backend::finish();
    }

    if (group_ > 1) {
      viennacl::backend::finish();
      cleanTmpSubBuffers(bottom, top);
    }
  }

  return err;
}

template<>
float ConvolutionLayerSpatial<float>::timed_convolve(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  Timer timer;
  timer.initted();
  timer.Start();
  cl_int err;
  dbgPrint(std::cout << "Bechmarking kernel: " << config->kernelName
           << std::endl);
  err = convolve(bottom, top, index, num_, config);
  timer.Stop();
  if (err != CL_SUCCESS) {
    config->tested = true;
    config->verified = false;
  }

  float elapsedTime = timer.MilliSeconds();
#ifdef dbg
  double out_w = output_w_;
  double out_h = output_h_;
  double out_z = M_;
  double k_w = kernel_w_;
  double k_h = kernel_h_;
  double k_z = channels_;
  double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z)*num_;
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

template<>
bool ConvolutionLayerSpatial<float>::verify_result(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp index,
    int_tp numImages, const Blob<float> &verify_blob, kernelConfig* config) {

  uint_tp verificationFail = 0;

  if (config->verified)
    return true;
  else if (config->tested)
    return false;

  config->executionTime = timed_convolve(bottom, top, index, numImages,
                                         config);
  const float *verify_data = verify_blob.cpu_data();
  const float *data = top[index]->cpu_data();

  for (int_tp n = 0; n < numImages; ++n) {
    for (int_tp g = 0; g < group_; ++g) {
      int_tp output_image_offset = n * this->top_dim_
          + output_w_ * output_h_ * M_ * g;
      for (int out_ch = 0; out_ch < M_ && !verificationFail; out_ch++)
        for (int h = 0; h < output_h_ && !verificationFail; h++)
          for (int w = 0; w < output_w_; w++) {
            size_t offset = output_image_offset + out_ch * output_w_ * output_h_
                            + h * output_w_ + w;
            if (fabs(data[offset] - verify_data[offset]) >
                       0.1 * fabs(verify_data[offset]) &&
                !(fabs(verify_data[offset]) < 1.e-3
                  && fabs(data[offset] - verify_data[offset]) < 1.e-4)) {
              dbgPrint(printf("test verification failed @ image %d group %d"
                              "out_ch %d h %d w %d got %G expected %G\n",
                      n, g, out_ch, h, w, data[offset], verify_data[offset]));
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

template<>
bool ConvolutionLayerSpatial<float>::setup_IDLF(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  std::string kernelUKey = generate_specific_key(2, blockWidth, blockHeight,
                                                 blockDepth);
  int_tp workItemOutput[3] = { blockWidth, blockHeight, blockDepth };
  std::string kernelDef = "MULTI";

  const int_tp num_output_maps = M_;
  int_tp output_width = output_w_;
  int_tp output_height = output_h_;
  int_tp output_block_width = blockWidth;
  int_tp output_block_height = blockHeight;
  int_tp simd_size = 16;
  int_tp num_batches = num_;

  kernel_name_ = "U";
  kernel_name_ += kernelUKey.c_str();
  kernel_name_ += "_SIMD16";
  kernelDef = "SIMD16";

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D IDLF" << " -D "
                << kernelDef.c_str() << " -D convolve_simd16=U"
                << kernelUKey.c_str() << "_SIMD16";

  const int_tp in_buffer_size = (output_block_height - 1) * stride_h_
                                 + kernel_h_;
  const int_tp last_block_width =
      (output_width % output_block_width == 0) ?
          output_block_width : output_width % output_block_width;
  const int_tp last_block_height =
      (output_height % output_block_height == 0) ?
          output_block_height : output_height % output_block_height;

  size_t global_size[3] = { (size_t) (output_width + output_block_width - 1)
      / output_block_width, (size_t) (output_height + output_block_height - 1)
      / output_block_height, (size_t) num_batches * num_output_maps };

  size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };

  optionsString << " -D SIMD_SIZE=" << simd_size
                << " -D filter_qualifier=__global" << " -D OUT_BLOCK_WIDTH="
                << output_block_width << " -D OUT_BLOCK_HEIGHT="
                << output_block_height << " -D IN_BUFFER_SIZE="
                << in_buffer_size << " -D LAST_BLOCK_WIDTH=" << last_block_width
                << " -D LAST_BLOCK_HEIGHT=" << last_block_height
                << " -D INPUT_DEPTH=" << channels_ / group_
                << " -DTOTAL_INPUT_DEPTH_SIZE=" << channels_
                << " -DTOTAL_OUTPUT_DEPTH=" << num_output_
                << " -DINPUT_START_X=" << 0 << " -DINPUT_START_Y=" << 0
                << " -DINPUT_START_Z=" << 0
                << " -DFILTER_WIDTH=" << kernel_w_
                << " -DFILTER_HEIGHT=" << kernel_h_
                << " -DNUM_FILTERS=" << M_ << " -DSTRIDEX=" << stride_w_
                << " -DSTRIDEY=" << stride_h_ << " -DOWPAD=" << 0 << " -DOHPAD="
                << 0 << " -DOUT_BUFF_OFFSET=" << 0;

  string options = optionsString.str();
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());

  viennacl::ocl::program & program = submit_conv_spatial_program(&ctx,
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

template<>
bool ConvolutionLayerSpatial<float>::tune_local_size(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
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

template<>
void ConvolutionLayerSpatial<float>::create_convolution_kernel(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    int_tp kernelType,
    int_tp blockWidth, int_tp blockHeight,
    int_tp blockDepth) {
  if (kernelType == 1)
    generate_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == 2)
    setup_IDLF(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == 4)
    create_basic_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
  else
    assert(0);
}

template<>
void ConvolutionLayerSpatial<float>::setup_convolution(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    const Blob<float> &verify_blob) {
  // Generates static key_
  generate_key();
  // Initializes unique kernel ID
  kernel_uid_ = 0;

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  const viennacl::ocl::device &device = ctx.current_device();
  if (device.vendor().find("Intel") != std::string::npos &&
    M_ % 16 == 0) {
    /* IDLF kernels are using Intel specific extension which make
       them intel only. */
    int kernelCnt = 0;
    for (uint32_t width = 14; width > 0; width--) {
      int candidate = 0;
      if (width > output_w_)
        continue;
      for (uint32_t height = 14; height > 0; height--) {
        if (height * width > 32 || height > output_h_)
          continue;
        int tile_x = kernel_w_ + (width - 1) * stride_w_;
        int tile_y = kernel_h_ + (height - 1) * stride_h_;
        int tile_y_stride = 64 / tile_x;

        if (tile_x % 4 != 0 && tile_x <= 16) {
          create_convolution_kernel(bottom, top, 2, width, height, 1);
          candidate++;
        } else if ((tile_x % 4 == 0) &&
                 ((tile_y + tile_y_stride - 1) / tile_y_stride < 4)) {
          create_convolution_kernel(bottom, top, 2, width, height, 1);
          candidate++;
        }
        if (candidate >= 4 && height == 2)
          break;
      }
      kernelCnt += candidate;
      if (kernelCnt >= 12 && width == 2)
        break;
    }
  } else {
    for (int_tp y = 1; y < 4; y += 1)
      for (int_tp z = 1; z < 16 && z < M_; z += 1) {
        if (4 * y * z > 32) continue;
        create_convolution_kernel(bottom, top, 1, 4, y, z);
      }
  }
  for (int_tp x = 0; x < kernelQueue.size(); x++)
    if (tune_local_size(bottom, top, kernelQueue[x])) {
      kernelQueue[x]->executionTime = timed_convolve(bottom, top, bottom_index_,
                                                     num_, kernelQueue[x]);
    } else {
      // skip those kernels without a good local size.
      kernelQueue[x]->verified = false;
      kernelQueue[x]->tested = true;
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
      bool verified = verify_result(bottom, top, bottom_index_, num_,
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
    verification = verify_result(bottom, top, bottom_index_, num_,
                                 verify_blob, kernelQueue[kernel_index_]);
    CHECK_EQ(verification, true) << "Basic kernel failed verification."
                                 << std::endl;
  }
  this->bestKernelConfig = kernelQueue[kernel_index_];

  dbgPrint(std::cout << "Convolution Time:"
                     << kernelQueue[kernel_index_]->executionTime << std::endl);

  for (int_tp x = 0; x < kernelQueue.size(); x++) {
    if (x != kernel_index_) {
      viennacl::ocl::current_context().delete_program(
          kernelQueue[x]->kernelName);
      delete kernelQueue[x];
    }
  }
  kernelQueue.clear();

  tuned_ = true;

  const boost::filesystem::path& path = CACHE_DIRECTORY;
  const boost::filesystem::path& dir =
                   boost::filesystem::unique_path(path).string();
  bool hasCacheDir = false;
  if (!boost::filesystem::exists(dir))
    hasCacheDir = boost::filesystem::create_directory(dir);
  else
    hasCacheDir = boost::filesystem::is_directory(dir);

  if (hasCacheDir != true) {
    std::cout << "Failed to create cache directory,"
              << "will tune again for next running" << std::endl;
    return;
  }

  string outputFile;
  outputFile = CACHE_DIRECTORY + key_;
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
}

template<>
void ConvolutionLayerSpatial<float>::Forward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {

  for (int_tp i = 0; i < bottom.size(); ++i) {
    bottom_index_ = i;
    bottom_data = bottom[i]->gpu_data();
    top_data = top[i]->mutable_gpu_data();
    col_data = spatial_col_buffer_.mutable_gpu_data();
    weight = this->blobs_[0]->gpu_data();
    swizzled_weights = swizzled_weights_.mutable_gpu_data();

    weight_offset = M_ * K_;
    col_offset = K_ * N_;
    top_offset = M_ * N_;

    bias_ = NULL;
    bias_offset_ = 0;

    if (bias_term_)
      bias_ = this->blobs_[1]->gpu_data();

    if (!tuned_) {
      Blob<float> verify_blob;
      verify_blob.ReshapeLike(*top[i]);
      float *verify_data = verify_blob.mutable_gpu_data();
      const float *weight_gpu_data = this->blobs_[0]->gpu_data();
      const float *bottom_gpu_data = bottom[i]->gpu_data();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->forward_gpu_gemm(bottom_gpu_data, n * this->bottom_dim_,
                               weight_gpu_data, verify_data,
                               n * this->top_dim_);
        if (this->bias_term_) {
          const float* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(verify_data, n * this->top_dim_, bias);
        }
      }
      setup_convolution(bottom, top, verify_blob);
      CHECK_EQ(tuned_, true) << "Spatial convolution auto-tuning failed.";
    }

    convolve(bottom, top, i, num_, bestKernelConfig);
  }
  viennacl::backend::finish();
}

template<>
void ConvolutionLayerSpatial<float>::Backward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom) {
  const float* weight = this->blobs_[0]->gpu_data();
  float* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const float* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      float* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const float* bottom_data = bottom[i]->gpu_data();
      float* bottom_diff = bottom[i]->mutable_gpu_diff();
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
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Generates static key_
  std::string previous_key = key_;
  generate_key();
  if (tuned_) {
    if (key_.compare(previous_key) == 0)
      return;
    tuned_ = false;
    viennacl::ocl::current_context().
      delete_program(bestKernelConfig->kernelName);
    delete bestKernelConfig;
    bestKernelConfig = NULL;
  }
  // Initializes unique kernel ID
  kernel_uid_ = 0;

  string outputFile;
  outputFile = CACHE_DIRECTORY + key_;
  std::ifstream cachedKernel(outputFile.c_str());

  if (cachedKernel) {
    int_tp x, y, z, type;
    cachedKernel >> x;
    cachedKernel >> y;
    cachedKernel >> z;
    cachedKernel >> type;
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

template<>
bool ConvolutionLayerSpatial<double>::generate_kernel(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  NOT_IMPLEMENTED;
  return false;
}

template void ConvolutionLayerSpatial<float>::SetUp(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
    caffe::Backend backend);

template void ConvolutionLayerSpatial<double>::SetUp(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    caffe::Backend backend);

template void ConvolutionLayerSpatial<float>::swizzleWeights(
    const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top,
    int_tp swizzle_factor);
template void ConvolutionLayerSpatial<double>::swizzleWeights(
    const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top,
    int_tp swizzle_factor);
template void ConvolutionLayerSpatial<float>::pad_image(
    const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top,
    int_tp image_offset, kernelConfig* config,
    int_tp imgNum);
template void ConvolutionLayerSpatial<double>::pad_image(
    const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top,
    int_tp image_offset, kernelConfig* config,
    int_tp imgNum);

template<>
void ConvolutionLayerSpatial<double>::create_convolution_kernel(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp kernelType,
    int_tp blockWidth, int_tp blockHeight,
    int_tp blockDepth) {
  NOT_IMPLEMENTED;
  return;
}
template<>
bool ConvolutionLayerSpatial<double>::setup_IDLF(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  NOT_IMPLEMENTED;
  return false;
}

template<>
bool ConvolutionLayerSpatial<double>::verify_result(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp index,
    int_tp numImages, const Blob<double> &verify_blob, kernelConfig* config) {
  NOT_IMPLEMENTED;
  return false;
}

template<>
bool ConvolutionLayerSpatial<double>::create_basic_kernel(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  NOT_IMPLEMENTED;
  return false;
}

template<>
bool ConvolutionLayerSpatial<double>::tune_local_size(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    kernelConfig* config) {
  NOT_IMPLEMENTED;
  return false;
}

template<>
cl_int ConvolutionLayerSpatial<double>::convolve(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  NOT_IMPLEMENTED;
  return false;
}

template<>
float ConvolutionLayerSpatial<double>::timed_convolve(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    int_tp index,
    int_tp numImages, kernelConfig* config) {
  NOT_IMPLEMENTED;
  return 0.f;
}

template<>
void ConvolutionLayerSpatial<double>::setup_convolution(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top,
    const Blob<double> &verify_blob) {
  NOT_IMPLEMENTED;
}

template<>
void ConvolutionLayerSpatial<double>::calculate_global_size(
    int_tp batch,
    int_tp* workItemOutput,
    size_t* localSizes, size_t* globalSizes) {
  NOT_IMPLEMENTED;
}

template<>
void ConvolutionLayerSpatial<double>::generate_key() {
  NOT_IMPLEMENTED;
}
template<>
std::string ConvolutionLayerSpatial<double>::generate_unique_key() {
  NOT_IMPLEMENTED;
  return "";
}

template<>
std::string ConvolutionLayerSpatial<double>::generate_specific_key(
    int_tp type, int_tp blockWidth, int_tp blockHeight, int_tp blockDepth) {
  NOT_IMPLEMENTED;
  return "";
}

template<>
void ConvolutionLayerSpatial<double>::Forward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template<>
void ConvolutionLayerSpatial<double>::Backward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
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
