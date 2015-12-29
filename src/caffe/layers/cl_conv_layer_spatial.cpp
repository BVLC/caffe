#ifdef USE_OCL

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/conv_spatial_layer.hpp"


extern "C" const char _cl_conv_layer_spatial_start;
extern "C" const char _cl_conv_layer_spatial_end;

namespace caffe {

//#define dbg

#ifdef dbg
#define dbgPrint(x) (x)
#else
#define dbgPrint(x)
#endif

  template <>
  void ConvolutionLayerSpatial<float>::generate_key() {
    std::stringstream keyBuilder;
    keyBuilder << kernel_w_ <<
    "_" << kernel_h_ <<
    "_" << channels_ <<
    "_" << group_ <<
    "_" << stride_h_ <<
    "_" << stride_w_ <<
    "_" << bias_term_ <<
    "_" << padded_width_ <<
    "_" << padded_height_ <<
    "_" << num_ <<
    "_" << group_ <<
    "_" << M_;
    key_ = keyBuilder.str();
  }

  template <>
  std::string ConvolutionLayerSpatial<float>::generate_unique_key() {
    std::stringstream keyBuilder;
    keyBuilder << key_ <<
    "" << kernel_uid_;
    kernel_uid_++;
    return keyBuilder.str();
  }

  template <>
  std::string ConvolutionLayerSpatial<float>::generate_specific_key(int type, int blockWidth, int blockHeight, int blockDepth) {
    std::stringstream keyBuilder;
    keyBuilder << key_ <<
    "_" << type <<
    "_" << blockWidth <<
    "_" << blockHeight <<
    "_" << blockDepth;
    return keyBuilder.str();
    }

  template <>
  bool ConvolutionLayerSpatial<float>::generate_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int blockWidth, int blockHeight, int blockDepth) {
    // Standard spatial setup is done here
    std::string kernelDef = "MULTI";
    std::string stringBuilder;
    std::stringstream optionsString;

    int workItemOutput[3];
    int yDim = blockHeight;
    int zDim = blockDepth;

    std::string kernelUKey = generate_specific_key(1,blockWidth,blockHeight,blockDepth);
    std::stringstream multFunctionBuilder;
    workItemOutput[0] = 4;
    workItemOutput[1] = yDim;
    workItemOutput[2] = zDim;

    std::string multiplication_func =
        "floatDotV4(V1,V2)=(V1.s0123*V2.s0123)";

    if (kernel_w_ <= 11) {
      multFunctionBuilder << "floatDotV4(V1,V2)=" << "(";
      for (int kw = 0; kw < kernel_w_; kw++) {
        multFunctionBuilder << "V1.s" << std::hex << kw << kw + 1*stride_w_
            << kw + 2*stride_w_ << kw + 3*stride_w_ << std::dec;
        multFunctionBuilder << "*";
        multFunctionBuilder << "V2.s" << std::hex << kw << std::dec;

        if (kw == kernel_w_ -1)
          multFunctionBuilder << ")";
        else
          multFunctionBuilder << "+";
      }
      multiplication_func = multFunctionBuilder.str();
    }

    int lineSize = kernel_w_ + (workItemOutput[0]-1)*stride_w_;

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
    optionsString << "-cl-fast-relaxed-math "
    << " -D KERNELSIZE=" << kernel_w_*kernel_h_
    << " -D KERNEL_W=" << kernel_w_
    << " -D KERNEL_H=" << kernel_h_
    << " -D CHANNELS=" << channels_/group_
    << " -D STRIDE_H=" << stride_h_
    << " -D STRIDE_W=" << stride_w_
    << " -D APPLY_BIAS=" << bias_term_
    << " -D OUTPUT_W=" << output_w_
    << " -D OUTPUT_H=" << output_h_
    << " -D OUTPUT_Z=" << M_
    << " -D WIDTH=" << padded_width_
    << " -D HEIGHT=" << padded_height_
    << " -D " << multiplication_func.c_str()
    << " -D XPAR=" << workItemOutput[0]
    << " -D YPAR=" << workItemOutput[1]
    << " -D ZPAR=" << workItemOutput[2]
    << " -D " << kernelDef.c_str()
    << " -D CFMulti_11_11_4=U" << kernelUKey.c_str()<< "_1"
    << " -D CFMulti_6=U" << kernelUKey.c_str() << "_2"
    << " -D CFMulti=U" << kernelUKey.c_str() << "_5";

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

    stringBuilder = optionsString.str();
    const char * options = stringBuilder.c_str();

    ClState& state = Caffe::cl_state();
    int err = state.submit_program(
        kernel_name_.c_str(),
        &_cl_conv_layer_spatial_start,
        &_cl_conv_layer_spatial_end,
        options,
        true);


    if (err != CL_SUCCESS && err != true)
      return false;

    ClKernel kernel;
    cl_ulong privateMemUsed;
    kernel = state.get_kernel(kernel_name_.c_str());
    clGetKernelWorkGroupInfo(kernel, state.get_device(),
        CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &privateMemUsed,
        NULL);

    size_t workSize[3] = {1, 1, 1};
    if (privateMemUsed == 0) {
      kernelQueue.push_back( new kernelConfig(kernel_name_,
          workSize, workSize, workItemOutput, true,
          false, false, false,1));
      dbgPrint(std::cout <<
          "successfully generated kernel using generate Kernel"
          << std::endl);
    } else {
      state.release_program(kernel_name_.c_str());
    }

    return true;
  }

  template <>
  bool ConvolutionLayerSpatial<float>::generate_batched_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int blockWidth, int blockHeight, int blockDepth) {
    std::string kernelDef = "MULTI";
    std::stringstream multFunctionBuilder;
    std::string stringBuilder;
    std::stringstream optionsString;
    int workItemOutput[3];
    std::string kernelUKey = generate_specific_key(3,blockWidth,blockHeight,blockDepth);

    workItemOutput[0] = 4;
    workItemOutput[1] = 1;
    workItemOutput[2] = 1;

    std::string multiplication_func = "floatDotV4(V1,V2)=(V1.s0123*V2.s0123)";

    if (kernel_w_ <= 11) {
      multFunctionBuilder << "floatDotV4(V1,V2)=" << "(";
      for (int kw = 0; kw < kernel_w_; kw++) {
        multFunctionBuilder << "V1.s" << std::hex
            << kw << kw + 1*stride_w_ << kw + 2*stride_w_ << kw + 3*stride_w_
            << std::dec;
        multFunctionBuilder << "*";
        multFunctionBuilder << "V2.s" << std::hex << kw << std::dec;

        if (kw == kernel_w_ -1)
          multFunctionBuilder << ")";
        else
          multFunctionBuilder << "+";
      }
      multiplication_func = multFunctionBuilder.str();
    }

    if (stride_h_ > 1)
      workItemOutput[1] = 1;
    else
      workItemOutput[1] = blockHeight;

      workItemOutput[2] = blockDepth;

    int lineSize = kernel_w_ + (workItemOutput[0]-1)*stride_w_;

    kernel_name_ = "U";
    kernel_name_ += kernelUKey.c_str();
    if (lineSize <= 16) {
      kernel_name_ += "_2";
      kernelDef = "MULTI_BATCHED";
    } else {
      return false;
    }

    // Build list of options and defines
    optionsString.str("");
    optionsString << " -cl-fast-relaxed-math "
    << " -D KERNELSIZE=" << kernel_w_*kernel_h_
    << " -D KERNEL_W=" << kernel_w_
    << " -D KERNEL_H=" << kernel_h_
    << " -D CHANNELS=" << channels_/group_
    << " -D STRIDE_H=" << stride_h_
    << " -D STRIDE_W=" << stride_w_
    << " -D APPLY_BIAS=" << bias_term_
    << " -D OUTPUT_W=" << output_w_
    << " -D OUTPUT_H=" << output_h_
    << " -D OUTPUT_Z=" << M_
    << " -D IMG_OFFSET=" << padded_width_*padded_height_*channels_
    << " -D OUTPUT_OFFSET=" << this->top_dim_
    << " -D WIDTH=" << padded_width_
    << " -D HEIGHT=" << padded_height_
    << " -D " << multiplication_func.c_str()
    << " -D XPAR=" << workItemOutput[0]
    << " -D YPAR=" << workItemOutput[1]
    << " -D ZPAR=" << workItemOutput[2]
    << " -D " << kernelDef.c_str()
    << " -D CFMulti_6=U" << kernelUKey.c_str() << "_2";

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

    stringBuilder = optionsString.str();
    const char * options = stringBuilder.c_str();

    ClState& state = Caffe::cl_state();
    int err = state.submit_program(kernel_name_.c_str(),
        &_cl_conv_layer_spatial_start, &_cl_conv_layer_spatial_end, options,
        true);

    if (err != CL_SUCCESS && err != true)
      return false;

    ClKernel kernel;
    cl_ulong privateMemUsed;
    kernel = state.get_kernel(kernel_name_.c_str());
    clGetKernelWorkGroupInfo(kernel, state.get_device(),
        CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &privateMemUsed,
        NULL);

    if (privateMemUsed == 0) {
      size_t workSize[3] = {1, 1, 1};
      kernelQueue.push_back(
          new kernelConfig(kernel_name_, workSize, workSize,
              workItemOutput, true, false, true,
              false,3));
    } else {
      state.release_program(kernel_name_.c_str());
    }

    dbgPrint(std::cout <<
        "successfully generated Batched kernel using generate Kernel" <<
        std::endl);

    return true;
  }



  template <>
  void ConvolutionLayerSpatial<float>::swizzleWeights(int swizzle_factor) {
    ClState& state = Caffe::cl_state();
    ClKernel copyWeightsKernel = state.get_kernel("copyWeightsSwizzled");
    cl_uint argIdx = 0;

    int channels = channels_ / group_;
    copyWeightsKernel.set_arg(argIdx++, weight);
    copyWeightsKernel.set_arg(argIdx++, swizzled_weights);
    copyWeightsKernel.set_arg(argIdx++, kernel_w_);
    copyWeightsKernel.set_arg(argIdx++, kernel_h_);
    copyWeightsKernel.set_arg(argIdx++, channels);
    copyWeightsKernel.set_arg(argIdx++, num_output_);
    copyWeightsKernel.set_arg(argIdx++, swizzle_factor);
    const size_t global_work_size_Copy[3] =
        {(size_t)(num_output_*channels*kernel_w_*kernel_h_), 1, 1};
    OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(),
        copyWeightsKernel, 3, NULL, global_work_size_Copy, NULL, 0, NULL,
        NULL));
  }

  template<>
  void ConvolutionLayerSpatial<float>::calculate_global_size(int batch,
      int* wio,  // work item output size
      size_t* lSize,  // local size
      size_t* gSize) {  // global size
    gSize[0] =
        ceil((fmax(static_cast<float>(output_w_)/wio[0], 1.0))
            /lSize[0])*lSize[0];
    gSize[1] =
        ceil((fmax(static_cast<float>(output_h_)/wio[1], 1.0))
            /lSize[1])*lSize[1];
    gSize[2] =
        ceil(static_cast<float>((ceil(static_cast<float>(M_)*batch/wio[2])))
            /lSize[2])*lSize[2];
  }

  template <>
  void ConvolutionLayerSpatial<float>::pad_image(
      int image_offset,
      kernelConfig* config,
      int imgNum) {
    ClState& state = Caffe::cl_state();
    // Copy kernel
    ClKernel copyKernel = state.get_kernel("copyImage");

    cl_uint argIdx = 0;
    int col_data_offset = 0;
    int channels = channels_/group_;

    if (config->batched_execute) {
      for (int x = 0; x < imgNum; x++) {
        argIdx = 0;
        int image_offsetLocal = height_*width_*channels_*x + image_offset;
        col_data_offset =
            padded_width_*padded_height_*channels_*x + image_offset;
        copyKernel.set_arg(argIdx++, bottom_data);
        copyKernel.set_arg(argIdx++, image_offsetLocal);
        copyKernel.set_arg(argIdx++, channels);
        copyKernel.set_arg(argIdx++, height_);
        copyKernel.set_arg(argIdx++, width_);
        copyKernel.set_arg(argIdx++, padded_height_);
        copyKernel.set_arg(argIdx++, padded_width_);
        copyKernel.set_arg(argIdx++, pad_h_);
        copyKernel.set_arg(argIdx++, pad_w_);
        copyKernel.set_arg(argIdx++, col_data);
        copyKernel.set_arg(argIdx++, col_data_offset);
        const size_t global_work_size_Copy[3] =
            {(size_t)padded_width_, (size_t)padded_height_, (size_t)channels};
        OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), copyKernel,
            3, NULL, global_work_size_Copy, NULL, 0, NULL, NULL));
      }
    } else {
      copyKernel.set_arg(argIdx++, bottom_data);
      copyKernel.set_arg(argIdx++, image_offset);
      copyKernel.set_arg(argIdx++, channels);
      copyKernel.set_arg(argIdx++, height_);
      copyKernel.set_arg(argIdx++, width_);
      copyKernel.set_arg(argIdx++, padded_height_);
      copyKernel.set_arg(argIdx++, padded_width_);
      copyKernel.set_arg(argIdx++, pad_h_);
      copyKernel.set_arg(argIdx++, pad_w_);
      copyKernel.set_arg(argIdx++, col_data);
      copyKernel.set_arg(argIdx++, col_data_offset);
      const size_t global_work_size_Copy[3] =
          {(size_t)padded_width_, (size_t)padded_height_, (size_t)channels};
      OCL_CHECK(clEnqueueNDRangeKernel(state.get_command_queue(), copyKernel,
          3, NULL, global_work_size_Copy, NULL, 0, NULL, NULL));
    }
  }

  template <>
  bool ConvolutionLayerSpatial<float>::create_basic_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int blockWidth, int blockHeight, int blockDepth) {
    // Standard spatial setup is done here
    std::stringstream keyBuilder;
    std::stringstream multFunctionBuilder;
    std::string stringBuilder;
    std::stringstream optionsString;
    std::string kernelDef = "MULTI";
    std::string kernelUKey = generate_specific_key(1,blockWidth,blockHeight,blockDepth);

    int workItemOutput[3];
    workItemOutput[0] = 1;
    workItemOutput[1] = 1;
    workItemOutput[2] = 1;

    kernel_name_ = "U";
    kernel_name_ += kernelUKey.c_str();
    kernel_name_ += "_BASIC";

    // Build list of options and defines
    optionsString.str("");
    optionsString << "-cl-fast-relaxed-math "
    << " -D KERNELSIZE=" << kernel_w_*kernel_h_
    << " -D KERNEL_W=" << kernel_w_
    << " -D KERNEL_H=" << kernel_h_
    << " -D CHANNELS=" << channels_/group_
    << " -D STRIDE_H=" << stride_h_
    << " -D STRIDE_W=" << stride_w_
    << " -D APPLY_BIAS=" << bias_term_
    << " -D OUTPUT_W=" << output_w_
    << " -D OUTPUT_H=" << output_h_
    << " -D OUTPUT_Z=" << M_
    << " -D WIDTH=" << padded_width_
    << " -D HEIGHT=" << padded_height_
    << " -D XPAR=" << workItemOutput[0]
    << " -D YPAR=" << workItemOutput[1]
    << " -D ZPAR=" << workItemOutput[2]
    << " -D " << kernelDef.c_str()
    << " -D CFMulti=U" << kernelUKey.c_str() << "_BASIC";

    stringBuilder = optionsString.str();
    const char * options = stringBuilder.c_str();

    ClState& state = Caffe::cl_state();
    int err = state.submit_program(kernel_name_.c_str(),
        &_cl_conv_layer_spatial_start, &_cl_conv_layer_spatial_end, options,
        true);

    if (err != CL_SUCCESS && err != true) {
      dbgPrint(std::cout << "Basic kernel generation failed" << std::endl);
      return false;
    }

    size_t localSize[3] = {1, 1, 1};
    size_t globalSize[3];
    calculate_global_size(1, workItemOutput, localSize, globalSize);

    kernelQueue.push_back(
        new kernelConfig( kernel_name_, globalSize, localSize,
            workItemOutput, false, false, false, true,4));

    return true;
  }

  template <>
  bool ConvolutionLayerSpatial<float>::create_verification_kernel(
      const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {
    // Standard spatial setup is done here
    std::stringstream keyBuilder;
    std::stringstream multFunctionBuilder;
    std::string stringBuilder;
    std::stringstream optionsString;
    std::string kernelDef = "VERIFICATION";

    verification_kernel = "U";
    verification_kernel += key_.c_str();
    verification_kernel += "_VERIFICATION";

    // Build list of options and defines
    optionsString.str("");
    optionsString << "-cl-fast-relaxed-math "
    << " -D KERNELSIZE=" << kernel_w_*kernel_h_
    << " -D KERNEL_W=" << kernel_w_
    << " -D KERNEL_H=" << kernel_h_
    << " -D CHANNELS=" << channels_/group_
    << " -D STRIDE_H=" << stride_h_
    << " -D STRIDE_W=" << stride_w_
    << " -D APPLY_BIAS=" << bias_term_
    << " -D OUTPUT_W=" << output_w_
    << " -D OUTPUT_H=" << output_h_
    << " -D OUTPUT_Z=" << M_
    << " -D WIDTH=" << padded_width_
    << " -D HEIGHT=" << padded_height_
    << " -D XPAR=1"
    << " -D YPAR=1"
    << " -D ZPAR=1"
    << " -D " << kernelDef.c_str()
    << " -D CFVerify=U" << key_.c_str() << "_VERIFICATION";

    stringBuilder = optionsString.str();
    const char * options = stringBuilder.c_str();

    ClState& state = Caffe::cl_state();
    int err = state.submit_program(verification_kernel.c_str(),
        &_cl_conv_layer_spatial_start, &_cl_conv_layer_spatial_end, options,
        true);

    if (err != true) {
      dbgPrint(
          std::cout << "Verification kernel generation failed" << std::endl);
      return false;
    }
    return true;
  }

  template <>
  cl_int ConvolutionLayerSpatial<float>::convolve(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    ClState& state = Caffe::cl_state();

    if (config->swizzle_weights)
      swizzleWeights(16);

    ClKernel kernel;
    kernel = state.get_kernel(config->kernelName.c_str());
    cl_int err = 0;

    for (int n = 0; n < numImages; ++n) {
      for (int g = 0; g < group_; ++g) {
        bias_offset_ = M_*g;
        int image_offset = n * this->bottom_dim_ + width_ * height_*(channels_/group_)* g;
        int output_image_offset = n * this->top_dim_ + output_w_ * output_h_ * M_ * g;

        cl_uint argIdx = 0;
        int kernel_offset = kernel_h_*kernel_w_*(channels_/group_) * M_ * g;

        // Copy image
        if (pad_w_ > 0 || pad_h_ > 0) {
          pad_image(image_offset, config, numImages);
          image_offset = 0;
          kernel.set_arg(argIdx++, col_data);
        } else {
          kernel.set_arg(argIdx++, bottom_data);
        }
        kernel.set_arg(argIdx++, image_offset);
        if (config->swizzle_weights)
          kernel.set_arg(argIdx++, swizzled_weights);
        else
          kernel.set_arg(argIdx++, weight);
        kernel.set_arg(argIdx++, kernel_offset);
        kernel.set_arg(argIdx++, bias_);
        kernel.set_arg(argIdx++, bias_offset_);
        kernel.set_arg(argIdx++, top_data);
        kernel.set_arg(argIdx++, output_image_offset);
        if (config->use_null_local) {
          err = clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3,
              NULL, config->global_work_size, NULL, 0, NULL, NULL);
        } else {
          err = clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3,
              NULL, config->global_work_size, config->local_work_size, 0, NULL,
              NULL);
        }

        if (err != CL_SUCCESS)
          return err;
      }
    }

    return err;
  }

  template <>
  cl_int ConvolutionLayerSpatial<float>::batched_convolve(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    ClState& state = Caffe::cl_state();

    if (config->swizzle_weights)
      swizzleWeights(16);

    ClKernel kernel;
    kernel = state.get_kernel(config->kernelName.c_str());
    cl_int err = 0;

    for (int g = 0; g < group_; ++g) {
      bias_offset_ = M_*g;
      int image_offset =  width_ * height_*(channels_/group_)* g;
      int output_image_offset =  output_w_ * output_h_ * M_ * g;

      cl_uint argIdx = 0;
      int kernel_offset = kernel_h_*kernel_w_*(channels_/group_) * M_ * g;

      pad_image(image_offset, config, numImages);
      kernel.set_arg(argIdx++, col_data);
      kernel.set_arg(argIdx++, image_offset);
      if (config->swizzle_weights)
        kernel.set_arg(argIdx++, swizzled_weights);
      else
        kernel.set_arg(argIdx++, weight);
      kernel.set_arg(argIdx++, kernel_offset);
      kernel.set_arg(argIdx++, bias_);
      kernel.set_arg(argIdx++, bias_offset_);
      kernel.set_arg(argIdx++, top_data);
      kernel.set_arg(argIdx++, output_image_offset);
      kernel.set_arg(argIdx++, numImages);
      if  (config->use_null_local)
        err = clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3,
            NULL, config->global_work_size, NULL, 0, NULL, NULL);
      else
        err = clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3,
            NULL, config->global_work_size, config->local_work_size, 0, NULL,
            NULL);
      if  (err != CL_SUCCESS)
        return err;
    }
    return err;
  }

  template <>
  float ConvolutionLayerSpatial<float>::timed_convolve(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    Timer timer;
    timer.initted();
    timer.Start();
    cl_int err;
    if (config->batched_execute)
      err = batched_convolve(bottom, top, index, num_, config);
    else
      err = convolve(bottom, top, index, num_, config);
    timer.Stop();
    if(err != CL_SUCCESS) {
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
    std::cout << "Estimated Gflops:" << ((totalFlops/1000)/1000)/1000
        << std::endl;
    std::cout << "Estimated GFLOPS/S: " <<
        (((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime) << std::endl;
    std::cout << "Estimated utilization: " <<
        ((((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime))/880.0
        << std::endl;
#endif
    return elapsedTime;
  }

  template <>
  bool ConvolutionLayerSpatial<float>::verify_result(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    ClState& state = Caffe::cl_state();
    ClKernel kernel;
    kernel = state.get_kernel(verification_kernel.c_str());
    cl_int err = 0;
    uint verificationFail = 0;
    cl_mem verifcationResult =
        clCreateBuffer(state.get_context(), CL_MEM_USE_HOST_PTR, sizeof(uint),
            &verificationFail, &err);
    OCL_CHECK(err);

    kernelConfig tempConfig;
    tempConfig.batched_execute = false;

    for (int n = 0; n < numImages; ++n) {
      for (int g = 0; g < group_; ++g) {
        cl_uint argIdx = 0;
        bias_offset_ = M_*g;
        int image_offset = n * this->bottom_dim_ + width_ * height_*(channels_/group_)* g;
        int output_image_offset = n * this->top_dim_ + output_w_ * output_h_ * M_ * g;
        int kernel_offset = kernel_h_*kernel_w_*(channels_/group_) * M_ * g;

        if (pad_w_ > 0 || pad_h_ > 0) {
          pad_image(image_offset, &tempConfig, num_);
          image_offset = 0;
          kernel.set_arg(argIdx++, col_data);
        } else {
          kernel.set_arg(argIdx++, bottom_data);
        }
        kernel.set_arg(argIdx++, image_offset);
        kernel.set_arg(argIdx++, weight);
        kernel.set_arg(argIdx++, kernel_offset);
        kernel.set_arg(argIdx++, bias_);
        kernel.set_arg(argIdx++, bias_offset_);
        kernel.set_arg(argIdx++, top_data);
        kernel.set_arg(argIdx++, output_image_offset);
        err = clSetKernelArg(kernel, argIdx++, sizeof(cl_mem),
            &verifcationResult);

        size_t global_work_sizeB[3] =
            {(size_t)output_w_, (size_t)output_h_, (size_t)M_};

        err = clEnqueueNDRangeKernel(state.get_command_queue(), kernel, 3,
            NULL, global_work_sizeB, NULL, 0, NULL, NULL);

        clFinish(state.get_command_queue());
        clEnqueueMapBuffer(state.get_command_queue(), verifcationResult, true,
            CL_MAP_READ, 0, sizeof(uint), 0, NULL, NULL, NULL);

        if (verificationFail)
          return false;

        if (err != CL_SUCCESS)
          return false;
      }
    }
    clFinish(state.get_command_queue());
    return true;
  }

  template <>
  bool ConvolutionLayerSpatial<float>::setup_IDLF(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top, int blockWidth, int blockHeight, int blockDepth) {
    std::stringstream multFunctionBuilder;
    std::string stringBuilder;
    std::stringstream optionsString;
    std::string kernelUKey = generate_specific_key(2,blockWidth,blockHeight,blockDepth);
    int workItemOutput[3] = {blockWidth,blockHeight,blockDepth};
    std::string kernelDef = "MULTI";

    const int num_output_maps = M_;
    int output_width = output_w_;
    int output_height = output_h_;
    int output_block_width = blockWidth;
    int output_block_height = blockHeight;
    int simd_size = 16;
    int num_batches = 1;

    kernel_name_ = "U";
    kernel_name_ += kernelUKey.c_str();
    kernel_name_ += "_SIMD16";
    kernelDef = "SIMD16";

    // Build list of options and defines
    optionsString.str("");
    optionsString << "-cl-fast-relaxed-math "
    << " -D IDLF"
    << " -D " << kernelDef.c_str()
    << " -D convolve_simd16=U" << kernelUKey.c_str() << "_SIMD16";

    const int in_buffer_size = output_block_height + 2;
    const int last_block_width = (output_width % output_block_width == 0) ?
            output_block_width : output_width % output_block_width;
    const int last_block_height = (output_height % output_block_height == 0) ?
            output_block_height : output_height % output_block_height;

    size_t global_size[3] =
      {(size_t)(output_width + output_block_width - 1) / output_block_width,
      (size_t)(output_height + output_block_height - 1) / output_block_height,
      (size_t) num_batches * num_output_maps};

    size_t local_size[3] = {1, 1, static_cast< size_t >(simd_size)};

    optionsString << " -D SIMD_SIZE=" << simd_size
    << " -D filter_qualifier=__global"
    << " -D OUT_BLOCK_WIDTH=" << output_block_width
    << " -D OUT_BLOCK_HEIGHT=" << output_block_height
    << " -D IN_BUFFER_SIZE=" << in_buffer_size
    << " -D LAST_BLOCK_WIDTH=" << last_block_width
    << " -D LAST_BLOCK_HEIGHT=" << last_block_height
    << " -D INPUT_WIDTH=" << padded_width_
    << " -D INPUT_HEIGHT=" << padded_height_
    << " -D INPUT_DEPTH=" << channels_ /group_
    << " -DTOTAL_INPUT_DEPTH_SIZE=" << channels_ /group_
    << " -DTOTAL_OUTPUT_DEPTH=" << channels_ / group_
    << " -DINPUT_START_X=" << 0
    << " -DINPUT_START_Y=" << 0
    << " -DINPUT_START_Z=" << 0
    << " -DOUTPUT_WIDTH=" << output_w_
    << " -DOUTPUT_HEIGHT=" << output_h_
    << " -DFILTER_WIDTH=" << kernel_w_
    << " -DFILTER_HEIGHT=" << kernel_h_
    << " -DNUM_FILTERS=" << M_
    << " -DSTRIDEX=" << stride_w_
    << " -DSTRIDEY=" << stride_h_
    << " -DOWPAD=" << 0
    << " -DOHPAD=" << 0
    << " -DOUT_BUFF_OFFSET=" << 0;

    stringBuilder = optionsString.str();
    const char * options = stringBuilder.c_str();

    ClState& state = Caffe::cl_state();
    int err = state.submit_program(kernel_name_.c_str(),
        &_cl_conv_layer_spatial_start, &_cl_conv_layer_spatial_end, options,
        true);

    ClKernel kernel;
    size_t workgroupSize_used;
    kernel = state.get_kernel(kernel_name_.c_str());
    clGetKernelWorkGroupInfo(kernel, state.get_device(),
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &workgroupSize_used,
        NULL);

    if (workgroupSize_used != simd_size) {
      state.release_program(kernel_name_.c_str());
      return false;
    }

    if (err == CL_SUCCESS || err == true) {
      kernelQueue.push_back(new kernelConfig(kernel_name_,
          global_size, local_size, workItemOutput, false, true, false, false,2));
      return true;
    } else {
      state.release_program(kernel_name_.c_str());
      return false;
    }
  }

  template <>
  bool ConvolutionLayerSpatial<float>::tune_local_size(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      kernelConfig* config) {
    if (config->use_null_local)
      return true;

    float fastestTime = 999999990000000000000000000.0f;
    uint multiplier = 4;
    uint localSize[3] = {1, 1, 1};

    int skip = 0;
    Timer timer;
    timer.initted();
    for (int z = 0; z <= 16; z++) {
      for (int y = 0; y <= 16; y++) {
        for (int x = 0; x <= 16; x++) {
          timer.Start();
          skip = 0;

          if (config->autoTune) {
            config->local_work_size[0] = (multiplier*x == 0) ?
                1 : multiplier*x;
            config->local_work_size[1] = (multiplier*y == 0) ?
                1 : multiplier*y;
            config->local_work_size[2] = (multiplier*z == 0) ?
                1 : multiplier*z;

            if (config->batched_execute) {
              calculate_global_size(2, config->workItem_output,
                  config->local_work_size, config->global_work_size);
            } else {
              calculate_global_size(1, config->workItem_output,
                  config->local_work_size, config->global_work_size);
            }
          }

          if (config->swizzle_weights)
            z = 32;

          int err = 0;
          if (config->batched_execute)
            err = batched_convolve(bottom, top, 0, 1, config);
          else
            err = convolve(bottom, top, 0, 1, config);

          if (err != CL_SUCCESS)
            skip = 1;

          if (skip) {
            timer.Stop();
            break;
          }
          timer.Stop();
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

    dbgPrint(std::cout << "Best local size[" << localSize[0] << "][" <<
        localSize[1] << "]["<< localSize[2] << "]: " << fastestTime <<
        " Kernel_h: " << kernel_h_ << " kernel_w_: " << kernel_w_ <<
        " stride_w: " << stride_w_ << " pad_w_: " << pad_w_ << std::endl);

    if (config->autoTune) {
      for (int li = 0; li < 3; li++)
        config->local_work_size[li] = localSize[li];

      if (config->batched_execute) {
        calculate_global_size(num_, config->workItem_output,
            config->local_work_size, config->global_work_size);
      } else {
        calculate_global_size(1, config->workItem_output,
            config->local_work_size, config->global_work_size);
      }
    }
    return true;
  }

  template <>
  void ConvolutionLayerSpatial<float>::create_convolution_kernel(const vector<Blob<float>*>& bottom,
        const vector<Blob<float>*>& top,int kernelType, int blockWidth, int blockHeight, int blockDepth)
  {
    if(kernelType == 1)
      generate_kernel(bottom,top,blockWidth,blockHeight,blockDepth);
    else if(kernelType == 2)
      setup_IDLF(bottom,top,blockWidth,blockHeight,blockDepth);
    else if(kernelType == 3)
      generate_batched_kernel(bottom,top,blockWidth,blockHeight,blockDepth);
    else if(kernelType == 4)
      create_basic_kernel(bottom,top,blockWidth,blockHeight,blockDepth);

  }

  template <>
  void ConvolutionLayerSpatial<float>::setup_convolution(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
    // Calculate variables used for kernel generation
    const int* kernel_shape_data = this->kernel_shape_.cpu_data();
    kernel_h_ = kernel_shape_data[0];
    kernel_w_ = kernel_shape_data[1];
    height_ = bottom[0]->shape(this->channel_axis_ + 1);
    width_ = bottom[0]->shape(this->channel_axis_ + 2);
    const int* pad_data = this->pad_.cpu_data();
    pad_h_ = pad_data[0];
    pad_w_ = pad_data[1];
    const int* stride_data = this->stride_.cpu_data();
    stride_h_ = stride_data[0];
    stride_w_ = stride_data[1];

    output_h_ = (height_ + 2*pad_h_ - kernel_h_)/stride_h_ +1;
    output_w_ = (width_ + 2*pad_w_ - kernel_w_)/stride_w_ +1;
    padded_width_ = width_ + 2*pad_w_;
    padded_height_ = height_ + 2*pad_h_;

    // Generates static key_
    generate_key();
    // Initializes unique kernel ID
    kernel_uid_ = 0;

    // Creates a verification kernel to verify kernel results
    if (create_verification_kernel(bottom, top) != true)
      exit(-1);

    string outputFile;
    outputFile = "./spatialkernels/" + key_;
    std::ifstream cachedKernel(outputFile.c_str());

    if(cachedKernel)
    {
      int x,y,z,type;
      cachedKernel >> x;
      cachedKernel >> y;
      cachedKernel >> z;
      cachedKernel >> type;
      create_convolution_kernel(bottom, top,type,x,y,z);
      kernel_index_ = kernelQueue.size()-1;
      cachedKernel >> kernelQueue[kernel_index_]->global_work_size[0];
      cachedKernel >> kernelQueue[kernel_index_]->global_work_size[1];
      cachedKernel >> kernelQueue[kernel_index_]->global_work_size[2];
      cachedKernel >> kernelQueue[kernel_index_]->local_work_size[0];
      cachedKernel >> kernelQueue[kernel_index_]->local_work_size[1];
      cachedKernel >> kernelQueue[kernel_index_]->local_work_size[2];
      cachedKernel >> kernelQueue[kernel_index_]->swizzle_weights;
      cachedKernel >> kernelQueue[kernel_index_]->batched_execute;
      cachedKernel >> kernelQueue[kernel_index_]->use_null_local;

      tuned_ = true;
      return;
    }
    else
    {
      create_convolution_kernel(bottom,top,4,1,1,1);

      for(int y = 1; y < 4; y++)
        for(int z = 1; z < 16 && z < M_; z++) {
          create_convolution_kernel(bottom, top,1,4,y,z);
          if(num_ > 1)
            create_convolution_kernel(bottom, top,3,4,y,z);
        }

      create_convolution_kernel(bottom,top,2,3,3,1);
      create_convolution_kernel(bottom,top,2,5,5,1);
      create_convolution_kernel(bottom,top,2,3,4,1);
      create_convolution_kernel(bottom,top,2,6,4,1);
    }

    for (int x = 0; x < kernelQueue.size(); x++)
      tune_local_size(bottom, top, kernelQueue[x]);

    for (int x = 0; x< kernelQueue.size(); x++)
      kernelQueue[x]->executionTime =
          timed_convolve(bottom, top, bottom_index_, num_, kernelQueue[x]);

    int failures = 0;
    while (failures < kernelQueue.size()) {
      int fastestKernel = 0;
      float fastestTime = kernelQueue[0]->executionTime;

      for (int x = 0; x< kernelQueue.size(); x++) {
        if (kernelQueue[x]->executionTime < fastestTime &&
            kernelQueue[x]->tested == false) {
          fastestKernel = x;
          fastestTime = kernelQueue[x]->executionTime;
        }
      }

      // Test fastest kernel
      timed_convolve(bottom, top, bottom_index_, num_,
          kernelQueue[fastestKernel]);
      bool verified = verify_result(bottom, top, bottom_index_, num_,
          kernelQueue[fastestKernel]);

      if (verified == true) {
        kernelQueue[fastestKernel]->verified = true;
        kernel_index_ = fastestKernel;
        break;
      } else {
        kernelQueue[fastestKernel]->tested = true;
        dbgPrint(std::cout << "Kernel " << fastestKernel <<
            " failed verification" << std::endl);
        failures++;
      }
    }

#ifdef dbg
    float convolve_time = timed_convolve(bottom, top, bottom_index_, num_,
        kernelQueue[kernel_index_]);
#else
    timed_convolve(bottom, top, bottom_index_, num_,
        kernelQueue[kernel_index_]);
#endif
    dbgPrint(std::cout << "Convolution Time:" << convolve_time << std::endl);

    bool verification = verify_result(bottom, top, bottom_index_, num_,
        kernelQueue[kernel_index_]);

    if (verification)
      dbgPrint(std::cout << "Kernel passed verification:" << verify_result(
          bottom, top, bottom_index_, num_, kernelQueue[kernel_index_]) <<
          std::endl);
    else
      std::cout << "Verification of kernel was not successful, results for "
          "this layer may not be accurate" << std::endl;

    for (int x = 0; x < kernelQueue.size(); x++) {
      if (x != kernel_index_)
        Caffe::cl_state().release_program(kernelQueue[x]->kernelName.c_str());
    }

    std::ofstream outputKernel;
    outputKernel.open(outputFile.c_str());
    outputKernel << kernelQueue[kernel_index_]->workItem_output[0] << " "
        << kernelQueue[kernel_index_]->workItem_output[1] << " "
        << kernelQueue[kernel_index_]->workItem_output[2] << " "
        << kernelQueue[kernel_index_]->kernelType << " "
        << kernelQueue[kernel_index_]->global_work_size[0] << " "
        << kernelQueue[kernel_index_]->global_work_size[1] << " "
        << kernelQueue[kernel_index_]->global_work_size[2] << " "
        << kernelQueue[kernel_index_]->local_work_size[0] << " "
        << kernelQueue[kernel_index_]->local_work_size[1] << " "
        << kernelQueue[kernel_index_]->local_work_size[2] << " "
        << kernelQueue[kernel_index_]->swizzle_weights << " "
        << kernelQueue[kernel_index_]->batched_execute << " "
        << kernelQueue[kernel_index_]->use_null_local << " ";
    outputKernel.close();

    tuned_ = true;
  }

  template <>
  void ConvolutionLayerSpatial<float>::Forward_gpu(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
    for (int i = 0; i < bottom.size(); ++i) {
      bottom_index_ = i;
      bottom_data = bottom[i]->gpu_data();
      top_data = top[i]->mutable_gpu_data();
      col_data = col_buffer_.mutable_gpu_data();
      weight = this->blobs_[0]->gpu_data();
      swizzled_weights = swizzled_weights_.mutable_gpu_data();

      weight_offset = M_ * K_;
      col_offset = K_ * N_;
      top_offset = M_ * N_;

      bias_ = NULL;

      bias_offset_ = 0;

      if (bias_term_) {
        bias_ = this->blobs_[1]->gpu_data();
      }

      if (!tuned_)
        setup_convolution(bottom, top);

      if (kernelQueue[kernel_index_]->batched_execute)
        OCL_CHECK(batched_convolve(bottom, top, i, num_,
            kernelQueue[kernel_index_]));
      else
        OCL_CHECK(convolve(bottom, top, i, num_,
            kernelQueue[kernel_index_]));


    }
    ClState& state = Caffe::cl_state();
    clFinish(state.get_command_queue());
  }

  template <>
  void ConvolutionLayerSpatial<float>::Backward_gpu(
      const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<float>*>& bottom) {
    const float* weight = NULL;
    float* weight_diff = NULL;

    if (this->param_propagate_down_[0]) {
      weight = this->blobs_[0]->gpu_data();
      weight_diff = this->blobs_[0]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[0]->count(), 0.f, weight_diff);
    }
    float* bias_diff = NULL;
    if (bias_term_ && this->param_propagate_down_[1]) {
      bias_diff = this->blobs_[1]->mutable_gpu_diff();
      caffe_gpu_set(this->blobs_[1]->count(), 0.f, bias_diff);
    }
    const int weight_offset = M_ * K_;
    const int col_offset = K_ * N_;
    const int top_offset = M_ * N_;
    for (int i = 0; i < top.size(); ++i) {
      const float* top_diff = NULL;
      // Bias gradient, if necessary.
      if (bias_term_ && this->param_propagate_down_[1]) {
        top_diff = top[i]->gpu_diff();
        for (int n = 0; n < num_; ++n) {
          caffe_gpu_gemv<float>(CblasNoTrans, num_output_, N_,
              1., top_diff + n * this->top_dim_,
              bias_multiplier_.gpu_data(), 1.,
              bias_diff);
        }
      }
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        if (!top_diff) {
          top_diff = top[i]->gpu_diff();
        }
        float* col_data = col_buffer_.mutable_gpu_data();
        float* col_diff = col_buffer_.mutable_gpu_diff();
        const float* bottom_data = bottom[i]->gpu_data();
        float* bottom_diff = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < num_; ++n) {
          // Since we saved memory in the forward pass by not storing all col
          // data, we will need to recompute them.
          im2col_gpu(bottom_data + n * this->bottom_dim_,
              channels_, height_,
              width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
              stride_h_, stride_w_, col_data);
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            for (int g = 0; g < group_; ++g) {
              caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans, M_, K_, N_,
                  1.f, top_diff + n * this->top_dim_ + top_offset * g,
                  col_data + col_offset * g, 1.f,
                  weight_diff + weight_offset * g);
            }
          }
          // gradient w.r.t. bottom data, if necessary
          if (propagate_down[i]) {
            if (weight == NULL) {
              weight = this->blobs_[0]->gpu_data();
            }
            for (int g = 0; g < group_; ++g) {
              caffe_gpu_gemm<float>(CblasTrans, CblasNoTrans, K_, N_, M_,
                  1.f, weight + weight_offset * g,
                  top_diff + n * this->top_dim_ + top_offset * g,
                  0.f, col_diff + col_offset * g);
            }
            // col2im back to the data
            col2im_gpu(col_diff, channels_, height_, width_,
                kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }
  }

  template <>
  bool ConvolutionLayerSpatial<double>::generate_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int blockWidth, int blockHeight, int blockDepth) {
    NOT_IMPLEMENTED;
    return false;
  }
  template <>
  void ConvolutionLayerSpatial<double>::create_convolution_kernel(const vector<Blob<float>*>& bottom,
         const vector<Blob<float>*>& top,int kernelType, int blockWidth, int blockHeight, int blockDepth)
    {
      NOT_IMPLEMENTED;
      return;
    }
  template <>
  bool ConvolutionLayerSpatial<double>::generate_batched_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int blockWidth, int blockHeight, int blockDepth) {
    NOT_IMPLEMENTED;
    return false;
  }
  template <>
  bool ConvolutionLayerSpatial<double>::setup_IDLF(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top, int blockWidth, int blockHeight, int blockDepth) {
    NOT_IMPLEMENTED;
    return false;
  }

  template <>
  bool ConvolutionLayerSpatial<double>::verify_result(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    NOT_IMPLEMENTED;
    return false;
  }

  template <>
  bool ConvolutionLayerSpatial<double>::create_basic_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int blockWidth, int blockHeight, int blockDepth) {
    NOT_IMPLEMENTED;
    return false;
  }

  template <>
  bool ConvolutionLayerSpatial<double>::create_verification_kernel(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
    NOT_IMPLEMENTED;
    return false;
  }

  template <>
  bool ConvolutionLayerSpatial<double>::tune_local_size(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      kernelConfig* config) {
    NOT_IMPLEMENTED;
    return false;
  }

  template <>
  cl_int ConvolutionLayerSpatial<double>::convolve(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    NOT_IMPLEMENTED;
    return false;
  }

  template <>
  cl_int ConvolutionLayerSpatial<double>::batched_convolve(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    NOT_IMPLEMENTED;
  }

  template <>
  float ConvolutionLayerSpatial<double>::timed_convolve(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top,
      int index, int numImages, kernelConfig* config) {
    NOT_IMPLEMENTED;
    return 0.f;
  }

  template <>
  void ConvolutionLayerSpatial<double>::setup_convolution(
      const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <>
  void ConvolutionLayerSpatial<double>::swizzleWeights(int swizzle_factor) {
    NOT_IMPLEMENTED;
  }

  template<>
  void ConvolutionLayerSpatial<double>::calculate_global_size(int batch,
      int* workItemOutput, size_t* localSizes, size_t* globalSizes) {
    NOT_IMPLEMENTED;
  }

  template <>
  void ConvolutionLayerSpatial<double>::pad_image(int image_offset,
      kernelConfig* config, int imgNum) {
    NOT_IMPLEMENTED;
  }

  template <>
  void ConvolutionLayerSpatial<double>::generate_key() {
    NOT_IMPLEMENTED;
  }
  template <>
  std::string ConvolutionLayerSpatial<double>::generate_unique_key() {
    NOT_IMPLEMENTED;
  }

  template <>
    std::string ConvolutionLayerSpatial<double>::generate_specific_key(int type, int blockWidth, int blockHeight, int blockDepth) {
    NOT_IMPLEMENTED;
  }
  template <>
  void ConvolutionLayerSpatial<double>::Forward_gpu(
      const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <>
  void ConvolutionLayerSpatial<double>::Backward_gpu(
      const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<double>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayerSpatial);

}  // namespace caffe
#endif  // USE_OCL
