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

#ifdef __WIN32__
#include <windows.h>
#include <shlobj.h>
#endif

#include <boost/filesystem.hpp>

// #define TEST_ALL_KERNELS

namespace caffe {

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

  if (!std::is_same<Dtype, double>::value &&
      !getenv("CLCAFFE_IGNORE_PRETUNE")) {
    // Initialize pretuned key value once.
    std::lock_guard<std::mutex>  lock(pretuned_mutex_);
    if (!pretuned_kv_initialized_) {
      InitPretunedKey();
    }
    pretuned_kv_initialized_ = true;
    // If CLCAFFE_TUNING is set or there is no enough pretuned entry,
    // we enable the tuning phase.
    if (getenv("CLCAFFE_TUNING") || pretuned_kv.size() < 1000)
      skip_tuning_phase_ = false;
    else
      skip_tuning_phase_ = true;
  } else {
    skip_tuning_phase_ = false;
  }
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
  winograd_weights_image_ = NULL;

  dwconv_ = (this->num_output_ == this->channels_ && this->channels_ == this->group_);

  if (IsFusedWithEltwise()) {
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

  if (IsFusedWithPReLU()) {
    int blob_index = this->blobs_.size();
    this->blobs_.resize(blob_index + 1);
    if (this->layer_param_.convolution_param().prelu_param().channel_shared()) {
      this->blobs_[blob_index].reset(new Blob<Dtype>(vector<int_tp>(0),
                                            this->device_));
    } else {
      this->blobs_[blob_index].reset(new Blob<Dtype>(vector<int_tp>(1, this->num_output_),
                                            this->device_));
    }
  }

  cache_path_ << Caffe::GetHome();
  if (cache_path_.str() != "") {
    cache_path_ << "/spatialkernels/";
    const boost::filesystem::path& path = cache_path_.str();
    const boost::filesystem::path& dir =
                   boost::filesystem::unique_path(path).string();
    bool hasCacheDir = false;
    if (!boost::filesystem::exists(dir))
      boost::filesystem::create_directories(dir);

    hasCacheDir = boost::filesystem::is_directory(dir);

    if (hasCacheDir != true) {
      std::cout << "Failed to create cache directory : \"" << cache_path_.str()
              << "\"" << std::endl
              << "clCaffe will tune again for next running" << std::endl;
    }
  }

  weight = this->blobs_[0]->gpu_data();
  weight_cpu = static_cast<const Dtype*>(this->blobs_[0]->cpu_data());
  if (this->bias_term_)
    bias_ = this->blobs_[1]->gpu_data();
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (IsFusedWithPReLU()) {
    if (this->layer_param_.convolution_param().prelu_param().channel_shared()) {
      ConvolutionParameter *conv_fuse_param = this->layer_param_.mutable_convolution_param();
      const Dtype* slope_data = this->bias_term_ ?
                                this->blobs_[2]->cpu_data() : this->blobs_[1]->cpu_data();
      switch (conv_fuse_param->fuse_type()) {
        case ConvolutionParameter_FuseType_FUSED_CONV_PRELU:
          conv_fuse_param->set_fuse_type(ConvolutionParameter_FuseType_FUSED_CONV_RELU);
          break;
        case ConvolutionParameter_FuseType_FUSED_CONV_ELTWISE_PRELU:
          conv_fuse_param->set_fuse_type(ConvolutionParameter_FuseType_FUSED_CONV_ELTWISE_RELU);
          break;
        default:
          std::cerr << "Unsupported fuse type: " << conv_fuse_param->fuse_type() << std::endl;
      }
      conv_fuse_param->mutable_relu_param()->set_negative_slope(slope_data[0]);
      negative_slope_ = *slope_data;
    }
  }

  if (IsFusedWithEltwise()) {
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

  if (this->device_->backend() != caffe::BACKEND_OpenCL ||
      std::is_same<Dtype, double>::value ||
      Caffe::mode() != Caffe::GPU)
    return;

  generate_key();
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  CHECK_EQ(IsFusedWithEltwise() == false && IsFusedWithReLU() == false,
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
#define dbgPrint(x) do { if (!skip_tuning_phase_) {(x);} } while(0)
#else
#define dbgPrint(x)
#endif

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::generate_key() {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  std::stringstream keyBuilder;
  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                (this->device_->id());

  pretuned_key_.set(ctx.current_device().max_compute_units(),
                    kernel_w_, kernel_h_, stride_w_, stride_h_,
                    dilation_w_, dilation_h_,
                    this->group_, this->channels_,
                    output_w_, output_h_,
                    this->num_, M_,
                    this->layer_param_.convolution_param().fuse_type());
}

template<typename Dtype>
std::string ConvolutionLayerSpatial<Dtype>::generate_specific_key(
    ConvType type,
    std::vector<int> &kernel_key,
    int_tp blockWidth,
    int_tp blockHeight,
    int_tp blockDepth) {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  std::stringstream keyBuilder;
  keyBuilder << static_cast<int_tp>(type);
  for( uint32_t i = 0; i < kernel_key.size(); i++)
    keyBuilder << "_" << kernel_key[i];

  keyBuilder << "_" << blockWidth
             << "_" << blockHeight
             << "_" << blockDepth;

  if (std::is_same<Dtype, float>::value)
    keyBuilder << "_float";
  else
    keyBuilder << "_half";
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
    viennacl::ocl::program &program = this->device_->template program<Dtype>();
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
void ConvolutionLayerSpatial<Dtype>::winogradWeights() {

  // Simply skip the weight swizzle if we already got a winograd_weights_
  // in test phase and not in auto tuning
  // This requires we always call convolve again with the winner configuration
  // during the auto tuning stage.
  if (tuned_ &&
      winograd_weights_image_ != NULL &&
      this->phase_ == TEST)
    return;

  cl_int err;
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  viennacl::ocl::program &program = this->device_->template program<Dtype>();
  if(!std::is_same<Dtype, double>::value) {
    cl_image_format format;
    cl_image_desc desc;
#ifdef CL_VERSION_1_2
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    if(this->channels_ > 256) {
      desc.image_width = 9*this->channels_/2;
      desc.image_height = 2*ALIGN(this->num_output_, 8);
    } else {
      desc.image_width = 9*this->channels_;
      desc.image_height = ALIGN(this->num_output_, 8);
    }
    desc.image_depth = 1;
    desc.image_array_size = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
#ifdef CL_VERSION_2_0
    desc.mem_object = 0;
#else
    desc.buffer = 0;
#endif
#endif
    if(std::is_same<Dtype, half_float::half>::value)
      format.image_channel_data_type = CL_HALF_FLOAT;
    else
      format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_RGBA;
    winograd_weights_image_ = clCreateImage(
                            ctx.handle().get(),
                            CL_MEM_READ_WRITE,
                            &format,
                            &desc,
                            NULL,
                            &err);
    OCL_CHECK(err);
  }

  viennacl::ocl::kernel &oclk_winograd_weights = this->channels_>256?
        program.get_kernel(CL_KERNEL_SELECT("filter_transform_4x4_v1")):
        program.get_kernel(CL_KERNEL_SELECT("filter_transform_4x4_v0"));
  cl_uint argIdx = 0;
  oclk_winograd_weights.arg(argIdx++, WrapHandle((cl_mem)this->blobs()[0]->gpu_data(), &ctx));
  oclk_winograd_weights.arg(argIdx++, WrapHandle(winograd_weights_image_, &ctx));
  oclk_winograd_weights.arg(argIdx++, (int_tp)this->channels_);
  oclk_winograd_weights.arg(argIdx++, ALIGN((int_tp)this->num_output_, 8));
  size_t global_work_size_U[2] = {(size_t)this->channels_, (size_t)this->num_output_};


  /* Compute data transform. */
  err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
         oclk_winograd_weights.handle().get(),
         2,//work_dim,
         NULL, //global_work_offset
         global_work_size_U, //global_work_size
         NULL, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
  OCL_CHECK(err);

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
  std::vector<int> kernel_key;
  kernel_key.push_back(kernel_w_);
  kernel_key.push_back(kernel_h_);
  kernel_key.push_back(this->channels_ / this->group_);
  kernel_key.push_back(stride_w_);
  kernel_key.push_back(stride_h_);
  kernel_key.push_back(dilation_w_);
  kernel_key.push_back(dilation_h_);
  kernel_key.push_back(this->bias_term_);
  kernel_key.push_back(this->num_output_);
  kernel_key.push_back(IsFusedWithEltwise());
  kernel_key.push_back(IsFusedWithReLU());
  kernel_key.push_back(IsFusedWithPReLU());
  std::string kernelUKey = generate_specific_key(ConvType::BASIC,
                                                 kernel_key,
                                                 blockWidth,
                                                 blockHeight,
                                                 blockDepth);
  int_tp workItemOutput[3];
  workItemOutput[0] = 1;
  workItemOutput[1] = 1;
  workItemOutput[2] = 1;

  std::string kernel_name = "BASIC_";
  kernel_name += kernelUKey.c_str();

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
                << " -D " << kernelDef.c_str()
                << " -DTOTAL_OUTPUT_DEPTH=" << this->num_output_
                << " -D CFMultiNoPadding=" << kernel_name;

  string options = optionsString.str();

  size_t localSize[3] = { 1, 1, 1 };
  size_t globalSize[3];
  calculate_global_size(1, workItemOutput, localSize, globalSize);

  kernelQueue.push_back(
      std::make_shared<kernelConfig>(pretuned_key_, kernel_name, options,
                       globalSize, localSize, workItemOutput,
                       false, true, ConvType::BASIC));

  return true;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::buildKernels(void) {

  int best_kernel_num = 0;
  for (auto &config : kernelQueue) {
    if (config->in_best_kernels)
      ++best_kernel_num;
  }
  // TODO, it's possible to build the candidate in parallel.
  // If we want to do that, we need to give up using viennacl which
  // is not thread-safe.
  auto it = kernelQueue.begin();
  for (; it != kernelQueue.end(); ) {
    stringstream optionsString;
    auto config = *it;
    if (config->built)
      continue;
    // ignore current kernel if not in tuning phase and the
    // kernel is not in any best configs and we have other
    // better candidates.
    if (skip_tuning_phase_ &&
        pretuned_vset.size() > 300 &&
        !config->in_best_kernels &&
        best_kernel_num != 0) {
      it = kernelQueue.erase(it);
      continue;
    }
    ++it;
    optionsString << config->options;
    if (IsFusedWithEltwise()) {
      optionsString << " -DFUSED_CONV_ELTWISE=1";
    }
  
    if (IsFusedWithReLU()) {
      optionsString << " -DFUSED_CONV_RELU=1";
    }
  
    if (IsFusedWithPReLU()) {
      optionsString << " -DFUSED_CONV_PRELU=1";
    }

    optionsString << this->device_->get_extra_build_options();
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
    try {
      submit_conv_spatial_program<Dtype>(&ctx, config->kernelName,
                                         optionsString.str());
      config->built = true;
    } catch (std::exception& e) {
      dbgPrint(std::cout << config->kernelName << std::endl);
      it = kernelQueue.erase(it);
    }
  }
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
    int_tp numImages, std::shared_ptr<kernelConfig>& config) {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
  viennacl::ocl::program &program = ctx.get_program(config->kernelName);
  viennacl::ocl::kernel &kernel = program.get_kernel(config->kernelName);
  cl_int err = CL_SUCCESS;

  const Dtype* slope_data = NULL;
  if (IsFusedWithPReLU())
    slope_data = this->bias_term_ ?
                 this->blobs_[2]->gpu_data() :
                 this->blobs_[1]->gpu_data();
  if (config->kernelType == ConvType::IDLF) {
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
      if (IsFusedWithEltwise())
        kernel.arg(argIdx++, WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
      if (IsFusedWithReLU())
        kernel.arg(argIdx++, fixup_arg_type(negative_slope_));
      if (IsFusedWithPReLU())
        kernel.arg(argIdx++, WrapHandle((cl_mem) slope_data, &ctx));

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
        if (this->bias_term_)
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
             / output_block_h, (size_t) this->num_ * ALIGN(M_, config->workItem_output[2])};

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
  } else if (config->kernelType == ConvType::WINOGRAD) {
    input_transform_blob_.Reshape(this->channels_ * this->num_, (output_h_+3)/4,
                                  (output_w_+3)/4, 36);
    winogradWeights();
    viennacl::ocl::kernel &oclk_data_transform =
      program.get_kernel(config->kernelName+"_data_transform_4x4");

    size_t num_h_tiles = ALIGN(output_h_,4)/4;
    size_t num_w_tiles = ALIGN(output_w_,4)/4;
    int P = num_h_tiles * num_w_tiles;
    const int_tp output_block_w = config->workItem_output[0];
    const int_tp output_block_h = config->workItem_output[1];
    /* Data transform, which calculates V. */
    size_t global_work_size_V[3] = {(size_t)this->channels_*this->num_, (size_t)num_h_tiles, (size_t)num_w_tiles};

    oclk_data_transform.arg(0, WrapHandle((cl_mem)bottom[index]->gpu_data(), &ctx));
    oclk_data_transform.arg(1, WrapHandle((cl_mem)this->input_transform_blob_.mutable_gpu_data(), &ctx));
    oclk_data_transform.arg(2, P);
    oclk_data_transform.arg(3, height_);
    oclk_data_transform.arg(4, width_);
    oclk_data_transform.arg(5, (int)num_h_tiles);
    oclk_data_transform.arg(6, (int)num_w_tiles);

    cl_uint argIdx = 0;
    if (IsFusedWithEltwise())
      kernel.arg(argIdx++, WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
    if (IsFusedWithReLU())
      kernel.arg(argIdx++, fixup_arg_type(negative_slope_));
    if (IsFusedWithPReLU())
      kernel.arg(argIdx++, WrapHandle((cl_mem) slope_data, &ctx));
    kernel.arg(argIdx++, WrapHandle((cl_mem)this->input_transform_blob_.gpu_data(), &ctx));
    kernel.arg(argIdx++, WrapHandle(winograd_weights_image_, &ctx));
    if (this->bias_term_)
      kernel.arg(argIdx++, WrapHandle((cl_mem) bias_, &ctx));
    kernel.arg(argIdx++, WrapHandle((cl_mem)top[index]->mutable_gpu_data(), &ctx));
    kernel.arg(argIdx++, (uint16_t)(ALIGN(output_w_,4)*9));
    kernel.arg(argIdx++, (uint16_t)(ALIGN(output_h_,4)/4));
    kernel.arg(argIdx++, (uint16_t)output_w_);
    kernel.arg(argIdx++, (uint16_t)output_h_);

    /* Compute data transform. */
    err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
         oclk_data_transform.handle().get(),
         3,//work_dim,
         NULL, //global_work_offset
         global_work_size_V, //global_work_size
         NULL, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
    OCL_CHECK(err);
    size_t global_size[3] = {ALIGN((size_t)(this->output_w_ + output_block_w - 1)/output_block_w, 1),
                           ALIGN((size_t)(this->output_h_ + output_block_h - 1)/output_block_h, 1),
                           (size_t) this->num_ * ALIGN(M_, config->workItem_output[2])
                           };

    /* Compute the pre-transformed output. */
    err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
         kernel.handle().get(),
         3,//work_dim,
         NULL, //global_work_offset
         global_size, //global_work_size
         config->local_work_size, //local_work_size
         0, //num_events_in_wait_list
         NULL, //event_wait_list
         NULL //
         );
    OCL_CHECK(err);

    if (err != CL_SUCCESS)
      return err;
  } else if (config->kernelType == ConvType::GEMM_LIKE) {
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
      if (IsFusedWithEltwise())
        kernel.arg(argIdx++, WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
      if (IsFusedWithReLU())
        kernel.arg(argIdx++, fixup_arg_type(negative_slope_));
      if (IsFusedWithPReLU())
        kernel.arg(argIdx++, WrapHandle((cl_mem) slope_data, &ctx));

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
        if (this->bias_term_)
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
        size_t global_size[3] = { gx, gy, size_t(this->num_) };

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
  } else if (config->kernelType == ConvType::DWCONV) {

      cl_uint argIdx = 0;
      if (IsFusedWithEltwise())
        kernel.arg(argIdx++,
                   WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
      if (IsFusedWithReLU())
        kernel.arg(argIdx++, fixup_arg_type(negative_slope_));
      if (IsFusedWithPReLU())
        kernel.arg(argIdx++, WrapHandle((cl_mem) slope_data, &ctx));
      kernel.arg(argIdx++, WrapHandle((cl_mem) bottom_data, &ctx));
      kernel.arg(argIdx++, WrapHandle((cl_mem) weight, &ctx));
      if (this->bias_term_)
        kernel.arg(argIdx++, WrapHandle((cl_mem) bias_, &ctx));
      kernel.arg(argIdx++, WrapHandle((cl_mem) top_data, &ctx));
      kernel.arg(argIdx++, (uint16_t)width_);
      kernel.arg(argIdx++, (uint16_t)height_);
      kernel.arg(argIdx++, (uint16_t)output_w_);
      kernel.arg(argIdx++, (uint16_t)output_h_);

      size_t globalSize[3];
      int blockWidth = config->workItem_output[0];
      int blockHeight = config->workItem_output[1];

      globalSize[0] = ((output_w_ + blockWidth - 1)/blockWidth + 3) & ~3;
      globalSize[1] = ((output_h_ + blockHeight - 1)/blockHeight + 3) & ~3;
      globalSize[2] = this->num_output_ * this->num_;

      size_t localSize[3];
      localSize[0] = 4;
      localSize[1] = 4;
      localSize[2] = 1;

      err = clEnqueueNDRangeKernel(ctx.get_queue().handle().get(),
                                   kernel.handle().get(), 3,
                                   NULL,
                                   globalSize, localSize, 0, NULL,
                                   NULL);

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
        if (IsFusedWithEltwise())
          kernel.arg(argIdx++,
                     WrapHandle((cl_mem) bottom[1]->gpu_data(), &ctx));
        if (IsFusedWithReLU())
          kernel.arg(argIdx++, fixup_arg_type(negative_slope_));
        if (IsFusedWithPReLU())
          kernel.arg(argIdx++, WrapHandle((cl_mem) slope_data, &ctx));

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
    int_tp numImages, std::shared_ptr<kernelConfig>& config) {
  // warm up.
  CHECK_EQ((std::is_same<Dtype, double>::value), false);
  bool saved_tuned = tuned_;
  tuned_ = false;
  convolve(bottom, top, index, this->num_, config);
  Timer timer;
  timer.initted();
  timer.Start();
  cl_int err;
  dbgPrint(std::cout << "Benchmarking kernel: " << config->kernelName
           << std::endl);
  tuned_ = true;
  double out_w = output_w_;
  double out_h = output_h_;
  double out_z = M_;
  double k_w = kernel_w_;
  double k_h = kernel_h_;
  double k_z = this->channels_;
  double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z) * this->num_;

  // For total flops less than 0.5 GOPS, we increase the loop count to 4
  // to increase tuning result stability.
  int loop_cnt = totalFlops > 5e8 ? 1 : 4;
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
  dbgPrint(std::cout << "\tEstimated Gflops:");
  dbgPrint(std::cout << ((totalFlops/1000)/1000)/1000 << std::endl);
  dbgPrint(std::cout << "\tEstimated GFLOPS/S: ");
  dbgPrint(std::cout << (((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime));
  dbgPrint(std::cout << std::endl);
  return elapsedTime;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::verify_result(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp index,
    int_tp numImages,
    const Blob<Dtype> &verify_blob,
    std::shared_ptr<kernelConfig>& config) {

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
  config->executionTime = timed_convolve(bottom, top, index, numImages, config);
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
            if(config->kernelType != ConvType::WINOGRAD) {
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

  if (this->group_ != 1 || ((M_ % 8 != 0) || (M_ % 32 == 24)))
    return false;

  if (blockM != 1 && blockM != 2)
    return false;

  if (blockN != 32)
    return false;

  if (blockK != 8 && blockK != 16)
    return false;

  if (blockK == 16) {
    if ((blockM == 1 && (kernel_w_ > 4 && std::is_same<Dtype, float>::value))
        || M_ % 32 != 0)
      return false;
    if ((blockM == 2 && (kernel_w_ > 4 || std::is_same<Dtype, float>::value))
        || M_ % 32 != 0)
      return false;
  }

  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  std::vector<int> kernel_key;
  kernel_key.push_back(kernel_w_);
  kernel_key.push_back(kernel_h_);
  kernel_key.push_back(pad_w_);
  kernel_key.push_back(pad_h_);
  kernel_key.push_back(this->channels_ / this->group_);
  kernel_key.push_back(stride_w_);
  kernel_key.push_back(stride_h_);
  kernel_key.push_back(dilation_w_);
  kernel_key.push_back(dilation_h_);
  kernel_key.push_back(this->bias_term_);
  kernel_key.push_back(this->M_);
  kernel_key.push_back(this->num_output_);
  kernel_key.push_back(IsFusedWithEltwise());
  kernel_key.push_back(IsFusedWithReLU());
  kernel_key.push_back(IsFusedWithPReLU());

  std::string kernelUKey = generate_specific_key(ConvType::GEMM_LIKE,
                                                 kernel_key,
                                                 blockM,
                                                 blockK,
                                                 blockN);
  int_tp workItemOutput[3] = { blockM, blockK, blockN };

  int_tp simd_size = blockK;
  int_tp num_batches = this->num_;
  int_tp globalWorkSizeDX = blockN;
  int_tp globalWorkSizeDY = blockM;

  std::string kernel_name = "U_GEMM_LIKE_CONV_";
  kernel_name += kernelUKey.c_str();
  if (blockK == 8)
    kernel_name += "_SIMD8";
  else
    kernel_name += "_SIMD16";
  std::stringstream kernelDef;
  kernelDef << "GEMM_LIKE_CONV_" << blockN << "_" << blockM;
  if (blockK == 16)
    kernelDef << "_SIMD16";

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D " << kernelDef.str()
                << " -D Conv_Interleaved=" << kernel_name.c_str();

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
        " -DTILE_N_LAST_DIV8=" << (M_ % 32) / 8 <<
        " -D APPLY_BIAS=" << this->bias_term_;

  optionsString << " -DINPUT_PAD_W=" << pad_w_ << " -DINPUT_PAD_H=" << pad_h_;

  size_t gz = num_batches;
  size_t global_size[3] = { 0, 0, gz };

  size_t local_size[3] = { 1, static_cast<size_t>(simd_size), 1 };
  string options = optionsString.str();

  kernelQueue.push_back(
      std::make_shared<kernelConfig>(pretuned_key_, kernel_name, options,
                       global_size, local_size, workItemOutput,
                       true, false, ConvType::GEMM_LIKE));
  return true;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::create_winograd_conv_kernel(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp simd_size) {
  if (std::is_same<Dtype, double>::value || this->group_ != 1 ||
      this->stride_w_ != 1 || this->stride_h_ != 1 ||
      this->dilation_w_ != 1 || this->dilation_h_ != 1 ||
      ALIGN(M_, 8) > 512 || this->kernel_w_ != 3 || this->kernel_h_ != 3)
    return false;

  if (blockWidth != 4 || blockHeight != 4 || simd_size != 8)
    return false;

  std::stringstream optionsString;
  const int_tp blockDepth = 1;
  std::vector<int> kernel_key;
  kernel_key.push_back(kernel_w_);
  kernel_key.push_back(kernel_h_);
  kernel_key.push_back(pad_w_);
  kernel_key.push_back(pad_h_);
  kernel_key.push_back(this->channels_ / this->group_);
  kernel_key.push_back(stride_w_);
  kernel_key.push_back(stride_h_);
  kernel_key.push_back(dilation_w_);
  kernel_key.push_back(dilation_h_);
  kernel_key.push_back(this->bias_term_);
  kernel_key.push_back(this->M_);
  kernel_key.push_back(this->num_output_);
  kernel_key.push_back(simd_size);
  kernel_key.push_back(IsFusedWithEltwise());
  kernel_key.push_back(IsFusedWithReLU());
  kernel_key.push_back(IsFusedWithPReLU());

  std::string kernelUKey = generate_specific_key(ConvType::WINOGRAD,
                                                 kernel_key,
                                                 blockWidth,
                                                 blockHeight,
                                                 blockDepth);
  int_tp workItemOutput[3] = { blockWidth, blockHeight, simd_size };
  const int_tp num_output_maps = M_;
  int_tp output_block_width = blockWidth;
  int_tp output_block_height = blockHeight;
  int_tp num_batches = this->num_;

  std::string kernel_name = "WINOGRAD_";
  std::string data_transform_name = "_data_transform_4x4";
  kernel_name += kernelUKey;

  kernel_name += "_SIMD8";

  bool is_large_input = 0;
  if(this->channels_ >256)
    is_large_input = 1;

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math "
                << " -D WINOGRAD"
                << " -D winograd_4x4="
                << kernel_name << " -D data_transform_4x4=" << kernel_name+data_transform_name;

  size_t global_size[3] = { 0, 0,
                (size_t) num_batches * ALIGN(num_output_maps, simd_size) };

  size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };
  int tile_x = output_block_width*9;
  int tile_y = output_block_height/4;

  optionsString << " -cl-mad-enable" <<" -D SIMD_SIZE=" << simd_size
                << " -D filter_qualifier=__global" << " -D OUT_BLOCK_WIDTH="
                << output_block_width << " -D OUT_BLOCK_HEIGHT="
                << output_block_height
                << " -D INPUT_DEPTH=" << this->channels_ / this->group_
                << " -D HALF_INPUT_DEPTH=" << this->channels_/this->group_/2
                << " -D IS_LARGE_INPUT=" << is_large_input
                << " -DTOTAL_INPUT_DEPTH_SIZE=" << this->channels_
                << " -DTOTAL_OUTPUT_DEPTH=" << this->num_output_
                << " -D APPLY_BIAS=" << this->bias_term_
                << " -DNUM_FILTERS=" << M_
                << " -DTILE_X=" << tile_x
                << " -DTILE_Y=" << tile_y
                << " -DALIGNED_NUM_FILTERS=" << ALIGN(M_, simd_size)
                << " -DINPUT_PAD_W=" << pad_w_ << " -DINPUT_PAD_H=" << pad_h_;
  // FIXME batch size should not be a macro.
  string options = optionsString.str();
  kernelQueue.push_back(
      std::make_shared<kernelConfig>(pretuned_key_, kernel_name, options,
                       global_size, local_size, workItemOutput,
                       true, false, ConvType::WINOGRAD));
  return true;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::create_dw_conv_kernel(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    int_tp blockWidth,
    int_tp blockHeight, int_tp blockDepth) {
  CHECK_EQ((std::is_same<Dtype, double>::value), false);

  if (!dwconv_)
    return false;

  int_tp output_block_width = blockWidth;
  int_tp output_block_height = blockHeight;
  // Standard spatial setup is done here
  std::stringstream keyBuilder;
  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  std::string kernelDef = "DWCONV";
  std::vector<int> kernel_key;
  kernel_key.push_back(kernel_w_);
  kernel_key.push_back(kernel_h_);
  kernel_key.push_back(pad_w_);
  kernel_key.push_back(pad_h_);
  kernel_key.push_back(this->channels_ / this->group_);
  kernel_key.push_back(stride_w_);
  kernel_key.push_back(stride_h_);
  kernel_key.push_back(dilation_w_);
  kernel_key.push_back(dilation_h_);
  kernel_key.push_back(this->bias_term_);
  kernel_key.push_back(this->num_output_);
  kernel_key.push_back(IsFusedWithEltwise());
  kernel_key.push_back(IsFusedWithReLU());
  kernel_key.push_back(IsFusedWithPReLU());

  std::string kernelUKey = generate_specific_key(ConvType::DWCONV,
                                                 kernel_key,
                                                 blockWidth,
                                                 blockHeight,
                                                 blockDepth);
  int_tp simd_size = 16;
  int_tp workItemOutput[3] = { blockWidth, blockHeight, simd_size };

  std::string kernel_name = "DWCONV_";
  kernel_name += kernelUKey.c_str();

  int_tp wvec_size = (kernel_w_ * kernel_h_ + simd_size -1) / simd_size;
  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math "
                << " -D SIMD_SIZE=" << simd_size
                << " -D KERNEL_SIZE=" << kernel_w_ * kernel_h_
                << " -D KERNEL_W=" << kernel_w_
                << " -D KERNEL_H=" << kernel_h_
                << " -D WVEC_SIZE=" << wvec_size
                << " -D STRIDE_H=" << stride_h_
                << " -DDILATION_X=" << dilation_w_
                << " -DDILATION_Y=" << dilation_h_
                << " -D STRIDE_W=" << stride_w_
                << " -D PAD_W=" << pad_w_
                << " -D PAD_H=" << pad_h_
                << " -D APPLY_BIAS=" << this->bias_term_
                << " -D OUTPUT_Z=" << this->num_output_*this->num_
                << " -D CHANNELS=" << this->num_output_
                << " -D OUT_BLOCK_WIDTH=" << output_block_width
                << " -D OUT_BLOCK_HEIGHT=" << output_block_height
                << " -D " << kernelDef.c_str() << " -D DWCONV="
                << kernel_name;

  string options = optionsString.str();
  size_t localSize[3] = { 1, 1, 1 };
  size_t globalSize[3];
  calculate_global_size(1, workItemOutput, localSize, globalSize);

  kernelQueue.push_back(
      std::make_shared<kernelConfig>(pretuned_key_, kernel_name, options,
                       globalSize, localSize, workItemOutput,
                       false, true, ConvType::DWCONV));

  return true;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::setup_IDLF(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    int_tp blockWidth,
    int_tp blockHeight,
    int_tp simd_size) {

  viennacl::ocl::context &ctx = viennacl::ocl::get_context
                                    (this->device_->id());
  int max_compute_units = ctx.current_device().max_compute_units();
  if (simd_size != 8 && simd_size != 16)
    return false;

  if (simd_size == 8
    && !((this->group_ == 1 || M_ % 8 == 0)))
    return false;
  if (simd_size == 16
    && !(this->group_ == 1 || M_ % 16 == 0))
    return false;
  int width_max, height_max, block_size_max;
  width_max = 14;
  height_max = 14;

  if (blockWidth > width_max)
    return false;
  if (blockHeight > height_max)
    return false;
  block_size_max = 48;
  if (blockWidth > output_w_)
    return false;
  if (blockHeight > output_h_)
    return false;
    // Only when the work items count is less than the device
    // max work items or the M_ is less than 16, we will tune
    // for simd 8.
  if (simd_size == 8
      && M_ >= 16
      && ((this->num_ * M_ * output_w_ * output_h_ /
           static_cast<float>(blockWidth * blockHeight))
           >= max_compute_units * 7 * 16))
    return false;
  int actual_tile_x = kernel_w_ * dilation_w_
                      + (blockWidth - 1) * stride_w_ ;
  int tile_x = (actual_tile_x + 3) & ~3;
  int tile_y = kernel_h_ * dilation_h_ + (blockHeight - 1) * stride_h_;
  if (tile_x > (4 * simd_size))
    return false;
  if ((blockWidth * blockHeight +
      (tile_x * tile_y + simd_size - 1)/ simd_size) > block_size_max)
    return false;
  int tile_y_stride = (4 * simd_size) / tile_x;

  int invec_size = (tile_y + tile_y_stride - 1) / tile_y_stride;
  if (invec_size > 4)
    return false;

  std::stringstream multFunctionBuilder;
  std::string stringBuilder;
  std::stringstream optionsString;
  const int_tp blockDepth = 1;
  std::vector<int> kernel_key;
  kernel_key.push_back(kernel_w_);
  kernel_key.push_back(kernel_h_);
  kernel_key.push_back(pad_w_);
  kernel_key.push_back(pad_h_);
  kernel_key.push_back(this->channels_ / this->group_);
  kernel_key.push_back(stride_w_);
  kernel_key.push_back(stride_h_);
  kernel_key.push_back(dilation_w_);
  kernel_key.push_back(dilation_h_);
  kernel_key.push_back(this->bias_term_);
  kernel_key.push_back(this->num_output_);
  kernel_key.push_back(simd_size);
  kernel_key.push_back(IsFusedWithEltwise());
  kernel_key.push_back(IsFusedWithReLU());
  kernel_key.push_back(IsFusedWithPReLU());
  std::string kernelUKey = generate_specific_key(ConvType::IDLF,
                                                 kernel_key,
                                                 blockWidth,
                                                 blockHeight,
                                                 blockDepth);
  int_tp workItemOutput[3] = { blockWidth, blockHeight, simd_size };
  const int_tp num_output_maps = M_;
  int_tp num_batches = this->num_;

  std::string kernel_name = "IDLF_";
  kernel_name += kernelUKey.c_str();

  if (simd_size == 16)
    kernel_name += "_SIMD16";
  else
    kernel_name += "_SIMD8";

  // Build list of options and defines
  optionsString.str("");
  optionsString << "-cl-fast-relaxed-math " << " -D IDLF"
                << " -D convolve_simd="
                << kernel_name;

  size_t global_size[3] = { 0, 0,
                (size_t) num_batches * ALIGN(num_output_maps, simd_size) };

  size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };

  optionsString << " -D SIMD_SIZE=" << simd_size
                << " -D filter_qualifier=__global" << " -D OUT_BLOCK_WIDTH="
                << blockWidth << " -D OUT_BLOCK_HEIGHT="
                << blockHeight
                << " -D INPUT_DEPTH=" << this->channels_ / this->group_
                << " -DTOTAL_INPUT_DEPTH_SIZE=" << this->channels_
                << " -DTOTAL_OUTPUT_DEPTH=" << this->num_output_
                << " -DKERNEL_WIDTH=" << kernel_w_
                << " -DKERNEL_HEIGHT=" << kernel_h_
                << " -DNUM_FILTERS=" << M_ << " -DSTRIDEX=" << stride_w_
                << " -DSTRIDEY=" << stride_h_ << " -DDILATION_X=" << dilation_w_
                << " -DDILATION_Y=" << dilation_h_
                << " -DTILE_X=" << tile_x
                << " -DTILE_Y=" << tile_y
                << " -DTILE_Y_STRIDE=" << tile_y_stride
                << " -DINVEC_SIZE=" << invec_size
                << " -DALIGNED_NUM_FILTERS=" << ALIGN(M_, simd_size)
                << " -D APPLY_BIAS=" << this->bias_term_;

  optionsString << " -DINPUT_PAD_W=" << pad_w_ << " -DINPUT_PAD_H=" << pad_h_;
  string options = optionsString.str();
  kernelQueue.push_back(
      std::make_shared<kernelConfig>(pretuned_key_, kernel_name, options,
                       global_size, local_size, workItemOutput,
                       true, false, ConvType::IDLF));
  return true;
}

template<typename Dtype>
bool ConvolutionLayerSpatial<Dtype>::create_convolution_kernel(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    ConvType kernelType,
    int_tp blockWidth, int_tp blockHeight,
    int_tp blockDepth) {

  bool ret = false;
  if (kernelType == ConvType::IDLF)
    ret = setup_IDLF(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == ConvType::WINOGRAD)
    ret = create_winograd_conv_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == ConvType::BASIC)
    ret = create_basic_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == ConvType::GEMM_LIKE)
    ret = create_gemm_like_conv_kernel(
            bottom, top, blockWidth, blockHeight, blockDepth);
  else if (kernelType == ConvType::DWCONV)
    ret = create_dw_conv_kernel(
            bottom, top, blockWidth, blockHeight, blockDepth);
  return ret;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::new_best_kernel(
                                 const std::shared_ptr<kernelConfig>& prevKernelConfig,
                                 std::shared_ptr<kernelConfig>& bestConfig,
                                 TunePhase tPhase) {
  bestKernelConfig = bestConfig;
  dbgPrint(std::cout << "Best kernel: " << bestConfig->kernelName << std::endl);
  dbgPrint(std::cout << "Convolution Time:" << bestConfig->executionTime << std::endl);

  PretunedValue v;

  if (tPhase != TUNE_FROM_PRETUNE) {
    v.set(bestKernelConfig->workItem_output[0], bestKernelConfig->workItem_output[1], bestKernelConfig->workItem_output[2],
          static_cast<int>(bestKernelConfig->kernelType));
    pretuned_vset.insert(v);
    if (pretuned_kv.find(pretuned_key_) == pretuned_kv.end()) {
      pretuned_kv[pretuned_key_] = v;
      // When tuning env is not set and we are in tuning phase
      // due to lack of enough pretuned record, we could check
      // whether we already get enough record thus we can skip
      // the tuning phase after that.
      if (pretuned_kv.size() > 1000 &&
          !skip_tuning_phase_ &&
          !getenv("CLCAFFE_TUNING"))
        skip_tuning_phase_ = true;
    }
  }
  best_kernels_.insert(bestKernelConfig->kernelName);

  if (need_swizzle(prevKernelConfig, bestKernelConfig))
    swizzled_weights_ = NULL;

  tuned_ = true;

  if (cache_path_.str() != "" && tPhase == TUNE_FROM_RUNTIME) {
    static std::map<PretunedKey, PretunedValue> saved;
    if (saved.find(pretuned_key_) == saved.end()) {
      //save to .txt file
      string fname = cache_path_.str() + "pretunedkv.txt";
      std::ofstream f(fname.c_str(), ios::app);
      if (f.tellp() == static_cast<std::ofstream::streampos>(0)) {
        f << "Version: " << PretunedVersion << std::endl;
      }
      //f << "{ //" << saved.size() << ":  " << key_ << std::endl;
      f << "{";
      f << pretuned_key_.str() << ",";
      f << v.str();
      f << "}," << std::endl;
      f.close();

      //save to binary file PreTunedBinary
      fname = cache_path_.str() + "pretunedkv.ptb";
      f.open(fname, ios::app|ios::binary);
      if (f.tellp() == static_cast<std::ofstream::streampos>(0)) {
        PretunedMagicNumType magic = PretunedMagicNum;
        PretunedVersionType version = PretunedVersion;
        f.write(static_cast<const char*>(static_cast<const void*>(&magic)), sizeof(PretunedMagicNumType));
        f.write(static_cast<const char*>(static_cast<const void*>(&version)), sizeof(PretunedVersionType));
      } 
      f.write(static_cast<char*>(static_cast<void*>(&pretuned_key_)),sizeof(pretuned_key_));
      f.write(static_cast<char*>(static_cast<void*>(&v)),sizeof(v));
      f.close();

      saved[pretuned_key_] = v;
    }

    string outputFile;
    outputFile = cache_path_.str() + pretuned_key_.key_str();
    std::ifstream cachedKernel(outputFile.c_str());
    if (!cachedKernel) {
      std::ofstream outputKernel;
      outputKernel.open(outputFile.c_str());
      outputKernel << bestKernelConfig->workItem_output[0] << " "
                   << bestKernelConfig->workItem_output[1] << " "
                   << bestKernelConfig->workItem_output[2] << " "
                   << static_cast<int>(bestKernelConfig->kernelType) << " ";
      outputKernel.close();
    } else {
      cachedKernel.close();
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::startTuning(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    TunePhase tPhase,
    const std::shared_ptr<kernelConfig>& prevConfig) {
  
  buildKernels();

  if (tPhase != TUNE_FROM_RUNTIME && kernelQueue.size() == 0) {
    return;
  } else if (kernelQueue.size() == 0) {
    create_basic_kernel(bottom, top, 1, 1, 1);
    buildKernels();
  }
  // Generate validation result.
  Blob<Dtype> verify_blob;
  verify_blob.ReshapeLike(*top[0]);
  Dtype *verify_data = verify_blob.mutable_gpu_data();
  const Dtype *weight_gpu_data = this->blobs_[0]->gpu_data();
  const Dtype *bottom_gpu_data = bottom[0]->gpu_data();
  if (!skip_tuning_phase_) {
    for (int_tp n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_gpu_data, n * this->bottom_dim_,
                             weight_gpu_data, verify_data,
                             n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(verify_data, n * this->top_dim_, bias);
      }
    }
  }
  for (int_tp x = 0; x < kernelQueue.size(); x++) {
    kernelQueue[x]->executionTime = timed_convolve(bottom, top, bottom_index_,
                                                   this->num_, kernelQueue[x]);
#ifdef TEST_ALL_KERNELS
    if (kernelQueue[x]->tested == false) {
      bool verified = skip_tuning_phase_ ? true :
                        verify_result(bottom, top, bottom_index_, this->num_,
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
      bool verified = skip_tuning_phase_ ? true :
                        verify_result(bottom, top, bottom_index_, this->num_,
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

  if (!skip_tuning_phase_) {
    if (verification) {
      dbgPrint(std::cout << "Kernel <" << kernelQueue[kernel_index_]->kernelName
                         << "> passed verification" << std::endl);
    } else {
      dbgPrint(std::cout << "Verification was not successful, "
                         << "fallback to basic kernel" << std::endl);
      create_basic_kernel(bottom, top, 1, 1, 1);
      buildKernels();
      kernel_index_ = kernelQueue.size() - 1;
      verification = skip_tuning_phase_ ? true :
                       verify_result(bottom, top, bottom_index_, this->num_,
                                   verify_blob, kernelQueue[kernel_index_]);
      CHECK_EQ(verification, true) << "Basic kernel failed verification."
                                 << std::endl;
    }
  }
  {
    std::lock_guard<std::mutex>  lock(pretuned_mutex_);
    new_best_kernel(prevConfig, kernelQueue[kernel_index_], tPhase);
    for (int_tp x = 0; x < kernelQueue.size(); x++) {
      if (x != kernel_index_ &&
          best_kernels_.find(kernelQueue[x]->kernelName) == best_kernels_.end()) {
        viennacl::ocl::current_context().delete_program(
            kernelQueue[x]->kernelName);
      }
    }
    kernelQueue.clear();
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::setup_convolution(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  if (this->device_->CheckCapability("cl_intel_subgroups")) {
    int has_dwconv_kernel = 0;
    has_dwconv_kernel = create_convolution_kernel(bottom, top, ConvType::DWCONV, 1, 1, 1);
    has_dwconv_kernel += create_convolution_kernel(bottom, top, ConvType::DWCONV, 2, 2, 1);
    has_dwconv_kernel +=  create_convolution_kernel(bottom, top, ConvType::DWCONV, 4, 4, 1);
     
    /* IDLF kernels are using Intel specific extension which make
       them intel only. */
    if (has_dwconv_kernel && this->group_ > 8)
      return;
    // Create WINOGRAD kernels.
    create_convolution_kernel(bottom, top, ConvType::WINOGRAD, 4, 4, 8);
    //Create GEMM like kernels.
    create_convolution_kernel(bottom, top, ConvType::GEMM_LIKE, 1, 8, 32);
    create_convolution_kernel(bottom, top, ConvType::GEMM_LIKE, 2, 8, 32);
    create_convolution_kernel(bottom, top, ConvType::GEMM_LIKE, 1, 16, 32);
    create_convolution_kernel(bottom, top, ConvType::GEMM_LIKE, 2, 16, 32);
    //Create IDLF kernels.
    for (int simd_size = 8; simd_size <= 16; simd_size += 8) {
      int width_max, height_max;
      width_max = 14;
      height_max = 14;
      for (uint32_t width = width_max; width > 0; width--) {
        for (uint32_t height = height_max; height > 0; height--) {
          create_convolution_kernel(bottom, top, ConvType::IDLF, width, height, simd_size);
          if (kernelQueue.size() >= 8 && height == 2)
            break;
        }
        if (kernelQueue.size() >= 12 && width == 2)
          break;
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->num_ == 0)
    return;

  if (!this->device_->CheckCapability("cl_intel_subgroups")) {
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int_tp i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      // Multi queue execution, all previous work needs to be done first
      this->device_->FinishQueues();
      for (int_tp n = 0; n < this->num_; ++n) {
        // Multi queue execution, go through work queues
        this->device_->SwitchQueue(n);
        this->forward_gpu_gemm(bottom_data, n * this->bottom_dim_, weight,
            top_data, n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data, n * this->top_dim_, bias);
        }
      }
      // Multi queue execution, finish all queues
      this->device_->FinishQueues();
    }
    return;
  }

  // When working as the test net for a training instance,
  // The weights may be changed, thus we have to regenerate
  // the weights.
  if (weight != this->blobs_[0]->gpu_data() ||
      !this->blobs_[0]->data().unique()) {
    weight = this->blobs_[0]->gpu_data();
    weight_cpu = static_cast<const Dtype*>(this->blobs_[0]->cpu_data());
    swizzled_weights_ = NULL;
  }

  if (this->bias_term_)
    bias_ = this->blobs_[1]->gpu_data();

  int bottom_size = bottom.size();
  if (IsFusedWithEltwise() || IsFusedWithPReLU())
    bottom_size = 1;
  for (int_tp i = 0; i < bottom_size; ++i) {
    bottom_index_ = i;
    bottom_data = bottom[i]->gpu_data();
    top_data = top[i]->mutable_gpu_data();
    bias_offset_ = 0;

    if (!tuned_) {
      load_cached_kernels(bottom, top);
      if (tuned_ == false) {
        setup_convolution(bottom, top);
        startTuning(bottom, top, TUNE_FROM_RUNTIME);
        CHECK_EQ(tuned_, true) << "Spatial convolution auto-tuning failed.";
      }
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
bool ConvolutionLayerSpatial<Dtype>::need_swizzle(
       const std::shared_ptr<kernelConfig>& prev,
       const std::shared_ptr<kernelConfig>& cur) {
  // For IDLF kernel or GEMM_LIKE kernel if kernel type changed
  // we need to do swizzle again.
  if ((!prev  || 
       prev->kernelType != cur->kernelType) &&
      (cur->kernelType == ConvType::IDLF ||
       cur->kernelType == ConvType::GEMM_LIKE))
    return true;
  // For IDLF kernel or GEMM_LIKE kernel when the kernel type
  // remains the same, but the simd size changed, we need to
  // do swizzle again.
  if (prev && prev->kernelType == cur->kernelType) {
    if (cur->kernelType == ConvType::IDLF &&
        cur->workItem_output[2] != prev->workItem_output[2])
      return true;
    if (cur->kernelType == ConvType::GEMM_LIKE &&
        cur->workItem_output[1] != prev->workItem_output[1])
      return true;
  }
  return false;
}

template<typename Dtype>
void ConvolutionLayerSpatial<Dtype>::load_cached_kernels(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  if (!this->device_->CheckCapability("cl_intel_subgroups"))
    return;

  std::shared_ptr<kernelConfig> prev_kernel_config = bestKernelConfig;
  if (tuned_) {
    if (prev_kernel_config &&
        prev_kernel_config->key == pretuned_key_) {
      // The same layer, many kernel key should be the same
      // Only the batch size, image size could be different.
      // As all kernels could support varying batch/image size now.
      // No need to build again.
      return;
    }
    tuned_ = false;
    bestKernelConfig = nullptr;
  }

  // Find cached kernel configuration
  if (cache_path_.str() != "") {
    string cacheFile;
    cacheFile = cache_path_.str() + pretuned_key_.key_str();
    std::ifstream cachedKernel(cacheFile.c_str());
    if (cachedKernel) {
      int_tp x, y, z, type;
      cachedKernel >> x;
      cachedKernel >> y;
      cachedKernel >> z;
      cachedKernel >> type;
      cachedKernel.close();
      if (type == 2) {
        if (z == 1)
          z = 16;
        CHECK_EQ(z == 16 || z == 8, true) << "invalid SIMD size" << std::endl;
      }
      if (!create_convolution_kernel(bottom, top, static_cast<ConvType>(type), x, y, z)) {
        dbgPrint(std::cout << "Invalid cache record: ");
        dbgPrint(std::cout << x << ", " << y << ", " << z << "type" << type << std::endl);
      } else {
        startTuning(bottom, top, TUNE_FROM_CACHE, prev_kernel_config);
        if (tuned_)
          return;
      }
    }
  }

  bool exactMatch = false;
  {
    std::lock_guard<std::mutex>  lock(pretuned_mutex_);
    if (pretuned_kv.find(pretuned_key_) != pretuned_kv.end() ||
        skip_tuning_phase_) {
      auto it = pretuned_kv.find(pretuned_key_);
      PretunedValue v;
      if (it == pretuned_kv.end()) {
        auto it0 = pretuned_kv.lower_bound(pretuned_key_);
        std::set<PretunedValue> tuning_set;
        for(int i = 0; it0 != pretuned_kv.end() && i < 12; ++i, --it0) {
          if (tuning_set.find(it0->second) != tuning_set.end()) {
            tuning_set.insert(it0->second);
            create_convolution_kernel(bottom, top,
                                      static_cast<ConvType>(it0->second.kernel_type),
                                      it0->second.block_w, it0->second.block_h, it0->second.block_d);
          }
        }
        auto it1 = pretuned_kv.upper_bound(pretuned_key_);
        for(int i = 0; it1 != pretuned_kv.end() && i < 12; ++i, ++it1) {
          if (tuning_set.find(it1->second) != tuning_set.end())
            continue;
          tuning_set.insert(it1->second);
          create_convolution_kernel(bottom, top,
                                    static_cast<ConvType>(it1->second.kernel_type),
                                    it1->second.block_w, it1->second.block_h, it1->second.block_d);
        }
      } else {
        exactMatch = true;
        v = it->second;
        if (!create_convolution_kernel(bottom, top,
                                  static_cast<ConvType>(v.kernel_type),
                                  v.block_w, v.block_h, v.block_d)) {
          dbgPrint(std::cout << "Invalid cache record: " << v.str() << std::endl);
          return;
        }
      }
    }
  }
  if (kernelQueue.size() > 0)
    startTuning(bottom, top,
                exactMatch ? TUNE_FROM_PRETUNE : TUNE_FROM_RUNTIME,
                prev_kernel_config);
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
template <typename Dtype> std::mutex ConvolutionLayerSpatial<Dtype>::pretuned_mutex_;
template <typename Dtype> std::set<std::string> ConvolutionLayerSpatial<Dtype>::best_kernels_;
template <typename Dtype> bool ConvolutionLayerSpatial<Dtype>::pretuned_kv_initialized_;

}  // namespace caffe
#endif  // USE_INTEL_SPATIAL
