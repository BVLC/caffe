#ifdef USE_INTEL_SPATIAL
#ifndef CAFFE_CONV_SPATIAL_LAYER_HPP_
#define CAFFE_CONV_SPATIAL_LAYER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template<typename Dtype>
class ConvolutionLayerSpatial : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ConvolutionLayerSpatial(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "Convolution";
  }

  virtual inline int_tp MinBottomBlobs() const {
    return 1;
  }
  virtual inline int_tp MinTopBlobs() const {
    return 1;
  }
  virtual inline bool EqualNumBottomTopBlobs() const {
    return IsFusedWithEltwiseReLU() ? false : true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual inline bool reverse_dimensions() {
    return false;
  }
  virtual void compute_output_shape();

  struct kernelConfig {
    string kernelName;
    float executionTime;
    size_t local_work_size[3];
    size_t global_work_size[3];
    int_tp workItem_output[3];
    bool verified;
    bool autoTune;
    bool tested;
    bool swizzle_weights;
    bool use_null_local;
    int_tp kernelType;

    kernelConfig() {
    }
    kernelConfig(string name, size_t* global_size, size_t* local_size,
    int_tp* workItem,
                 bool tune, bool swizzle, bool null_local,
                 int_tp type = 0) {
      kernelName = name;
      for (int_tp x = 0; x < 3; x++) {
        local_work_size[x] = local_size[x];
        global_work_size[x] = global_size[x];
        workItem_output[x] = workItem[x];
      }
      autoTune = tune;
      swizzle_weights = swizzle;
      use_null_local = null_local;
      verified = false;
      tested = false;
      kernelType = type;
    }
  };

#ifndef CPU_ONLY
#ifdef USE_GREENTEA
  virtual void setup_convolution(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top,
                                 const Blob<Dtype> &verify_blob);
  virtual void create_convolution_kernel(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top,
                                         int_tp kernelType,
                                         int_tp blockWidth,
                                         int_tp blockHeight,
                                         int_tp blockDepth);
  virtual bool setup_IDLF(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top, int_tp blockWidth,
                          int_tp blockHeight,
                          int_tp blockDepth);
  virtual bool create_basic_kernel(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top,
                                   int_tp blockWidth,
                                   int_tp blockHeight,
                                   int_tp blockDepth);
  virtual bool create_gemm_like_conv_kernel(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top,
                                   int_tp blockWidth,
                                   int_tp blockHeight,
                                   int_tp blockDepth);

  virtual cl_int convolve(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top, int_tp index,
                          int_tp numImages,
                          kernelConfig* config);
  virtual float timed_convolve(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top, int_tp index,
                               int_tp numImages,
                               kernelConfig* config);
  virtual bool verify_result(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top, int_tp index,
                             int_tp numImages, const Blob<Dtype> &verify_blob,
                             kernelConfig* config);
  virtual bool tune_local_size(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top, kernelConfig*);
  virtual void swizzleWeights(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top,
                              int_tp swizzle_factor,
                              bool interleave = false);
  virtual void generate_key();
  virtual std::string generate_specific_key(int_tp type, int_tp blockWidth,
  int_tp blockHeight,
                                            int_tp blockDepth);
  virtual void calculate_global_size(int_tp batch, int_tp* workItemOutput,
                                     size_t* localSizes, size_t* globalSizes);
  void load_cached_kernels(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  void SetUp(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top, caffe::Backend backend);
  void setBufferKernelArg(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top,
                          viennacl::ocl::kernel *cl_kernel,
                          const cl_uint &argIdx,
                          viennacl::ocl::context *ctx,
                          cl_mem buffer, size_t offset,
                          size_t size, bool readOnly,
                          bool preserved);
  void cleanTmpSubBuffers(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  std::map<std::tuple<cl_mem, size_t, size_t>, cl_mem> subBufferMap;
  std::vector<cl_mem> tmpSubBuffers;

  bool IsFused() const {
    return (this->layer_param_.convolution_param().fuse_type()
            != ConvolutionParameter_FuseType_UNFUSED);
  }

  bool IsFusedWithMaxPoolAndReLU() const {
    return (this->layer_param_.convolution_param().fuse_type()
            == ConvolutionParameter_FuseType_FUSED_CONV_MAX_POOLING_RELU);
  }

  bool IsFusedWithEltwiseReLU() const {
    return (this->layer_param_.convolution_param().fuse_type()
            == ConvolutionParameter_FuseType_FUSED_CONV_ELTWISE_RELU);
  }

  bool IsFusedWithReLU() const {
    return IsFusedWithEltwiseReLU() ||
           (this->layer_param_.convolution_param().fuse_type()
            == ConvolutionParameter_FuseType_FUSED_CONV_RELU);
  }

#endif
#endif

  const Dtype* bottom_data;
  Dtype* top_data;
  const Dtype* weight;
  const Dtype* weight_cpu;
  Dtype* swizzled_weights_;
  int_tp weight_offset;
  int_tp col_offset;
  int_tp top_offset;
  int_tp output_h_, output_w_;
  const Dtype* bias_;
  int_tp bias_offset_;
  int_tp bottom_index_;

  int_tp kernel_h_;
  int_tp kernel_w_;
  int_tp height_;
  int_tp width_;
  int_tp pad_h_;
  int_tp pad_w_;
  int_tp stride_h_;
  int_tp stride_w_;
  int_tp dilation_h_;
  int_tp dilation_w_;

  /// M_ is the channel dimension of the output for a single group, which is the
  /// leading dimension of the filter matrix.
  int_tp M_;
  /// K_ is the dimension of an unrolled input for a single group, which is the
  /// leading dimension of the data matrix.
  int_tp K_;
  /// N_ is the spatial dimension of the output, the H x W, which are the last
  /// dimensions of the data and filter matrices.
  int_tp N_;

  bool tuned_;

  std::string key_;
  std::string short_key_;
  std::string kernel_name_;
  std::stringstream cache_path_;

  Blob<Dtype> swizzled_weights_blob_;
  Blob<Dtype> bias_multiplier_;

  int_tp kernel_index_;
  int_tp kernel_uid_;

  vector<kernelConfig*> kernelQueue;
  kernelConfig* bestKernelConfig;

  // parameters for fused eltwise layer.
  EltwiseParameter_EltwiseOp op_;
  vector<Dtype> coeffs_;
  Blob<int_tp> max_idx_;
  // parameter for relu
  Dtype negative_slope_;

  bool stable_prod_grad_;
};
}  // namespace caffe

#endif  // CAFFE_CONV_SPATIAL_LAYER_HPP_
#endif  // USE_INTEL_SPATIAL
