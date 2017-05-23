#ifndef CAFFE_GREENTEA_LIBDNN_HPP_
#define CAFFE_GREENTEA_LIBDNN_HPP_

#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "caffe/device.hpp"
#include "caffe/greentea/libdnn_tuner.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#endif  // USE_GREENTEA

#ifdef USE_CUDA
#include "cuda.h"
#include "nvrtc.h"
#endif  // USE_CUDA

namespace caffe {

typedef enum {
  // Stack the batch update into one GEMM block
  // (deterministic, 1 kernel call)
  // Serializes the batch and may therefore under use
  // the GPUs compute units.
  LIBDNN_CONVOLUTION_WG_ALGO_DIRECT        = 0,
  // Use multiple GEMM blocks in parallel and update weights atomically
  // (non deterministic, 1 kernel call, not supported on all devices)
  // Parallelizes the batch and has therefore higher GPU usage.
  LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC        = 1,
  // Use multiple GEMM blocks and an intermediate buffer
  // to reduce weight updates
  // (deterministic, >= 2 kernel calls)
  // Parallelizes the batch and has therefore higher GPU usage.
  // NOT IMPLEMENTED YET
  LIBDNN_CONVOLUTION_WG_ALGO_REDUCTION     = 2
} libdnnConvolutionWeightAlgo_t;

typedef enum {
  // Transform data before GEMM (load, im2col, gemm, store)
  // This method is suitable for convolutions with similar
  // spatial input == output sizes, but can become inefficient
  // if input >> output (with large strides and kernels).
  LIBDNN_CONVOLUTION_BW_ALGO_IM2COL        = 0,
  // Transform data after GEMM (load, gemm, col2im, store)
  // Sometimes faster than im2col method, but uses
  // atomic operations and is not deterministic.
  LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC = 1
} libdnnConvolutionBackwardAlgo_t;

typedef enum {
  LIBDNN_POOLING_METHOD_MAX                 = 0,
  LIBDNN_POOLING_METHOD_AVE                 = 1,
  LIBDNN_POOLING_METHOD_STO                 = 2
} libdnnPoolingMethod_t;

typedef enum {
  LIBDNN_POOLING_BW_ALGO_DIRECT             = 0,
  LIBDNN_POOLING_BW_ALGO_ATOMIC             = 1
} libdnnPoolingBackwardAlgo_t;


template<typename Dtype>
class LibDNN {
 protected:
  explicit LibDNN();
  virtual void GenerateKernels() = 0;
  virtual std::string string_identifier() = 0;
  std::string generate_header();
  std::string generate_common_defs();
  bool CompileKernels();
  void AllocateMemory(void** ptr, uint_tp size, int_tp flags);
  void SetMemory(Dtype* memory, int_tp count, int_tp offset, Dtype value);
#ifdef USE_GREENTEA
  viennacl::ocl::program CompileKernelsOpenCL(viennacl::ocl::context *ctx);
#endif  // USE_GREENTEA
#ifdef USE_CUDA
  nvrtcProgram CompileKernelsCuda();
#endif  // USE_CUDA

  template<class T>
  inline void add_def(std::stringstream& ss,  // NOLINT
      const char* name, T value) {
    ss << "#ifdef " << name << std::endl;
    ss << "#undef " << name << std::endl;
    ss << "#endif" << std::endl;
    if (std::is_same<T, float>::value) {
      ss << "#define " << name << " (float) " << std::setprecision(32) << value
          << std::endl;
    } else if (std::is_same<T, double>::value) {
      ss << "#define " << name << " (double) " << std::setprecision(32) << value
          << std::endl;
    } else {
      ss << "#define " << name << " " << value << std::endl;
    }
  }

  template<class T>
  inline void add_def(std::stringstream& ss,  // NOLINT
      const std::string name, T value) {
    add_def(ss, name.c_str(), value);
  }

  device* dev_ptr_;

#ifdef USE_GREENTEA
  viennacl::ocl::program ocl_program_;
#endif  // USE_GREENTEA

#ifdef USE_CUDA
  nvrtcProgram cuda_program_;
  CUmodule cuda_module_;
#endif  // USE_CUDA

  std::string kernel_;
  bool fast_unsafe_math_;
};

struct LibDNNConvConfig {
  LibDNNConvConfig() :
    in_shape(3, 1),
    out_shape(3, 1),
    kernel(1, 1),
    pad(0, 0),
    stride(1, 1),
    dilation(1, 1)
  {}
  device* dev_ptr = nullptr;
  std::vector<int_tp> in_shape;
  std::vector<int_tp> out_shape;
  std::vector<int_tp> kernel;
  std::vector<int_tp> pad;
  std::vector<int_tp> stride;
  std::vector<int_tp> dilation;
  int_tp group = 1;
  bool bias_term = false;
  bool fast_unsafe_math = false;
  bool weights_backward = true;
  bool bias_backward = true;
  bool phase_test = false;
  libdnnConvolutionWeightAlgo_t wgalgo =
      LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
  libdnnConvolutionBackwardAlgo_t bwalgo =
      LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
  std::function<void*(void**, const uint_tp, const int_tp)>
      memory_allocator = nullptr;
};

template<typename Dtype>
class LibDNNConv : public LibDNN<Dtype> {
 public:
  explicit LibDNNConv();
  explicit LibDNNConv(LibDNNConvConfig config);
  void Forward(const Dtype* bottom_data, const Dtype* weight,
               const Dtype* bias,
               Dtype* top_data, int_tp batch_size);
  void Backward(bool prop_down_data, bool prop_down_weights,
                const Dtype* top_data, const Dtype* top_diff,
                const Dtype* weight, Dtype* weight_diff,
                const Dtype* bias, Dtype* bias_diff,
                const Dtype* bottom_data, Dtype* bottom_diff,
                int_tp batch_size);

  void Tune(Dtype* top_data, Dtype* top_diff,
            Dtype* weight, Dtype* weight_diff,
            Dtype* bias, Dtype* bias_diff,
            Dtype* bottom_data, Dtype* bottom_diff,
            int_tp batch_size);

  const LibDNNConvConfig get_config();

 protected:
  void GenerateKernels();
  std::string string_identifier();
  std::string generate_fw_defs();
  std::string generate_bw_defs();
  std::string generate_wg_defs();
  std::string generate_gemm_core(std::shared_ptr<LibDNNTuner> tuner,
                                 bool dterm);
  std::string generate_accreg_init(std::shared_ptr<LibDNNTuner> tuner,
                                   bool dterm, bool load);
  std::string generate_fw_kernels(std::string name);
  std::string generate_bw_kernels(std::string name);
  std::string generate_wg_kernels(std::string name);

  // Autotuners
  std::shared_ptr<LibDNNTuner> fw_tuner_;
  std::shared_ptr<LibDNNTuner> bw_tuner_;
  std::shared_ptr<LibDNNTuner> wg_tuner_;

  // Forward GEMM sizes
  int_tp M_FW_;
  int_tp MG_FW_;
  int_tp N_FW_;
  int_tp K_FW_;
  int_tp KG_FW_;

  // Backward GEMM sizes
  int_tp M_BW_;
  int_tp MG_BW_;
  int_tp N_BW_;
  int_tp K_BW_;
  int_tp KG_BW_;

  // Weight GEMM sizes
  int_tp M_WG_;
  int_tp MG_WG_;
  int_tp N_WG_;
  int_tp NG_WG_;
  int_tp K_WG_;

  // Convolution parameters
  int_tp num_axes_;
  int_tp fmaps_in_;
  int_tp fmaps_out_;
  int_tp group_;

  std::vector<int_tp> pad_;
  std::vector<int_tp> stride_;
  std::vector<int_tp> dilation_;
  std::vector<int_tp> kernel_shape_;
  std::vector<int_tp> im_in_shape_;
  std::vector<int_tp> im_out_shape_;

  // Compile and method flags
  bool weights_backward_;
  bool bias_backward_;
  bool bias_term_;
  bool skip_range_check_;
  Dtype bias_multiplier_;
  libdnnConvolutionWeightAlgo_t wgalgo_;
  libdnnConvolutionBackwardAlgo_t bwalgo_;

 private:
  LibDNNConvConfig config_;
};

struct LibDNNDeconvConfig {
  LibDNNDeconvConfig() :
    in_shape(3, 1),
    out_shape(3, 1),
    kernel(1, 1),
    pad(0, 0),
    stride(1, 1),
    dilation(1, 1)
  {}
  device* dev_ptr = nullptr;
  std::vector<int_tp> in_shape;
  std::vector<int_tp> out_shape;
  std::vector<int_tp> kernel;
  std::vector<int_tp> pad;
  std::vector<int_tp> stride;
  std::vector<int_tp> dilation;
  int_tp group = 1;
  bool bias_term = false;
  bool fast_unsafe_math = false;
  bool weights_backward = true;
  bool bias_backward = true;
  libdnnConvolutionWeightAlgo_t wgalgo =
      LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
  libdnnConvolutionBackwardAlgo_t bwalgo =
      LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
  std::function<void*(void**, const uint_tp, const int_tp)>
      memory_allocator = nullptr;
};

template<typename Dtype>
class LibDNNDeconv : public LibDNNConv<Dtype> {
 public:
  explicit LibDNNDeconv(LibDNNDeconvConfig config);
  void Forward(const Dtype* bottom_data, const Dtype* weight,
               const Dtype* bias,
               Dtype* top_data, int_tp batch_size);
  void Backward(bool prop_down_data, bool prop_down_weights,
                const Dtype* top_data, const Dtype* top_diff,
                const Dtype* weight, Dtype* weight_diff,
                const Dtype* bias, Dtype* bias_diff,
                const Dtype* bottom_data, Dtype* bottom_diff,
                int_tp batch_size);

  void Tune(Dtype* top_data, Dtype* top_diff,
            Dtype* weight, Dtype* weight_diff,
            Dtype* bias, Dtype* bias_diff,
            Dtype* bottom_data, Dtype* bottom_diff,
            int_tp batch_size);

  const LibDNNDeconvConfig get_config();

 protected:
  void GenerateKernels();
  std::string string_identifier();
  std::string generate_fw_defs();
  std::string generate_bw_defs();
  std::string generate_wg_defs();
  std::string generate_fw_kernels(std::string name);
  std::string generate_bw_kernels(std::string name);
  std::string generate_wg_kernels(std::string name);

  // Bias GEMV sizes
  int_tp M_BG_;
  int_tp MG_BG_;
  int_tp N_BG_;
  int_tp NG_BG_;
  int_tp K_BG_;

 private:
  LibDNNDeconvConfig config_;
};

#ifdef USE_GREENTEA
template<typename Dtype>
class LibDNNConvSpatial : public LibDNNConv<Dtype> {
 public:
  explicit LibDNNConvSpatial(LibDNNConvConfig config);
  void Forward(const Dtype* bottom_data, const Dtype* weight,
               const Dtype* bias,
               Dtype* top_data, int_tp batch_size);
  void ForwardBenchmark(const Dtype* bottom_data, const Dtype* weight,
               const Dtype* bias,
               Dtype* top_data, int_tp batch_size);
  void Backward(bool prop_down_data, bool prop_down_weights,
                const Dtype* top_data, const Dtype* top_diff,
                const Dtype* weight, Dtype* weight_diff,
                const Dtype* bias, Dtype* bias_diff,
                const Dtype* bottom_data, Dtype* bottom_diff,
                int_tp batch_size);

  void Tune(Dtype* top_data, Dtype* top_diff,
            const Dtype* weight, Dtype* weight_diff,
            const Dtype* bias, Dtype* bias_diff,
            const Dtype* bottom_data, Dtype* bottom_diff,
            int_tp batch_size);

 protected:
  void GenerateKernels();
  std::string string_identifier();
  std::string generate_fw_defs();
  std::string generate_fw_kernels(int_tp kernelType,
                                  int_tp blockM,
                                  int_tp blockK,
                                  int_tp blockN);

 private:
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

  void GenerateHelperKernels();
  viennacl::ocl::program compile_fw_kernel();
  void calculate_verify_data(const Dtype* bottom,
                             const Dtype* w,
                             const Dtype* bias,
                             Dtype* verify_data);

  virtual void setup_convolution(const Dtype *bottom,
                                 const Dtype *top,
                                 const Dtype *verify_blob);
  virtual void create_convolution_kernel(const Dtype *bottom,
                                         const Dtype *top,
                                         int_tp kernelType,
                                         int_tp blockWidth,
                                         int_tp blockHeight,
                                         int_tp blockDepth);
  virtual bool setup_IDLF(const Dtype *bottom,
                          const Dtype *top, int_tp blockWidth,
                          int_tp blockHeight,
                          int_tp blockDepth);
  virtual bool create_basic_kernel(const Dtype *bottom,
                                   const Dtype *top,
                                   int_tp blockWidth,
                                   int_tp blockHeight,
                                   int_tp blockDepth);
  virtual bool create_gemm_like_conv_kernel(const Dtype *bottom,
                                   const Dtype *top,
                                   int_tp blockWidth,
                                   int_tp blockHeight,
                                   int_tp blockDepth);
  virtual cl_int convolve(const Dtype *bottom,
                          const Dtype *top, int_tp index,
                          int_tp numImages,
                          kernelConfig* config);
  virtual float timed_convolve(const Dtype *bottom,
                               const Dtype *top, int_tp index,
                               int_tp numImages,
                               kernelConfig* config);
  virtual bool verify_result(const Dtype *bottom,
                             const Dtype *top, int_tp index,
                             int_tp numImages, const Dtype *verify_blob,
                             kernelConfig* config);
  virtual bool tune_local_size(const Dtype *bottom,
                               const Dtype *top, kernelConfig*);
  virtual void swizzleWeights(const Dtype *bottom,
                              const Dtype *top,
                              int_tp swizzle_factor,
                              bool interleave = false);
  virtual void generate_key();
  virtual std::string generate_specific_key(int_tp type, int_tp blockWidth,
  int_tp blockHeight,
                                            int_tp blockDepth);
  virtual void calculate_global_size(int_tp batch, int_tp* workItemOutput,
                                     size_t* localSizes, size_t* globalSizes);
  void load_cached_kernels(const Dtype *bottom,
                           const Dtype *top);
  void SetUp(const Dtype *bottom,
             const Dtype *top, caffe::Backend backend);
  void setBufferKernelArg(const Dtype *bottom,
                          const Dtype *top,
                          viennacl::ocl::kernel *cl_kernel,
                          const cl_uint &argIdx,
                          viennacl::ocl::context *ctx,
                          cl_mem buffer, size_t offset,
                          size_t size, bool readOnly,
                          bool preserved);
  void cleanTmpSubBuffers(const Dtype *bottom,
                          const Dtype *top);
  std::map<std::tuple<cl_mem, size_t, size_t>, cl_mem> subBufferMap;
  std::vector<cl_mem> tmpSubBuffers;
  const Dtype* bottom_data_;
  Dtype* top_data_;
  Dtype* col_data_;
  const Dtype* weight_;
  uint64_t prev_weight_seq_id_;
  Dtype* swizzled_weights;
  int_tp weight_offset;
  int_tp col_offset;
  int_tp top_offset;
  int_tp output_h_, output_w_;
  int_tp padded_height_, padded_width_;
  const Dtype* bias_;
  int_tp bias_offset_;
  int_tp bottom_index_;

  int_tp height_;
  int_tp width_;

  /// M_ is the channel dimension of the output for a single group, which is the
  /// leading dimension of the filter matrix.

  /// K_ is the dimension of an unrolled input for a single group, which is the
  /// leading dimension of the data matrix.

  /// N_ is the spatial dimension of the output, the H x W, which are the last
  /// dimensions of the data and filter matrices.

  bool tuned_;
  bool try_cache_;
  // if need_padding_ is true, we need to pad the input image,
  // otherwise, we don't need to pad it then the convolution kernel
  // need to handle it.
  bool need_padding_;

  std::string key_;
  std::string short_key_;
  std::string kernel_name_;
  std::stringstream cache_path_;

  Dtype *swizzled_weights_;

  int_tp kernel_index_;
  int_tp kernel_uid_;

  vector<kernelConfig*> kernelQueue;
  kernelConfig* bestKernelConfig;

  // derived from BaseConvolutionLayer
  int_tp bottom_dim_;
  int_tp top_dim_;

  int_tp num_;
  int_tp out_spatial_dim_;
  bool is_1x1_;

  int_tp kernel_dim_;
  int_tp in_spatial_dim_;

  int_tp kernelType_;
  int_tp blockM_;
  int_tp blockK_;
  int_tp blockN_;
  std::string options_;

  LibDNNConvConfig config_;
  shared_ptr<LibDNNConv<Dtype> > libdnn_conv_;
};
#endif

struct LibDNNPoolConfig {
  LibDNNPoolConfig() :
    in_shape(3, 1),
    out_shape(3, 1),
    kernel(1, 1),
    pad(0, 0),
    stride(1, 1),
    dilation(1, 1)
  {}
  device* dev_ptr = nullptr;
  std::vector<int_tp> in_shape;
  std::vector<int_tp> out_shape;
  std::vector<int_tp> kernel;
  std::vector<int_tp> pad;
  std::vector<int_tp> stride;
  std::vector<int_tp> dilation;
  bool use_top_mask = false;
  bool fast_unsafe_math = false;
  libdnnPoolingMethod_t pool_method = LIBDNN_POOLING_METHOD_MAX;
  libdnnPoolingBackwardAlgo_t bwalgo = LIBDNN_POOLING_BW_ALGO_ATOMIC;
  bool global_pooling = false;
  std::function<void*(void**, const uint_tp, const int_tp)>
      memory_allocator = nullptr;
};

template<typename Dtype>
class LibDNNPool : public LibDNN<Dtype> {
 public:
  explicit LibDNNPool(LibDNNPoolConfig config);
  void Forward(const Dtype* bottom_data, Dtype* top_data,
               int_tp channels, int_tp batch_size,
               bool test_mode, int_tp* mask,
               Dtype* top_mask, Dtype* rand_idx);
  void Backward(const Dtype* top_diff, Dtype* bottom_diff,
                int_tp channels, int_tp batch_size,
                const int_tp* mask, const Dtype* top_mask,
                const Dtype* rand_idx);

  const LibDNNPoolConfig get_config();

 protected:
  void GenerateKernels();
  std::string string_identifier();
  std::string generate_fw_defs();
  std::string generate_bw_defs();
  std::string generate_fw_kernels(std::string name, bool test_mode);
  std::string generate_fwtr_kernels(std::string name);
  std::string generate_fwte_kernels(std::string name);
  std::string generate_bw_kernels(std::string name);

 private:
  LibDNNPoolConfig config_;

  // Autotuners
  std::shared_ptr<LibDNNTuner> fw_tuner_;
  std::shared_ptr<LibDNNTuner> bw_tuner_;

  // Pooling parameters
  int_tp num_axes_;

  std::vector<int_tp> pad_;
  std::vector<int_tp> stride_;
  std::vector<int_tp> dilation_;
  std::vector<int_tp> kernel_shape_;
  std::vector<int_tp> im_in_shape_;
  std::vector<int_tp> im_out_shape_;

  // Working memory for stochastic and max pooling
  int_tp* mask_ = nullptr;
  Dtype* rand_idx_ = nullptr;

  // Compile and method flags
  bool skip_range_check_;
  libdnnPoolingMethod_t pool_method_;
  libdnnPoolingBackwardAlgo_t bwalgo_;
  bool use_top_mask_;
};


}  // namespace caffe

#endif /* CAFFE_GREENTEA_LIBDNN_HPP_ */
