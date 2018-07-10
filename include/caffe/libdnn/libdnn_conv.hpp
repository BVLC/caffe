#ifndef CAFFE_LIBDNN_LIBDNN_CONV_HPP_
#define CAFFE_LIBDNN_LIBDNN_CONV_HPP_

#ifdef USE_LIBDNN

#include "caffe/definitions.hpp"
#include "caffe/common.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/libdnn/libdnn.hpp"
#include "caffe/libdnn/libdnn_tuner.hpp"

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

struct LibDNNConvConfig {
  LibDNNConvConfig() :
    in_shape(3, 1),
    out_shape(3, 1),
    kernel(1, 1),
    pad(0, 0),
    stride(1, 1),
    dilation(1, 1)
  {}
  Device* dev_ptr = nullptr;
  vector<int_tp> in_shape;
  vector<int_tp> out_shape;
  vector<int_tp> kernel;
  vector<int_tp> pad;
  vector<int_tp> stride;
  vector<int_tp> dilation;
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
};

template<typename MItype, typename MOtype>
class LibDNNConv : public LibDNN<MItype, MOtype> {
 public:
  explicit LibDNNConv(LibDNNConvConfig config);
  void Forward(vptr<const MItype> bottom_data, vptr<const MItype> weight,
               const MItype bias_mult, vptr<const MItype> bias, vptr<MOtype> top_data,
               int_tp batch_size,
               const QuantizerValues* const bottom_quant = nullptr,
               const QuantizerValues* const weight_quant = nullptr,
               const QuantizerValues* const bias_mult_quant = nullptr,
               const QuantizerValues* const bias_quant = nullptr,
               const QuantizerValues* const top_quant = nullptr);

  void Backward(bool prop_down_data, bool prop_down_weights,
                vptr<const MOtype> top_data, vptr<const MOtype> top_diff,
                vptr<const MItype> weight, vptr<MItype> weight_diff,
                const MItype bias_mult, vptr<const MItype> bias,
                vptr<MItype> bias_diff, vptr<const MItype> bottom_data,
                vptr<MItype> bottom_diff, int_tp batch_size);

  void Tune(vptr<MOtype> top_data, vptr<MOtype> top_diff,
            vptr<MItype> weight, vptr<MItype> weight_diff,
            const MItype bias_mult, vptr<MItype> bias, vptr<MItype> bias_diff,
            vptr<MItype> bottom_data, vptr<MItype> bottom_diff,
            int_tp batch_size);

  const LibDNNConvConfig get_config();

 protected:
  explicit LibDNNConv(Device* dev_ptr);

  virtual void GenerateKernels();
  virtual bool CompileKernels();
  string string_identifier();
  string generate_fw_defs();
  string generate_bw_defs();
  string generate_wg_defs();
  string generate_fw_kernels(string name);
  string generate_bw_kernels(string name);
  string generate_wg_kernels(string name);

  // Autotuners
  shared_ptr<LibDNNTuner> fw_tuner_;
  shared_ptr<LibDNNTuner> bw_tuner_;
  shared_ptr<LibDNNTuner> wg_tuner_;

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

  vector<int_tp> pad_;
  vector<int_tp> stride_;
  vector<int_tp> dilation_;
  vector<int_tp> kernel_shape_;
  vector<int_tp> im_in_shape_;
  vector<int_tp> im_out_shape_;

  // Compile and method flags
  bool weights_backward_;
  bool bias_backward_;
  bool bias_term_;
  bool skip_range_check_;
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
  Device* dev_ptr = nullptr;
  vector<int_tp> in_shape;
  vector<int_tp> out_shape;
  vector<int_tp> kernel;
  vector<int_tp> pad;
  vector<int_tp> stride;
  vector<int_tp> dilation;
  int_tp group = 1;
  bool bias_term = false;
  bool fast_unsafe_math = false;
  bool weights_backward = true;
  bool bias_backward = true;
  libdnnConvolutionWeightAlgo_t wgalgo =
      LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
  libdnnConvolutionBackwardAlgo_t bwalgo =
      LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
};

template<typename MItype, typename MOtype>
class LibDNNDeconv : public LibDNNConv<MItype, MOtype> {
 public:
  explicit LibDNNDeconv(LibDNNDeconvConfig config);
  void Forward(vptr<const MItype> bottom_data, vptr<const MItype> weight,
               const MItype bias_mult, vptr<const MItype> bias,
               vptr<MOtype> top_data, int_tp batch_size);
  void Backward(bool prop_down_data, bool prop_down_weights,
                vptr<const MOtype> top_data, vptr<const MOtype> top_diff,
                vptr<const MItype> weight, vptr<MItype> weight_diff,
                const MItype bias_mult, vptr<const MItype> bias,
                vptr<MItype> bias_diff, vptr<const MItype> bottom_data,
                vptr<MItype> bottom_diff, int_tp batch_size);

  void Tune(vptr<MOtype> top_data, vptr<MOtype> top_diff,
            vptr<MItype> weight, vptr<MItype> weight_diff,
            const MItype bias_mult,
            vptr<MItype> bias, vptr<MItype> bias_diff,
            vptr<MItype> bottom_data, vptr<MItype> bottom_diff,
            int_tp batch_size);

  const LibDNNDeconvConfig get_config();

 protected:
  virtual void GenerateKernels();
  virtual bool CompileKernels();
  string string_identifier();
  string generate_fw_defs();
  string generate_bw_defs();
  string generate_wg_defs();
  string generate_fw_kernels(string name);
  string generate_bw_kernels(string name);
  string generate_wg_kernels(string name);

  // Bias GEMV sizes
  int_tp M_BG_;
  int_tp MG_BG_;
  int_tp N_BG_;
  int_tp NG_BG_;
  int_tp K_BG_;

 private:
  LibDNNDeconvConfig config_;
};

#ifdef USE_INTEL_SPATIAL
template<typename MItype, typename MOtype>
class LibDNNConvSpatial : public LibDNNConv<MItype, MOtype> {
 public:
  explicit LibDNNConvSpatial(LibDNNConvConfig config);
  void Forward(vptr<const MItype> bottom_data, vptr<const MItype> weight,
               vptr<const MItype> bias,
               vptr<MOtype> top_data, int_tp batch_size);
  void ForwardBenchmark(vptr<const MItype> bottom_data, vptr<const MItype> weight,
               vptr<const MItype> bias,
               vptr<MOtype> top_data, int_tp batch_size);
  void Backward(bool prop_down_data, bool prop_down_weights,
                vptr<const MOtype> top_data, const vptr<MOtype> top_diff,
                vptr<const MItype> weight, MItype* weight_diff,
                vptr<const MItype> bias, vptr<MItype> bias_diff,
                vptr<const MItype> bottom_data, vptr<MItype> bottom_diff,
                int_tp batch_size);

  void Tune(vptr<MOtype> top_data, vptr<MOtype> top_diff,
            vptr<const MItype> weight, MItype* weight_diff,
            vptr<const MItype> bias, vptr<MItype> bias_diff,
            vptr<const MItype> bottom_data, vptr<MItype> bottom_diff,
            int_tp batch_size);

 protected:
  void GenerateKernels();
  string string_identifier();
  string generate_fw_defs();
  string generate_fw_kernels(int_tp kernelType,
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
      for (int_tp X = 0; X < 3; X++) {
        local_work_size[X] = local_size[X];
        global_work_size[X] = global_size[X];
        workItem_output[X] = workItem[X];
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
  void calculate_verify_data(const MItype* bottom,
                             const MItype* w,
                             vptr<const MItype> bias,
                             MItype* verify_data);

  virtual void setup_convolution(const MItype *bottom,
                                 const MItype *top,
                                 const MItype *verify_blob);
  virtual void create_convolution_kernel(const MItype *bottom,
                                         const MItype *top,
                                         int_tp kernelType,
                                         int_tp blockWidth,
                                         int_tp blockHeight,
                                         int_tp blockDepth);
  virtual bool setup_IDLF(const MItype *bottom,
                          const MItype *top, int_tp blockWidth,
                          int_tp blockHeight,
                          int_tp blockDepth);
  virtual bool create_basic_kernel(const MItype *bottom,
                                   const MItype *top,
                                   int_tp blockWidth,
                                   int_tp blockHeight,
                                   int_tp blockDepth);
  virtual bool create_gemm_like_conv_kernel(const MItype *bottom,
                                   const MItype *top,
                                   int_tp blockWidth,
                                   int_tp blockHeight,
                                   int_tp blockDepth);
  virtual cl_int convolve(const MItype *bottom,
                          const MItype *top, int_tp index,
                          int_tp numImages,
                          kernelConfig* config);
  virtual float timed_convolve(const MItype *bottom,
                               const MItype *top, int_tp index,
                               int_tp numImages,
                               kernelConfig* config);
  virtual bool verify_result(const MItype *bottom,
                             const MItype *top, int_tp index,
                             int_tp numImages, const MItype *verify_blob,
                             kernelConfig* config);
  virtual bool tune_local_size(const MItype *bottom,
                               const MItype *top, kernelConfig*);
  virtual void swizzleWeights(const MItype *bottom,
                              const MItype *top,
                              int_tp swizzle_factor,
                              bool interleave = false);
  virtual void generate_key();
  virtual string generate_specific_key(int_tp type, int_tp blockWidth,
  int_tp blockHeight,
                                            int_tp blockDepth);
  virtual void calculate_global_size(int_tp batch, int_tp* workItemOutput,
                                     size_t* localSizes, size_t* globalSizes);
  void load_cached_kernels(const MItype *bottom,
                           const MItype *top);
  void SetUp(const MItype *bottom,
             const MItype *top, caffe::Backend backend);
  void setBufferKernelArg(const MItype *bottom,
                          const MItype *top,
                          viennacl::ocl::kernel *cl_kernel,
                          const cl_uint &argIdx,
                          viennacl::ocl::context *ctx,
                          cl_mem buffer, size_t offset,
                          size_t size, bool readOnly,
                          bool preserved);
  void cleanTmpSubBuffers(const MItype *bottom,
                          const MItype *top);
  std::map<std::tuple<cl_mem, size_t, size_t>, cl_mem> subBufferMap;
  vector<cl_mem> tmpSubBuffers;
  vptr<const MItype> bottom_data_;
  vptr<MOtype> top_data_;
  vptr<MItype> col_data_;
  vptr<const MItype> weight_;
  uint64_t prev_weight_seq_id_;
  vptr<MItype> swizzled_weights;
  int_tp weight_offset;
  int_tp col_offset;
  int_tp top_offset;
  int_tp output_h_, output_w_;
  int_tp padded_height_, padded_width_;
  vptr<const MItype> bias_;
  int_tp bias_offset_;
  int_tp bottom_index_;

  int_tp height_;
  int_tp width_;

  /// M_ is the channel dimension of the output for a single group, which is the
  /// leading dimension of the filter matrix.

  /// K_ is the dimension of an unrolled input for a single group, which is the
  /// leading dimension of the data matrix.

  /// N_ is the spatial dimension of the output, the H X W, which are the last
  /// dimensions of the data and filter matrices.

  bool tuned_;
  bool try_cache_;
  // if need_padding_ is true, we need to pad the input image,
  // otherwise, we don't need to pad it then the convolution kernel
  // need to handle it.
  bool need_padding_;

  string key_;
  string short_key_;
  string kernel_name_;
  stringstream cache_path_;

  MItype *swizzled_weights_;

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
  string options_;

  LibDNNConvConfig config_;
  shared_ptr<LibDNNConv<MItype, MOtype> > libdnn_conv_;
};
#endif

}  // namespace caffe

#endif  // USE_LIBDNN

#endif  // INCLUDE_CAFFE_LIBDNN_LIBDNN_CONV_HPP_
