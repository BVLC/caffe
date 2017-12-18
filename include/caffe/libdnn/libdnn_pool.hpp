#ifndef CAFFE_LIBDNN_LIBDNN_POOL_HPP_
#define CAFFE_LIBDNN_LIBDNN_POOL_HPP_

#ifdef USE_LIBDNN

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/libdnn/libdnn.hpp"

namespace caffe {

typedef enum {
  LIBDNN_POOLING_METHOD_MAX                 = 0,
  LIBDNN_POOLING_METHOD_AVE                 = 1,
  LIBDNN_POOLING_METHOD_STO                 = 2
} libdnnPoolingMethod_t;

typedef enum {
  LIBDNN_POOLING_BW_ALGO_DIRECT             = 0,
  LIBDNN_POOLING_BW_ALGO_ATOMIC             = 1
} libdnnPoolingBackwardAlgo_t;


struct LibDNNPoolConfig {
  LibDNNPoolConfig() :
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
  bool use_top_mask = false;
  bool fast_unsafe_math = false;
  libdnnPoolingMethod_t pool_method = LIBDNN_POOLING_METHOD_MAX;
  libdnnPoolingBackwardAlgo_t bwalgo = LIBDNN_POOLING_BW_ALGO_ATOMIC;
  bool global_pooling = false;
  shared_ptr<QuantizerBase> quant;
};

template<typename MItype, typename MOtype>
class LibDNNPool : public LibDNN<MItype, MOtype> {
 public:
  explicit LibDNNPool(LibDNNPoolConfig config);
  void Forward(vptr<const MItype> bottom_data, vptr<MOtype> top_data,
               int_tp channels, int_tp batch_size,
               bool test_mode, vptr<int_tp> mask,
               vptr<MOtype> top_mask, vptr<MItype> rand_idx);
  void Backward(vptr<const MOtype> top_diff, vptr<MItype> bottom_diff,
                int_tp channels, int_tp batch_size,
                vptr<const int_tp> mask, vptr<const MOtype> top_mask,
                vptr<const MItype> rand_idx);

  const LibDNNPoolConfig get_config();

 protected:
  virtual void GenerateKernels();
  virtual bool CompileKernels();
  string string_identifier();
  string generate_fw_defs();
  string generate_bw_defs();
  string generate_fw_kernels(string name, bool test_mode);
  string generate_fwtr_kernels(string name);
  string generate_fwte_kernels(string name);
  string generate_bw_kernels(string name);

 private:
  LibDNNPoolConfig config_;

  // Autotuners
  shared_ptr<LibDNNTuner> fw_tuner_;
  shared_ptr<LibDNNTuner> bw_tuner_;

  // Pooling parameters
  int_tp num_axes_;

  vector<int_tp> pad_;
  vector<int_tp> stride_;
  vector<int_tp> dilation_;
  vector<int_tp> kernel_shape_;
  vector<int_tp> im_in_shape_;
  vector<int_tp> im_out_shape_;

  // Working memory for stochastic and max pooling
  int_tp* mask_ = nullptr;
  MItype* rand_idx_ = nullptr;

  // Compile and method flags
  bool skip_range_check_;
  libdnnPoolingMethod_t pool_method_;
  libdnnPoolingBackwardAlgo_t bwalgo_;
  bool use_top_mask_;

  shared_ptr<Quantizer<MItype, MItype> > quant_;
};

}  // namespace caffe

#endif  // USE_LIBDNN

#endif  // CAFFE_LIBDNN_LIBDNN_POOL_HPP_
