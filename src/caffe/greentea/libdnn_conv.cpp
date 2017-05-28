#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#ifdef USE_LIBDNN
#include "caffe/device.hpp"
#include "caffe/greentea/libdnn.hpp"
#include "caffe/util/benchmark.hpp"

// #define LIBDNN_DEBUG 1

namespace caffe {

template<typename Dtype>
LibDNNConv<Dtype>::LibDNNConv() {
}

template<typename Dtype>
LibDNNConv<Dtype>::LibDNNConv(LibDNNConvConfig config) {
  config_ = config;
  LibDNN<Dtype>::dev_ptr_ = config.dev_ptr;
  bias_term_ = config.bias_term;
  bias_multiplier_ = config.bias_term ? 1.0 : 0.0;
  LibDNN<Dtype>::fast_unsafe_math_ = config.fast_unsafe_math;
  int_tp dims = config.in_shape.size();
  int_tp spatial_dims = config.kernel.size();

  num_axes_ = spatial_dims;
  fmaps_in_ = config.in_shape[dims - spatial_dims - 1];
  fmaps_out_ = config.out_shape[dims - spatial_dims - 1];
  group_ = config.group;

  wgalgo_ = config.wgalgo;
  bwalgo_ = config.bwalgo;

  weights_backward_ = config.weights_backward;
  bias_backward_ = config.bias_backward;

  skip_range_check_ = true;

  for (int_tp i = 0; i < spatial_dims; ++i) {
    kernel_shape_.push_back(config.kernel[i]);
    pad_.push_back(config.pad[i]);
    if (pad_[i] > 0) {
      skip_range_check_ = false;
    }
    stride_.push_back(config.stride[i]);
    dilation_.push_back(config.dilation[i]);
    im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
    im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
  }

  fw_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());
  bw_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());
  wg_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());

  // Setup tuning parameters

  // Work groups
  for (int id = 0; id < 2; ++id) {
    std::vector<int_tp> workgroup_sizes;
    for (int_tp i = 0; i < LibDNN<Dtype>::dev_ptr_->workgroup_size(id);
            i += 4) {
      workgroup_sizes.push_back(i);
    }
    fw_tuner_->add_set_param < int_tp
        > ("workgroup_size_" + std::to_string(id), 16, workgroup_sizes);
    bw_tuner_->add_set_param < int_tp
        > ("workgroup_size_" + std::to_string(id), 16, workgroup_sizes);
    wg_tuner_->add_set_param < int_tp
        > ("workgroup_size_" + std::to_string(id), 16, workgroup_sizes);
  }

  // TSK
  fw_tuner_->add_range_param<int_tp>("TSK", 8, 1, 32, 1);
  bw_tuner_->add_range_param<int_tp>("TSK", 8, 1, 32, 1);
  wg_tuner_->add_range_param<int_tp>("TSK", 8, 1, 32, 1);

  fw_tuner_->add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);
  bw_tuner_->add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);
  wg_tuner_->add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);

  // WPTM, WPTN
  fw_tuner_->add_range_param<int_tp>("WPTM", 4, 4, 16, 4);
  bw_tuner_->add_range_param<int_tp>("WPTM", 4, 4, 16, 4);
  wg_tuner_->add_range_param<int_tp>("WPTM", 4, 4, 16, 4);

  fw_tuner_->add_set_param<int_tp>("VWM", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  bw_tuner_->add_set_param<int_tp>("VWM", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  wg_tuner_->add_set_param<int_tp>("VWM", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));

  fw_tuner_->add_range_param<int_tp>("WPTN", 4, 4, 16, 4);
  bw_tuner_->add_range_param<int_tp>("WPTN", 4, 4, 16, 4);
  wg_tuner_->add_range_param<int_tp>("WPTN", 4, 4, 16, 4);

  fw_tuner_->add_set_param<int_tp>("VWN", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  bw_tuner_->add_set_param<int_tp>("VWN", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  wg_tuner_->add_set_param<int_tp>("VWN", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));

  // Constraint using TSK, TSM, RTSM and RTSN. Adapt TSK if constraint fails.
  fw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTM", "workgroup_size_1"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  bw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTM", "workgroup_size_1"}), std::vector<
    std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  wg_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTM", "workgroup_size_1"}), std::vector<
    std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  // Constraint using TSK, TSN, RTSN and RTSM. Adapt TSK if constraint fails.
  fw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTN", "workgroup_size_0"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  bw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTN", "workgroup_size_0"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  wg_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTN", "workgroup_size_0"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  fw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "TSK_UNROLL"}),
    std::vector<std::string>({"TSK_UNROLL"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  bw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "TSK_UNROLL"}),
    std::vector<std::string>({"TSK_UNROLL"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  wg_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "TSK_UNROLL"}),
    std::vector<std::string>({"TSK_UNROLL"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  fw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"WPTM", "VWM"}),
    std::vector<std::string>({"WPTM"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  bw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"WPTM", "VWM"}),
    std::vector<std::string>({"WPTM"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  wg_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"WPTM", "VWM"}),
    std::vector<std::string>({"WPTM"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  fw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"WPTN", "VWN"}),
    std::vector<std::string>({"WPTN"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  bw_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"WPTN", "VWN"}),
    std::vector<std::string>({"WPTN"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  wg_tuner_->add_constraint<int64_t>(
    std::vector<std::string>({"WPTN", "VWN"}),
    std::vector<std::string>({"WPTN"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  // pad_A, pad_B
  fw_tuner_->add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  bw_tuner_->add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  wg_tuner_->add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  fw_tuner_->add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);
  bw_tuner_->add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);
  wg_tuner_->add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);

  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    // CUDA needs the vector elements unrolled
    fw_tuner_->add_boolean_param("vector_unroll", true, false);
    bw_tuner_->add_boolean_param("vector_unroll", true, false);
    wg_tuner_->add_boolean_param("vector_unroll", true, false);
  } else {
    // OpenCL does not need the vector elements unrolled, and may
    // save registers by not doing it
    fw_tuner_->add_boolean_param("vector_unroll", true, true);
    bw_tuner_->add_boolean_param("vector_unroll", true, true);
    wg_tuner_->add_boolean_param("vector_unroll", true, true);
  }

  GenerateKernels();
  LibDNN<Dtype>::CompileKernels();
}

template<typename Dtype>
const LibDNNConvConfig LibDNNConv<Dtype>::get_config() {
  return config_;
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::string_identifier() {
  std::stringstream ss;
  ss << "CONV_";
  if (std::is_same<Dtype, double>::value) {
    ss << "double_";
  } else {
    ss << "float_";
  }
  // Device name
  ss << LibDNN<Dtype>::dev_ptr_->name();
  ss << "_";
  ss << num_axes_ << "D_";
  ss << "IN[";
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    ss << im_in_shape_[i];
    if (i < im_in_shape_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_OUT[";
  for (int_tp i = 0; i < im_out_shape_.size(); ++i) {
    ss << im_out_shape_[i];
    if (i < im_out_shape_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_K[";
  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    ss << kernel_shape_[i];
    if (i < kernel_shape_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_S[";
  for (int_tp i = 0; i < stride_.size(); ++i) {
    ss << stride_[i];
    if (i < stride_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_P[";
  for (int_tp i = 0; i < pad_.size(); ++i) {
    ss << pad_[i];
    if (i < pad_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_D[";
  for (int_tp i = 0; i < dilation_.size(); ++i) {
    ss << dilation_[i];
    if (i < dilation_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_";
  ss << "FIN[" << fmaps_in_ << "]_";
  ss << "FOUT[" << fmaps_out_ << "]_";
  ss << "G[" << group_ << "]";
  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_fw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", num_axes_);

  // Groups
  LibDNN<Dtype>::add_def(ss, "v_g", group_);

  int_tp B_off = fmaps_in_;
  int_tp C_off = fmaps_out_;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    B_off *= im_in_shape_[i];
    C_off *= im_out_shape_[i];
  }
  // Input image batch offset
  LibDNN<Dtype>::add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  LibDNN<Dtype>::add_def(ss, "v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
    imsi *= im_in_shape_[i];
    LibDNN<Dtype>::add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
    imso *= im_out_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_imsi", imsi);
  LibDNN<Dtype>::add_def(ss, "v_imso", imso);

  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
  }

  for (int_tp i = 0; i < pad_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i), pad_[i]);
  }

  for (int_tp i = 0; i < stride_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
  }

  for (int_tp i = 0; i < dilation_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
  }

  LibDNN<Dtype>::add_def(ss, "v_fin", fmaps_in_);
  LibDNN<Dtype>::add_def(ss, "v_fout", fmaps_out_);

  if (bias_term_) {
    LibDNN<Dtype>::add_def(ss, "v_bmul", bias_multiplier_);
  }

  MG_FW_ = fmaps_out_;
  M_FW_ = fmaps_out_ / group_;
  N_FW_ = 1;
  KG_FW_ = fmaps_in_;
  K_FW_ = fmaps_in_ / group_;

  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    K_FW_ *= kernel_shape_[i];
    KG_FW_ *= kernel_shape_[i];
    N_FW_ *= im_out_shape_[i];
  }

  // GEMM definitions
  LibDNN<Dtype>::add_def(ss, "MG", MG_FW_);
  LibDNN<Dtype>::add_def(ss, "M", M_FW_);
  LibDNN<Dtype>::add_def(ss, "N", N_FW_);
  LibDNN<Dtype>::add_def(ss, "KG", KG_FW_);
  LibDNN<Dtype>::add_def(ss, "K", K_FW_);

  // Local memory padding
  LibDNN<Dtype>::add_def(ss, "v_pad_A",
                         fw_tuner_->get_param<int>("lmem_pad_A"));
  LibDNN<Dtype>::add_def(ss, "v_pad_B",
                         fw_tuner_->get_param<int>("lmem_pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  LibDNN<Dtype>::add_def(
      ss, "TSM", fw_tuner_->get_param<int>("WPTM")
          * fw_tuner_->get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  LibDNN<Dtype>::add_def(
      ss, "TSN", fw_tuner_->get_param<int>("WPTN")
          * fw_tuner_->get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  LibDNN<Dtype>::add_def(ss, "TSK", fw_tuner_->get_param<int>("TSK"));
  // TSK unrolling
  LibDNN<Dtype>::add_def(ss, "TSK_UNROLL",
                         fw_tuner_->get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  LibDNN<Dtype>::add_def(ss, "WPTM", fw_tuner_->get_param<int>("WPTM"));
  LibDNN<Dtype>::add_def(ss, "VWM", fw_tuner_->get_param<int>("VWM"));
  // The work-per-thread in dimension N
  LibDNN<Dtype>::add_def(ss, "WPTN", fw_tuner_->get_param<int>("WPTN"));
  LibDNN<Dtype>::add_def(ss, "VWN", fw_tuner_->get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  LibDNN<Dtype>::add_def(ss, "RTSM",
                         fw_tuner_->get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  LibDNN<Dtype>::add_def(ss, "RTSN",
                         fw_tuner_->get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  LibDNN<Dtype>::add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  LibDNN<Dtype>::add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  LibDNN<Dtype>::add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_bw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", num_axes_);

  // Groups
  LibDNN<Dtype>::add_def(ss, "v_g", group_);

  int_tp A_off = fmaps_in_ * fmaps_out_;
  int_tp B_off = fmaps_out_;
  int_tp C_off = fmaps_in_;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    A_off *= kernel_shape_[i];
    B_off *= im_out_shape_[i];
    C_off *= im_in_shape_[i];
  }
  // Weight offset (only used for groups)
  LibDNN<Dtype>::add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  LibDNN<Dtype>::add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  LibDNN<Dtype>::add_def(ss, "v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
    imsi *= im_in_shape_[i];
    LibDNN<Dtype>::add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
    imso *= im_out_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_imsi", imsi);
  LibDNN<Dtype>::add_def(ss, "v_imso", imso);

  int_tp v_ks = 1;
  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
    v_ks *= kernel_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_ks", v_ks);

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Set padding to account for padding loss (backward),
    // remove forward padding
    for (int_tp i = 0; i < pad_.size(); ++i) {
      LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i),
              (kernel_shape_[i] - 1) * dilation_[i] - pad_[i]);
    }
  }

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    for (int_tp i = 0; i < pad_.size(); ++i) {
      LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i), pad_[i]);
    }
  }

  for (int_tp i = 0; i < stride_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
  }

  for (int_tp i = 0; i < dilation_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
  }

  LibDNN<Dtype>::add_def(ss, "v_fin", fmaps_in_);
  LibDNN<Dtype>::add_def(ss, "v_fout", fmaps_out_);

  if (bias_term_) {
    LibDNN<Dtype>::add_def(ss, "v_bmul", bias_multiplier_);
  }

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    MG_BW_ = fmaps_in_;
    M_BW_ = fmaps_in_ / group_;
    N_BW_ = 1;
    KG_BW_ = fmaps_out_;
    K_BW_ = fmaps_out_ / group_;

    for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
      K_BW_ *= kernel_shape_[i];
      KG_BW_ *= kernel_shape_[i];
      N_BW_ *= im_in_shape_[i];
    }
  }

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    MG_BW_ = fmaps_in_;
    M_BW_ = fmaps_in_ / group_;
    N_BW_ = 1;
    KG_BW_ = fmaps_out_;
    K_BW_ = fmaps_out_ / group_;

    for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
      MG_BW_ *= kernel_shape_[i];
      M_BW_ *= kernel_shape_[i];
      N_BW_ *= im_out_shape_[i];
    }
  }

  // GEMM definitions
  LibDNN<Dtype>::add_def(ss, "MG", MG_BW_);
  LibDNN<Dtype>::add_def(ss, "M", M_BW_);
  LibDNN<Dtype>::add_def(ss, "N", N_BW_);
  LibDNN<Dtype>::add_def(ss, "KG", KG_BW_);
  LibDNN<Dtype>::add_def(ss, "K", K_BW_);

  // Local memory padding
  LibDNN<Dtype>::add_def(ss, "v_pad_A",
                         bw_tuner_->get_param<int>("lmem_pad_A"));
  LibDNN<Dtype>::add_def(ss, "v_pad_B",
                         bw_tuner_->get_param<int>("lmem_pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  LibDNN<Dtype>::add_def(
      ss,
      "TSM",
      bw_tuner_->get_param<int>("WPTM")
          * bw_tuner_->get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  LibDNN<Dtype>::add_def(
      ss,
      "TSN",
      bw_tuner_->get_param<int>("WPTN")
          * bw_tuner_->get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  LibDNN<Dtype>::add_def(ss, "TSK", bw_tuner_->get_param<int>("TSK"));
  // TSK unrolling
  LibDNN<Dtype>::add_def(ss, "TSK_UNROLL",
                         bw_tuner_->get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  LibDNN<Dtype>::add_def(ss, "WPTM", bw_tuner_->get_param<int>("WPTM"));
  LibDNN<Dtype>::add_def(ss, "VWM", bw_tuner_->get_param<int>("VWM"));
  // The work-per-thread in dimension N
  LibDNN<Dtype>::add_def(ss, "WPTN", bw_tuner_->get_param<int>("WPTN"));
  LibDNN<Dtype>::add_def(ss, "VWN", bw_tuner_->get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  LibDNN<Dtype>::add_def(ss, "RTSM",
                         bw_tuner_->get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  LibDNN<Dtype>::add_def(ss, "RTSN",
                         bw_tuner_->get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  LibDNN<Dtype>::add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  LibDNN<Dtype>::add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  LibDNN<Dtype>::add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_wg_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", num_axes_);

  // Groups
  LibDNN<Dtype>::add_def(ss, "v_g", group_);

  int_tp A_off = fmaps_out_;
  int_tp B_off = fmaps_in_;
  int_tp C_off = fmaps_in_ * fmaps_out_;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    A_off *= im_out_shape_[i];
    B_off *= im_in_shape_[i];
    C_off *= kernel_shape_[i];
  }
  // Output image batch offset
  LibDNN<Dtype>::add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  LibDNN<Dtype>::add_def(ss, "v_B_off", B_off);
  // Weights offset
  LibDNN<Dtype>::add_def(ss, "v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
    imsi *= im_in_shape_[i];
    LibDNN<Dtype>::add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
    imso *= im_out_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_imsi", imsi);
  LibDNN<Dtype>::add_def(ss, "v_imso", imso);

  int_tp v_ks = 1;
  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
    v_ks *= kernel_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_ks", v_ks);

  // Set padding to account for padding loss (backward), remove forward padding
  for (int_tp i = 0; i < pad_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i), pad_[i]);
  }

  for (int_tp i = 0; i < stride_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
  }

  for (int_tp i = 0; i < dilation_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
  }

  LibDNN<Dtype>::add_def(ss, "v_fin", fmaps_in_);
  LibDNN<Dtype>::add_def(ss, "v_fout", fmaps_out_);

  if (bias_term_) {
    LibDNN<Dtype>::add_def(ss, "v_bmul", bias_multiplier_);
  }

  MG_WG_ = fmaps_out_;
  M_WG_ = fmaps_out_ / group_;
  NG_WG_ = fmaps_in_;
  N_WG_ = fmaps_in_ / group_;
  K_WG_ = 1;

  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    N_WG_ *= kernel_shape_[i];
    NG_WG_ *= kernel_shape_[i];
    K_WG_ *= im_out_shape_[i];
  }

  // GEMM definitions
  LibDNN<Dtype>::add_def(ss, "MG", MG_WG_);
  LibDNN<Dtype>::add_def(ss, "M", M_WG_);
  LibDNN<Dtype>::add_def(ss, "N", N_WG_);
  LibDNN<Dtype>::add_def(ss, "NG", NG_WG_);
  LibDNN<Dtype>::add_def(ss, "K", K_WG_);

  // Local memory padding
  LibDNN<Dtype>::add_def(ss, "v_pad_A",
                         wg_tuner_->get_param<int>("lmem_pad_A"));
  LibDNN<Dtype>::add_def(ss, "v_pad_B",
                         wg_tuner_->get_param<int>("lmem_pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  LibDNN<Dtype>::add_def(
      ss,
      "TSM",
      wg_tuner_->get_param<int>("WPTM")
          * wg_tuner_->get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  LibDNN<Dtype>::add_def(
      ss,
      "TSN",
      wg_tuner_->get_param<int>("WPTN")
          * wg_tuner_->get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  LibDNN<Dtype>::add_def(ss, "TSK", wg_tuner_->get_param<int>("TSK"));
  // TSK unrolling
  LibDNN<Dtype>::add_def(ss, "TSK_UNROLL",
                         wg_tuner_->get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  LibDNN<Dtype>::add_def(ss, "WPTM", wg_tuner_->get_param<int>("WPTM"));
  LibDNN<Dtype>::add_def(ss, "VWM", wg_tuner_->get_param<int>("VWM"));
  // The work-per-thread in dimension N
  LibDNN<Dtype>::add_def(ss, "WPTN", wg_tuner_->get_param<int>("WPTN"));
  LibDNN<Dtype>::add_def(ss, "VWN", wg_tuner_->get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  LibDNN<Dtype>::add_def(ss, "RTSM",
                         wg_tuner_->get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  LibDNN<Dtype>::add_def(ss, "RTSN",
                         wg_tuner_->get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  LibDNN<Dtype>::add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  LibDNN<Dtype>::add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  LibDNN<Dtype>::add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_gemm_core(
    std::shared_ptr<LibDNNTuner> tuner, bool dterm) {
  std::stringstream ss;
  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  int rtsn = tuner->get_param<int>("workgroup_size_0");
  int rtsm = tuner->get_param<int>("workgroup_size_1");
  bool unroll = tuner->get_param<bool>("vector_unroll");

  // Temporary registers for A and B
  ss << "Dtype" << vwm << " Areg;" << std::endl;
  ss << "Dtype" << vwn << " Breg[WPTN/VWN];" << std::endl;

  // Loop over the values of a single tile
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp kt=0; kt<TSK; kt+=TSK_UNROLL) {" << std::endl;
  ss << "#pragma unroll " << tuner->get_param<int>("TSK_UNROLL") << std::endl;
  ss << "for (int_tp ku=0; ku<TSK_UNROLL; ++ku) {" << std::endl;
  ss << "int_tp k = kt + ku;" << std::endl;

  // Cache the values of Bsub in registers
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  ss << "int_tp col = tidn + wn*VWN*RTSN;" << std::endl;
  for (int i = 0; i < vwn; ++i) {
    ss << "VEC_" << vwn << "_" << i << "(Breg[wn])"
       << " = Bsub[k][col + " << (i*rtsn)
       << "];" << std::endl;
  }
  ss << "}" << std::endl;

  // Perform the computation
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
  ss << "int_tp row = tidm + wm*VWM*RTSM;" << std::endl;
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Areg)" << " = Asub[row + " << (i*rtsm)
       << "][k];" << std::endl;
  }
  if (dterm) {
    if (unroll) {
      for (int i = 0; i < vwm; ++i) {
        ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) " << "+= VEC_" << vwm
           << "_" << i << "(Areg) * v_bmul;" << std::endl;
      }
    } else {
      ss << "Dreg[wm] += Areg * v_bmul;" << std::endl;
    }
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  if (unroll) {
    for (int n = 0; n < vwn; ++n) {
      for (int m = 0; m < vwm; ++m) {
        ss << "VEC_" << vwn << "_" << n << "(Creg[wm * VWM + " << m << "][wn])"
           << " += VEC_" << vwm << "_" << m << "(Areg)" << " * VEC_" << vwn
           << "_" << n << "(Breg[wn]);" << std::endl;
      }
    }
  } else {
    for (int m = 0; m < vwm; ++m) {
      ss << "Creg[wm * VWM + " << m << "][wn]"
         << " += VEC_"<< vwm << "_" << m << "(Areg)" << " * (Breg[wn]);"
         << std::endl;
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_accreg_init(
    std::shared_ptr<LibDNNTuner> tuner, bool dterm, bool load) {
  std::stringstream ss;

  int vwm = tuner->get_param<int>("VWM");
  int vwn = tuner->get_param<int>("VWN");
  bool unroll = tuner->get_param<bool>("vector_unroll");

  if (dterm) {
    ss << "Dtype" << vwm << " Dreg[WPTM/VWM];" << std::endl;
  }
  ss << "Dtype" << vwn << " Creg[WPTM][WPTN/VWN];" << std::endl;

  // Initialize the accumulation registers
  if (load) {
    // Load
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
      ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
         << std::endl;
      ss << "((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM] = Dptr[globalRow];"
         << std::endl;
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
       << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
    ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
       << std::endl;
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] = "
       << "Cptr[globalRow * N + globalCol];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  } else {
    // Zero init
    if (dterm) {
      ss << "#pragma unroll" << std::endl;
      ss << "for (int_tp wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
      if (unroll) {
        for (int i = 0; i < vwm; ++i) {
          ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) = 0.0;" << std::endl;
        }
      } else {
        ss << "Dreg[wm] = 0.0;" << std::endl;
      }
      ss << "}" << std::endl;
    }
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
    if (unroll) {
      for (int i = 0; i < vwn; ++i) {
        ss << "VEC_" << vwn << "_" << i << "(Creg[wm][wn]) = 0.0;" << std::endl;
      }
    } else {
      ss << "Creg[wm][wn] = 0.0;" << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }
  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;

  int wptn = fw_tuner_->get_param<int>("WPTN");
  int wptm = fw_tuner_->get_param<int>("WPTM");
  int tsk = fw_tuner_->get_param<int>("TSK");
  int rtsn = fw_tuner_->get_param<int>("workgroup_size_0");
  int rtsm = fw_tuner_->get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = fw_tuner_->get_param<int>("VWM");
  int vwn = fw_tuner_->get_param<int>("VWN");
  int lpta = (tsm * tsk) / (rtsm * rtsn);
  int lptb = (tsn * tsk) / (rtsm * rtsn);

  // Forward kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << rtsn << ", " << rtsm << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(vwm, vwn) << ")))" << std::endl;
  ss << "void " + name + "(";
  ss << "__global const Dtype* __restrict im_in, ";
  ss << "__global const Dtype* __restrict wg, ";
  if (bias_term_) {
    ss << "__global const Dtype* __restrict bias, ";
  }
  ss << "__global Dtype* __restrict im_out";
  ss << ") {" << std::endl;

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int_tp tidn = get_local_id(0);" << std::endl;
  // Local col ID (max: RTSN=TSN/WPTN)
  ss << "const int_tp tidm = get_local_id(1);" << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN*get_group_id(0);" << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM*get_group_id(1);" << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << "volatile __local Dtype Asub[" << tsm << "][" << tsk << " + v_pad_A];"
     << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << "volatile __local Dtype Bsub[" << tsk << "][" << tsn << " + v_pad_B];"
     << std::endl;

  // Batch and group
  if (group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (M * K);" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch + group * (M * N);"
       << std::endl;
    if (bias_term_) {
      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);"
         << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch;" << std::endl;
    if (bias_term_) {
      ss << "__global const Dtype* Dptr = bias;" << std::endl;
    }
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << generate_accreg_init(fw_tuner_, false, false);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  /*if (rtsn * rtsm % tsk == 0) {
    ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
    ss << "int_tp row = tid / TSK;" << std::endl;
    ss << "int_tp col = tid % TSK;" << std::endl;
    ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;
    int rowstep = (rtsn * rtsm) / tsk;
    for (int i = 0; i < lpta; ++i) {
      ss << "if ((offM + row + " << i * rowstep << ") < M && tiledIndex < K) {"
         << std::endl;
      ss << "Asub[row+" << i * rowstep << "][col] = Aptr[(offM + row + "
         << i * rowstep << ") * K + tiledIndex];" << std::endl;
      ss << "} else {" << std::endl;  // M-K-Guard
      ss << "Asub[row+" << i * rowstep << "][col] = 0.0;" << std::endl;
      ss << "}";
    }
  } else {*/
    ss << "#pragma unroll 4" << std::endl;
    ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
    ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
    ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
    ss << "int_tp row = id / TSK;" << std::endl;
    ss << "int_tp col = id % TSK;" << std::endl;
    ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    ss << "Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];" << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;  // LPTA
  //  }
  ss << "}" << std::endl;  // Scoping for loading A

  // Load one tile of B into local memory
  ss << "{" << std::endl;  // Scoping for loading B
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = lb * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp col = id % TSN;" << std::endl;
  ss << "int_tp row = id / TSN;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + row;" << std::endl;

  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;
  // Define temporary registers
  for (int_tp i = 0; i < num_axes_; ++i) {
    ss << "int_tp d_iter_" << i << ";" << std::endl;
    ss << "int_tp d_temp_" << i << ";" << std::endl;
  }

  ss << "int_tp imageIndex = offN + col;" << std::endl;
  for (int_tp i = num_axes_ - 1; i >= 0; --i) {
    // Compute d_iter, final tiledIndex becomes input feature map ID
    // Scale d_iter by the dilation factor
    ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
       << ";" << std::endl;
    ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

    // Compute d_temp
    // Scale d_temp by the stride and subtract the padding
    ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ") * v_s_" << i
       << " - v_p_" << i << ";" << std::endl;
    ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
  }

  // Recombine final index, compute in-range
  if (!skip_range_check_) {
    ss << "bool in_range = true;" << std::endl;
  }
  ss << "int_tp d_iter_im;" << std::endl;
  for (int_tp i = 0; i < num_axes_; ++i) {
    // Here, d_temp_ represents the column shift,
    // while d_iter_ is the kernel shift
    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
    ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im;"
       << std::endl;
    if (!skip_range_check_) {
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i << ";"
         << std::endl;
    }
  }

  if (!skip_range_check_) {
    ss << "if (in_range) {" << std::endl;
  }
  // tiledIndex now holds the memory offset for the input image
  ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
  if (!skip_range_check_) {
    ss << "} else {" << std::endl;
    ss << "Bsub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading B

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << generate_gemm_core(fw_tuner_, false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block


  // Store the final results in C
  /*ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN/VWN; ++wn) {" << std::endl;
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
  for (int j = 0; j < vwn; ++j) {
    for (int i = 0; i < vwm; ++i) {
      ss << "Asub[(tidn+wn*RTSN)*VWN + " << j << "][(tidm + wn*RTSN)*VWM + " << i << "] = VEC_" << vwm << "_" << i << "(Creg[wn + " << j << "][wm]);" << std::endl;
    }
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for C registers

  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Store the final results in C
  ss << "{" << std::endl; // Scoping for storing C
  ss << "Dtype" << vwm << " Creg;" << std::endl;
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp lc = 0; lc < ((TSM*TSN-1)/(RTSM*RTSN))/VWM+1; ++lc) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = lc * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = (id / TSN) * VWM;" << std::endl;
  ss << "int_tp col = id % TSN;" << std::endl;
  ss << "int_tp globalRow = offM + row;" << std::endl;
  ss << "int_tp globalCol = offN + col;" << std::endl;
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Creg) = Asub[col][row + " << i << "];" << std::endl;
    ss << "if ((globalRow +" << i << ") < M && globalCol < N) {" << std::endl;
    if (bias_term_) {
      ss << "Cptr[(globalRow +" << i << ") * N + globalCol] = VEC_" << vwm << "_" << i << "(Creg) + Dptr[globalRow +" << i << "];" << std::endl;
    } else {
      ss << "Cptr[(globalRow +" << i << ") * N + globalCol] = VEC_" << vwm << "_" << i << "(Creg);" << std::endl;
    }
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl; // Scoping for storing C*/

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  if (bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  if (bias_term_) {
    ss << "Cptr[globalRow * N + globalCol] = "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + v_bmul * biasval;"
       << std::endl;
  } else {
    ss << "Cptr[globalRow * N + globalCol] = "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  }
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_wg_kernels(std::string name) {
  std::stringstream ss;

  int wptn = wg_tuner_->get_param<int>("WPTN");
  int wptm = wg_tuner_->get_param<int>("WPTM");
  int tsk = wg_tuner_->get_param<int>("TSK");
  int rtsn = wg_tuner_->get_param<int>("workgroup_size_0");
  int rtsm = wg_tuner_->get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = wg_tuner_->get_param<int>("VWM");
  int vwn = wg_tuner_->get_param<int>("VWN");
  int lpta = (tsm * tsk) / (rtsm * rtsn);
  int lptb = (tsn * tsk) / (rtsm * rtsn);

  // Weight kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << rtsn << ", " << rtsm << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(vwm, vwn) << ")))" << std::endl;
  ss << "void " + name + "(";
  ss << "__global const Dtype* __restrict im_in, ";
  ss << "__global const Dtype* __restrict im_out, ";
  if (bias_term_) {
    ss << "__global Dtype* __restrict bias, ";
  }
  ss << "__global Dtype* __restrict wg, ";
  ss << "int_tp batch_size";
  ss << ") {" << std::endl;

  // Thread identifiers
  // Local row ID (max: TSM/WPTM)
  ss << "const int_tp tidn = get_local_id(0);" << std::endl;
  // Local col ID (max: TSN/WPTN)
  ss << "const int_tp tidm = get_local_id(1);" << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN*get_group_id(0);" << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM*get_group_id(1);" << std::endl;

  // Local tile memory
  ss << "volatile __local Dtype Asub[" << tsm << "][" << tsk << " + v_pad_A];"
     << std::endl;
  ss << "volatile __local Dtype Bsub[" << tsk << "][" << tsn << " + v_pad_B];"
     << std::endl;

  // Batch and group
  if (group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (group_ > 1) {
    ss << "__global const Dtype* Aptr = im_out + batch * v_A_off"
       << " + group * (v_A_off / v_g);" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + batch * v_B_off"
       << " + group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = wg + group * (M * N);" << std::endl;
    if (bias_term_) {
      ss << "__global Dtype* Dptr = bias + group * (v_fout / v_g);"
         << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = im_out + batch * v_A_off;" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + batch * v_B_off;" << std::endl;
    ss << "__global Dtype* Cptr = wg;" << std::endl;
    if (bias_term_) {
      ss << "__global Dtype* Dptr = bias;" << std::endl;
    }
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << generate_accreg_init(wg_tuner_, bias_term_,
                             wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT);

  ss << "{" << std::endl;  // Scoping for load & compute block
  if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Additional batch loop, keep the same accumulator for the weight gradient
    ss << "for (batch = 0; batch < batch_size; ++batch) {" << std::endl;
  }

  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id / TSK;" << std::endl;
  ss << "int_tp col = id % TSK;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;

  // Load weights (wg) into Asub
  ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
  ss << "Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "Asub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading A

  // Load one tile of B into local memory
  ss << "{" << std::endl;  // Scoping for loading B
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = lb * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp col = id % TSN;" << std::endl;
  ss << "int_tp row = id / TSN;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + row;" << std::endl;

  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;
  // Define temporary registers
  for (int_tp i = 0; i < num_axes_; ++i) {
    ss << "int_tp d_iter_" << i << ";" << std::endl;
    ss << "int_tp d_temp_" << i << ";" << std::endl;
  }

  ss << "int_tp imageIndex = offN + col;" << std::endl;
  for (int_tp i = num_axes_ - 1; i >= 0; --i) {
    // Compute d_iter, final imageIndex becomes input feature map ID
    // Scale d_iter by the dilation factor
    ss << "d_iter_" << i << " = (imageIndex % v_k_" << i << ") * v_d_" << i
       << ";" << std::endl;
    ss << "imageIndex = imageIndex / v_k_" << i << ";" << std::endl;

    // Compute d_temp
    // Scale d_temp by the stride and subtract the padding
    ss << "d_temp_" << i << " = (tiledIndex % v_imso_" << i << ") * v_s_" << i
       << " - v_p_" << i << ";" << std::endl;
    ss << "tiledIndex = tiledIndex / v_imso_" << i << ";" << std::endl;
  }

  // Recombine final index, compute in-range
  if (!skip_range_check_) {
    ss << "bool in_range = true;" << std::endl;
  }
  ss << "int_tp d_iter_im;" << std::endl;
  for (int_tp i = 0; i < num_axes_; ++i) {
    // Here, d_temp_ represents the column shift,
    // while d_iter_ is the kernel shift
    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
    ss << "imageIndex = imageIndex * v_imsi_" << i << " + d_iter_im;"
       << std::endl;
    if (!skip_range_check_) {
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i << ";"
         << std::endl;
    }
  }

  if (!skip_range_check_) {
    ss << "if (in_range) {" << std::endl;
  }
  // imageIndex now holds the memory offset for the input image
  ss << "Bsub[row][col] = Bptr[imageIndex];" << std::endl;
  if (!skip_range_check_) {
    ss << "} else {" << std::endl;
    ss << "Bsub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading B


  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << generate_gemm_core(wg_tuner_, bias_term_) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;

  if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Shift batch
    ss << "Aptr += v_A_off;" << std::endl;
    ss << "Bptr += v_B_off;" << std::endl;
    // The batch loop
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;  // Scoping for load & compute block


  // Store the final results in C and D
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  if (bias_term_) {
    ss << "if (tidn == 0 && offN == 0 && globalRow < M) {" << std::endl;
    if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
      ss << "Dptr[globalRow] = ((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM];"
         << std::endl;
    }
    if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
      ss << "atomicAdd(&(Dptr[globalRow]), "
         << "((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM]);" << std::endl;
    }
    ss << "}" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    ss << "Cptr[globalRow * N + globalCol] = "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  }
  if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
    ss << "atomicAdd(&(Cptr[globalRow * N + globalCol]), "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]);" << std::endl;
  }
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
std::string LibDNNConv<Dtype>::generate_bw_kernels(std::string name) {
  std::stringstream ss;

  int wptn = bw_tuner_->get_param<int>("WPTN");
  int wptm = bw_tuner_->get_param<int>("WPTM");
  int tsk = bw_tuner_->get_param<int>("TSK");
  int rtsn = bw_tuner_->get_param<int>("workgroup_size_0");
  int rtsm = bw_tuner_->get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = bw_tuner_->get_param<int>("VWM");
  int vwn = bw_tuner_->get_param<int>("VWN");
  int lpta = (tsm * tsk) / (rtsm * rtsn);
  int lptb = (tsn * tsk) / (rtsm * rtsn);

  // Backward kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << rtsn << ", " << rtsm << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(vwm, vwn) << ")))" << std::endl;
  ss << "void " + name + "(";
  ss << "__global const Dtype* __restrict im_out, ";
  ss << "__global const Dtype* __restrict wg, ";
  if (bias_term_) {
    ss << "__global const Dtype* __restrict bias, ";
  }
  ss << "__global Dtype* __restrict im_in";
  ss << ") {" << std::endl;

  // Thread identifiers
  // Local row ID (max: TSM/WPTM)
  ss << "const int_tp tidn = get_local_id(0);" << std::endl;
  // Local col ID (max: TSN/WPTN)
  ss << "const int_tp tidm = get_local_id(1);" << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN*get_group_id(0);" << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM*get_group_id(1);" << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << "volatile __local Dtype Asub[" << tsm << "][" << tsk << " + v_pad_A];"
     << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << "volatile __local Dtype Bsub[" << tsk << "][" << tsn << " + v_pad_B];"
     << std::endl;

  // Batch and group
  if (group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (v_A_off / (v_g * v_g));"
       << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch "
       << "+ group * (v_C_off / v_g);" << std::endl;
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << generate_accreg_init(bw_tuner_, false, false);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
  ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id / TSK;" << std::endl;
  ss << "int_tp col = id % TSK;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Load weights (wg) into Asub, flip fin/fout and inverse spatially
    // Compute kidx and midx, the column and row index of the
    // weights in the original A (weights) matrix
    ss << "int_tp kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;"
       << std::endl;
    ss << "int_tp midx = tiledIndex / v_ks;" << std::endl;
    // Check range of the spatially flipped, fin/fout inverted weights
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    // Access weights with the original (translated) weight indices
    ss << "Asub[row][col] = Aptr[kidx + (v_fin / v_g * v_ks) * midx];"
       << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    // Load weights (wg) into Asub, read A transposed
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    ss << "Asub[row][col] = Aptr[tiledIndex * M + offM + row];" << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading A

  // Load one tile of B into local memory
  ss << "{" << std::endl;  // Scoping for loading B
  ss << "#pragma unroll 4" << std::endl;
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidm * RTSN + tidn;" << std::endl;
  ss << "int_tp id = lb * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp col = id % TSN;" << std::endl;
  ss << "int_tp row = id / TSN;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + row;" << std::endl;

  ss << "if ((offN + col) < N && tiledIndex < K) {" << std::endl;

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Load from B with im2col transformation

    // Define temporary registers
    for (int_tp i = 0; i < num_axes_; ++i) {
      ss << "int_tp d_iter_" << i << ";" << std::endl;
      ss << "int_tp d_temp_" << i << ";" << std::endl;
    }

    // Compute in-range
    ss << "bool in_range = true;" << std::endl;

    ss << "int_tp imageIndex = offN + col;" << std::endl;
    for (int_tp i = num_axes_ - 1; i >= 0; --i) {
      // Compute d_iter, final tiledIndex becomes input feature map ID
      // Scale d_iter by the dilation factor
      ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

      // Compute d_temp
      // Subtract the padding from d_temp, note v_p_i can be negative
      ss << "d_temp_" << i << " = (imageIndex % v_imsi_" << i << ")"
         << " - v_p_" << i << ";" << std::endl;
      ss << "imageIndex = imageIndex / v_imsi_" << i << ";" << std::endl;
    }

    ss << "int_tp d_iter_im;" << std::endl;
    for (int_tp i = 0; i < num_axes_; ++i) {
      // Here, d_temp_ represents the column shift,
      // while d_iter_ is the kernel shift
      ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
      ss << "tiledIndex = tiledIndex * v_imso_" << i << " + d_iter_im / v_s_"
         << i << ";" << std::endl;
      // In range: Not before or after actual image data
      // and not between image strides
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imso_" << i
         << " * v_s_" << i << " && d_iter_im % v_s_" << i << " == 0;"
         << std::endl;
    }

    ss << "if (in_range) {" << std::endl;
    // tiledIndex now holds the memory offset for the input image
    ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
    ss << "} else {" << std::endl;
    // Out of B's image dimensions
    ss << "Bsub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    // Load from B without transformation
    ss << "Bsub[row][col] = Bptr[(offN + col) + tiledIndex * N];" << std::endl;
  }

  ss << "} else {" << std::endl;
  // Out of B's matrix dimensions
  ss << "Bsub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading B

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << generate_gemm_core(bw_tuner_, false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;" <<std::endl;
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;" << std::endl;

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "Cptr[globalRow * N + globalCol] = ";
    ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
    ss << "}" << std::endl;
  }

  if (bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    // Define temporary registers
    for (int_tp i = 0; i < num_axes_; ++i) {
      ss << "int_tp d_iter_" << i << ";" << std::endl;
      ss << "int_tp d_temp_" << i << ";" << std::endl;
    }

    // Compute in-range
    ss << "bool in_range = true;" << std::endl;
    ss << "int_tp tiledIndex = globalRow;" << std::endl;
    ss << "int_tp imageIndex = globalCol;" << std::endl;
    for (int_tp i = num_axes_ - 1; i >= 0; --i) {
      // Compute d_iter, final tiledIndex becomes input feature map ID
      // Scale d_iter by the dilation factor
      ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

      // Compute d_temp
      // Scale d_temp by the stride
      ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ") * v_s_" << i
         << ";" << std::endl;
      ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
    }

    ss << "in_range &= tiledIndex < v_fin && globalRow < M && globalCol < N;"
       << std::endl;
    ss << "int_tp d_iter_im;" << std::endl;
    for (int_tp i = 0; i < num_axes_; ++i) {
      // Here, d_temp_ represents the column shift,
      // while d_iter_ is the kernel shift
      // d_iter_im is the combined offset in the current dimension i
      ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << " - v_p_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im;"
         << std::endl;
      // In range: Not before or after actual image data
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i << ";"
         << std::endl;
    }

    ss << "if (in_range) {" << std::endl;
    ss << "atomicAdd(&(Cptr[tiledIndex]), "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]);" << std::endl;
    ss << "}" << std::endl;
  }

  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
void LibDNNConv<Dtype>::GenerateKernels() {
  std::stringstream ss;

  ss << LibDNN<Dtype>::generate_header();
  ss << generate_fw_defs();
  ss << generate_fw_kernels("conv_forward");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("conv_backward");
  ss << generate_wg_defs();
  ss << generate_wg_kernels("conv_weights");

  // Write complete kernel string
  LibDNN<Dtype>::kernel_ = ss.str();
}

template<typename Dtype>
void LibDNNConv<Dtype>::Forward(const Dtype* bottom_data, const Dtype* weight,
                                const Dtype* bias, Dtype* top_data,
                                int_tp batch_size) {
  int fw_wptn = fw_tuner_->get_param<int>("WPTN");
  int fw_wptm = fw_tuner_->get_param<int>("WPTM");
  int fw_wgs0 = fw_tuner_->get_param<int>("workgroup_size_0");
  int fw_wgs1 = fw_tuner_->get_param<int>("workgroup_size_1");
  int fw_div_N = fw_wptn * fw_wgs0;
  int fw_div_M = fw_wptm * fw_wgs1;

#ifdef USE_GREENTEA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
    viennacl::ocl::kernel &kernel =
        LibDNN<Dtype>::ocl_program_.get_kernel("conv_forward");
    viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

    kernel.local_work_size(0, fw_wgs0);
    kernel.local_work_size(1, fw_wgs1);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((this->N_FW_ - 1) / fw_div_N + 1) * fw_wgs0);
    kernel.global_work_size(1, ((this->M_FW_ - 1) / fw_div_M + 1) * fw_wgs1);
    kernel.global_work_size(2, batch_size * group_);

    // for (int i = 0; i < 3; ++i) {
    // std::cout << i << "; local: "
    //           << kernel.local_work_size(i) << ", global: "
    //           << kernel.global_work_size(i) << std::endl;
    // }

    if (bias_term_) {
      viennacl::ocl::enqueue(
          kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                 WrapHandle((cl_mem) weight, &ctx),
                 WrapHandle((cl_mem) bias, &ctx),
                 WrapHandle((cl_mem) top_data, &ctx)),
          ctx.get_queue());
    } else {
      viennacl::ocl::enqueue(
          kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                 WrapHandle((cl_mem) weight, &ctx),
                 WrapHandle((cl_mem) top_data, &ctx)),
          ctx.get_queue());
    }
  }
#endif  // USE_GREENTEA

#ifdef USE_CUDA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    CUfunction kernel;
    cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_, "conv_forward");

    if (bias_term_) {
      void *args[] = { &bottom_data, &weight, &bias, &top_data };
      cuLaunchKernel(kernel, (this->N_FW_ - 1) / fw_div_N + 1,  // Grid X
                     (this->M_FW_ - 1) / fw_div_M + 1,  // Grid Y
                     batch_size * group_,               // Grid Z
                     fw_wgs0, fw_wgs1, 1,               // Local
                     0, NULL, args, 0);                 // Arguments
    } else {
      void *args[] = { &bottom_data, &weight, &top_data };
      cuLaunchKernel(kernel, (this->N_FW_ - 1) / fw_div_N + 1,  // Grid X
                     (this->M_FW_ - 1) / fw_div_M + 1,  // Grid Y
                     batch_size * group_,               // Grid Z
                     fw_wgs0, fw_wgs1, 1,               // Local
                     0, NULL, args, 0);                 // Arguments
    }
    cuCtxSynchronize();
  }
#endif  // USE_CUDA
}

template<typename Dtype>
void LibDNNConv<Dtype>::Backward(bool prop_down_data, bool prop_down_weights,
                                 const Dtype* top_data, const Dtype* top_diff,
                                 const Dtype* weight, Dtype* weight_diff,
                                 const Dtype* bias, Dtype* bias_diff,
                                 const Dtype* bottom_data, Dtype* bottom_diff,
                                 int_tp batch_size) {
  int bw_wptn = bw_tuner_->get_param<int>("WPTN");
  int bw_wptm = bw_tuner_->get_param<int>("WPTM");
  int bw_wgs0 = bw_tuner_->get_param<int>("workgroup_size_0");
  int bw_wgs1 = bw_tuner_->get_param<int>("workgroup_size_1");
  int bw_div_N = bw_wptn * bw_wgs0;
  int bw_div_M = bw_wptm * bw_wgs1;

  int wg_wptn = wg_tuner_->get_param<int>("WPTN");
  int wg_wptm = wg_tuner_->get_param<int>("WPTM");
  int wg_wgs0 = wg_tuner_->get_param<int>("workgroup_size_0");
  int wg_wgs1 = wg_tuner_->get_param<int>("workgroup_size_1");
  int wg_div_N = wg_wptn * wg_wgs0;
  int wg_div_M = wg_wptm * wg_wgs1;

  if (prop_down_data && bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    int_tp ims = batch_size * fmaps_in_;
    for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
      ims *= im_in_shape_[i];
    }
    LibDNN<Dtype>::SetMemory(bottom_diff, ims, 0, (Dtype) 0);
  }

#ifdef USE_GREENTEA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
    // Backprop w.r.t. data
    if (prop_down_data) {
      viennacl::ocl::kernel &kernel =
          LibDNN<Dtype>::ocl_program_.get_kernel("conv_backward");
      viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

      kernel.local_work_size(0, bw_wgs0);
      kernel.local_work_size(1, bw_wgs1);
      kernel.local_work_size(2, 1);

      kernel.global_work_size(0, ((this->N_BW_ - 1) / bw_div_N + 1) * bw_wgs0);
      kernel.global_work_size(1, ((this->M_BW_ - 1) / bw_div_M + 1) * bw_wgs1);
      kernel.global_work_size(2, batch_size * group_);

      // for (int i = 0; i < 3; ++i) {
      // std::cout << i << "; local: ">
      //           << kernel.local_work_size(i) << ", global: "
      //           << kernel.global_work_size(i) << std::endl;
      // }

      if (bias_term_) {
        viennacl::ocl::enqueue(
            kernel(WrapHandle((cl_mem) top_diff, &ctx),
                   WrapHandle((cl_mem) weight, &ctx),
                   WrapHandle((cl_mem) bias, &ctx),
                   WrapHandle((cl_mem) bottom_diff, &ctx)),
            ctx.get_queue());
      } else {
        viennacl::ocl::enqueue(
            kernel(WrapHandle((cl_mem) top_diff, &ctx),
                   WrapHandle((cl_mem) weight, &ctx),
                   WrapHandle((cl_mem) bottom_diff, &ctx)),
            ctx.get_queue());
      }
    }

    // Backprop w.r.t. weights and bias
    if (prop_down_weights
        && (this->weights_backward_ || this->bias_backward_)) {
      viennacl::ocl::kernel &kernel =
          LibDNN<Dtype>::ocl_program_.get_kernel("conv_weights");

      viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

      kernel.local_work_size(0, wg_wgs0);
      kernel.local_work_size(1, wg_wgs1);
      kernel.local_work_size(2, 1);

      kernel.global_work_size(0, ((this->N_WG_ - 1) / wg_div_N + 1) * wg_wgs0);
      kernel.global_work_size(1, ((this->M_WG_ - 1) / wg_div_M + 1) * wg_wgs1);

      if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
        kernel.global_work_size(2, group_);
      }
      if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
        kernel.global_work_size(2, batch_size * group_);
      }

      // for (int i = 0; i < 3; ++i) {
      // std::cout << i << "; local: "
      //           << kernel.local_work_size(i) << ", global: "
      //           << kernel.global_work_size(i) << std::endl;
      // }

      if (bias_term_) {
        viennacl::ocl::enqueue(
            kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                   WrapHandle((cl_mem) top_diff, &ctx),
                   WrapHandle((cl_mem) bias_diff, &ctx),
                   WrapHandle((cl_mem) weight_diff, &ctx), batch_size),
            ctx.get_queue());
      } else {
        viennacl::ocl::enqueue(
            kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                   WrapHandle((cl_mem) top_diff, &ctx),
                   WrapHandle((cl_mem) weight_diff, &ctx), batch_size),
            ctx.get_queue());
      }
    }
  }
#endif  // USE_GREENTEA

#ifdef USE_CUDA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    // Backprop w.r.t. data
    if (prop_down_data) {
      CUfunction kernel;
      cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                          "conv_backward");

      if (bias_term_) {
        void *args[] = { &top_diff, &weight, &bias, &bottom_diff };
        cuLaunchKernel(kernel, (this->N_BW_ - 1) / bw_div_N + 1,  // Grid X
                       (this->M_BW_ - 1) / bw_div_M + 1,  // Grid Y
                       batch_size * group_,               // Grid Z
                       bw_wgs0, bw_wgs1, 1,               // Local
                       0, NULL, args, 0);                 // Arguments
      } else {
        void *args[] = { &top_diff, &weight, &bottom_diff };
        cuLaunchKernel(kernel, (this->N_BW_ - 1) / bw_div_N + 1,  // Grid X
                       (this->M_BW_ - 1) / bw_div_M + 1,  // Grid Y
                       batch_size * group_,               // Grid Z
                       bw_wgs0, bw_wgs1, 1,               // Local
                       0, NULL, args, 0);                 // Arguments
      }
    }

    // Backprop w.r.t. weights and bias
    if (prop_down_weights &&
        (this->weights_backward_ || this->bias_backward_)) {
      CUfunction kernel;
      cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_, "conv_weights");

      int gws2 = 0;

      if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
        gws2 = group_;
      }
      if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
        gws2 = batch_size * group_;
      }

      if (bias_term_) {
        void *args[] = { &bottom_data, &top_diff, &bias_diff, &weight_diff,
            &batch_size };
        cuLaunchKernel(kernel, (this->N_WG_ - 1) / wg_div_N + 1,  // Grid X
                       (this->M_WG_ - 1) / wg_div_M + 1,  // Grid Y
                       gws2,                              // Grid Z
                       wg_wgs0, wg_wgs1, 1,               // Local
                       0, NULL, args, 0);                 // Arguments
      } else {
        void *args[] = { &bottom_data, &top_diff, &weight_diff, &batch_size };
        cuLaunchKernel(kernel, (this->N_WG_ - 1) / wg_div_N + 1,  // Grid X
                       (this->M_WG_ - 1) / wg_div_M + 1,  // Grid Y
                       gws2,                              // Grid Z
                       wg_wgs0, wg_wgs1, 1,               // Local
                       0, NULL, args, 0);                 // Arguments
      }
    }
  }
#endif  // USE_CUDA
}

template<typename Dtype>
void LibDNNConv<Dtype>::Tune(Dtype* top_data, Dtype* top_diff, Dtype* weight,
                             Dtype* weight_diff, Dtype* bias, Dtype* bias_diff,
                             Dtype* bottom_data, Dtype* bottom_diff,
                             int_tp batch_size) {
  LibDNNConv* self = this;
  // Autotune forward kernel
  fw_tuner_->set_setup_routine([&]() -> bool {
    try {
      self->GenerateKernels();
      return self->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  fw_tuner_->set_benchmark_routine([&]() -> double {
    try {
      Timer timer;
      timer.Start();
      self->Forward(bottom_data, weight, bias, top_data, batch_size);
      timer.Stop();
      // Score is 1/time
      return 1.0 / timer.MicroSeconds();
    } catch(...) {
      // Failure score
      return -1.0;
    }
  });
  fw_tuner_->Tune(LIBDNN_TUNER_METHOD_ANNEALING);

  // Autotune backward kernel
  bw_tuner_->set_setup_routine([&]() -> bool {
    try {
      self->GenerateKernels();
      return self->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  bw_tuner_->set_benchmark_routine([&]() -> double {
    try {
      Timer timer;
      timer.Start();
      self->Backward(true, false,
          top_data, top_diff,
          weight, weight_diff,
          bias, bias_diff,
          bottom_data, bottom_diff,
          batch_size);
      timer.Stop();
      // Score is 1/time
      return 1.0 / timer.MicroSeconds();
    } catch(...) {
      // Failure score
      return -1.0;
    }
  });
  bw_tuner_->Tune(LIBDNN_TUNER_METHOD_ANNEALING);

  // Autotune weight/bias error kernel
  wg_tuner_->set_setup_routine([&]() -> bool {
    try {
      self->GenerateKernels();
      return self->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  wg_tuner_->set_benchmark_routine([&]() -> double {
    try {
      Timer timer;
      timer.Start();
      self->Backward(false, true,
          top_data, top_diff,
          weight, weight_diff,
          bias, bias_diff,
          bottom_data, bottom_diff,
          batch_size);
      timer.Stop();
      // Score is 1/time
      return 1.0 / timer.MicroSeconds();
    } catch(...) {
      // Failure score
      return -1.0;
    }
  });
  wg_tuner_->Tune(LIBDNN_TUNER_METHOD_ANNEALING);
}

INSTANTIATE_CLASS(LibDNNConv);

}  // namespace caffe

#endif  // USE_LIBDNN
