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
LibDNNDeconv<Dtype>::LibDNNDeconv(LibDNNDeconvConfig config) {
  config_ = config;
  LibDNN<Dtype>::dev_ptr_ = config.dev_ptr;
  this->bias_term_ = config.bias_term;
  this->bias_multiplier_ = config.bias_term ? 1.0 : 0.0;
  LibDNN<Dtype>::fast_unsafe_math_ = config.fast_unsafe_math;
  int_tp dims = config.in_shape.size();
  int_tp spatial_dims = config.kernel.size();

  this->num_axes_ = spatial_dims;
  this->fmaps_in_ = config.in_shape[dims - spatial_dims - 1];
  this->fmaps_out_ = config.out_shape[dims - spatial_dims - 1];
  this->group_ = config.group;

  this->wgalgo_ = config.wgalgo;
  this->bwalgo_ = config.bwalgo;

  this->weights_backward_ = config.weights_backward;
  this->bias_backward_ = config.bias_backward;

  this->skip_range_check_ = true;

  for (int_tp i = 0; i < spatial_dims; ++i) {
    this->kernel_shape_.push_back(config.kernel[i]);
    this->pad_.push_back(config.pad[i]);
    if (this->pad_[i] > 0) {
      this->skip_range_check_ = false;
    }
    this->stride_.push_back(config.stride[i]);
    this->dilation_.push_back(config.dilation[i]);
    this->im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
    this->im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
  }

  this->bw_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());
  this->fw_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());
  this->wg_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());

  // Setup tuning parameters

  // Work groups
  for (int id = 0; id < 2; ++id) {
    std::vector<int_tp> workgroup_sizes;
    workgroup_sizes.push_back(1);
    workgroup_sizes.push_back(2);
    for (int_tp i = 4; i < LibDNN<Dtype>::dev_ptr_->workgroup_size(id);
            i += 4) {
      workgroup_sizes.push_back(i);
    }
    this->bw_tuner_->template add_set_param<int_tp>
        ("workgroup_size_" + std::to_string(id), 16, workgroup_sizes);
    this->fw_tuner_->template add_set_param<int_tp>
        ("workgroup_size_" + std::to_string(id), 16, workgroup_sizes);
    this->wg_tuner_->template add_set_param<int_tp>
        ("workgroup_size_" + std::to_string(id), 16, workgroup_sizes);
  }

  // TSK
  this->bw_tuner_->template add_range_param<int_tp>("TSK", 8, 1, 32, 1);
  this->fw_tuner_->template add_range_param<int_tp>("TSK", 8, 1, 32, 1);
  this->wg_tuner_->template add_range_param<int_tp>("TSK", 8, 1, 32, 1);

  this->bw_tuner_->template add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);
  this->fw_tuner_->template add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);
  this->wg_tuner_->template add_range_param<int_tp>("TSK_UNROLL", 1, 1, 16, 1);

  // WPTM, WPTN
  this->bw_tuner_->template add_range_param<int_tp>("WPTM", 4, 2, 16, 2);
  this->fw_tuner_->template add_range_param<int_tp>("WPTM", 4, 2, 16, 2);
  this->wg_tuner_->template add_range_param<int_tp>("WPTM", 4, 2, 16, 2);

  this->bw_tuner_->template add_set_param<int_tp>("VWM", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->fw_tuner_->template add_set_param<int_tp>("VWM", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->wg_tuner_->template add_set_param<int_tp>("VWM", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));

  this->bw_tuner_->template add_range_param<int_tp>("WPTN", 4, 2, 16, 2);
  this->fw_tuner_->template add_range_param<int_tp>("WPTN", 4, 2, 16, 2);
  this->wg_tuner_->template add_range_param<int_tp>("WPTN", 4, 2, 16, 2);

  this->bw_tuner_->template add_set_param<int_tp>("VWN", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->fw_tuner_->template add_set_param<int_tp>("VWN", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->wg_tuner_->template add_set_param<int_tp>("VWN", 4, std::vector<int_tp>(
      {1, 2, 4, 8, 16 }));

  // Constraint using TSK, TSM, RTSM and RTSN. Adapt TSK if constraint fails.
  this->bw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTM", "workgroup_size_1"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTM", "workgroup_size_1"}), std::vector<
    std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTM", "workgroup_size_1"}), std::vector<
    std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  // Constraint using TSK, TSN, RTSN and RTSM. Adapt TSK if constraint fails.
  this->bw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTN", "workgroup_size_0"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTN", "workgroup_size_0"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "WPTN", "workgroup_size_0"}),
    std::vector<std::string>({"TSK"}), [](std::vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->bw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "TSK_UNROLL"}),
    std::vector<std::string>({"TSK_UNROLL"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "TSK_UNROLL"}),
    std::vector<std::string>({"TSK_UNROLL"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"TSK", "TSK_UNROLL"}),
    std::vector<std::string>({"TSK_UNROLL"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->bw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"WPTM", "VWM"}),
    std::vector<std::string>({"WPTM"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"WPTM", "VWM"}),
    std::vector<std::string>({"WPTM"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"WPTM", "VWM"}),
    std::vector<std::string>({"WPTM"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->bw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"WPTN", "VWN"}),
    std::vector<std::string>({"WPTN"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"WPTN", "VWN"}),
    std::vector<std::string>({"WPTN"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    std::vector<std::string>({"WPTN", "VWN"}),
    std::vector<std::string>({"WPTN"}),
    [](std::vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  // this->pad_A, this->pad_B
  this->bw_tuner_->template
      add_range_param<int_tp>("lmem_this->pad_A", 0, 0, 8, 1);
  this->fw_tuner_->template
      add_range_param<int_tp>("lmem_this->pad_A", 0, 0, 8, 1);
  this->wg_tuner_->template
      add_range_param<int_tp>("lmem_this->pad_A", 0, 0, 8, 1);
  this->bw_tuner_->template
      add_range_param<int_tp>("lmem_this->pad_B", 0, 0, 8, 1);
  this->fw_tuner_->template
      add_range_param<int_tp>("lmem_this->pad_B", 0, 0, 8, 1);
  this->wg_tuner_->template
      add_range_param<int_tp>("lmem_this->pad_B", 0, 0, 8, 1);

  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    // CUDA needs the vector elements unrolled
    this->bw_tuner_->add_boolean_param("vector_unroll", true, false);
    this->fw_tuner_->add_boolean_param("vector_unroll", true, false);
    this->wg_tuner_->add_boolean_param("vector_unroll", true, false);
  } else {
    // OpenCL does not need the vector elements unrolled, and may
    // save registers by not doing it
    this->bw_tuner_->add_boolean_param("vector_unroll", true, true);
    this->fw_tuner_->add_boolean_param("vector_unroll", true, true);
    this->wg_tuner_->add_boolean_param("vector_unroll", true, true);
  }

  GenerateKernels();
  LibDNN<Dtype>::CompileKernels();
}

template<typename Dtype>
const LibDNNDeconvConfig LibDNNDeconv<Dtype>::get_config() {
  return config_;
}

template<typename Dtype>
std::string LibDNNDeconv<Dtype>::string_identifier() {
  std::stringstream ss;
  ss << "DECONV_";
  if (std::is_same<Dtype, double>::value) {
    ss << "double_";
  } else {
    ss << "float_";
  }
  // Device name
  ss << LibDNN<Dtype>::dev_ptr_->name();
  ss << "_";
  ss << this->num_axes_ << "D_";
  ss << "IN[";
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    ss << this->im_in_shape_[i];
    if (i < this->im_in_shape_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_OUT[";
  for (int_tp i = 0; i < this->im_out_shape_.size(); ++i) {
    ss << this->im_out_shape_[i];
    if (i < this->im_out_shape_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_K[";
  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    ss << this->kernel_shape_[i];
    if (i < this->kernel_shape_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_S[";
  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    ss << this->stride_[i];
    if (i < this->stride_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_P[";
  for (int_tp i = 0; i < this->pad_.size(); ++i) {
    ss << this->pad_[i];
    if (i < this->pad_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_D[";
  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    ss << this->dilation_[i];
    if (i < this->dilation_.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_";
  ss << "FIN[" << this->fmaps_in_ << "]_";
  ss << "FOUT[" << this->fmaps_out_ << "]_";
  ss << "G[" << this->group_ << "]";
  return ss.str();
}

template<typename Dtype>
std::string LibDNNDeconv<Dtype>::generate_bw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", this->num_axes_);

  // Groups
  LibDNN<Dtype>::add_def(ss, "v_g", this->group_);

  int_tp B_off = this->fmaps_out_;
  int_tp C_off = this->fmaps_in_;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    B_off *= this->im_out_shape_[i];
    C_off *= this->im_in_shape_[i];
  }
  // Input image batch offset
  LibDNN<Dtype>::add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  LibDNN<Dtype>::add_def(ss, "v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    LibDNN<Dtype>::add_def(ss, "v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_imsi", imsi);
  LibDNN<Dtype>::add_def(ss, "v_imso", imso);

  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
  }

  for (int_tp i = 0; i < this->pad_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i), this->pad_[i]);
  }

  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_d_" + std::to_string(i), this->dilation_[i]);
  }

  LibDNN<Dtype>::add_def(ss, "v_fin", this->fmaps_in_);
  LibDNN<Dtype>::add_def(ss, "v_fout", this->fmaps_out_);

  if (this->bias_term_) {
    LibDNN<Dtype>::add_def(ss, "v_bmul", this->bias_multiplier_);
  }

  this->MG_BW_ = this->fmaps_in_;
  this->M_BW_ = this->fmaps_in_ / this->group_;
  this->N_BW_ = 1;
  this->KG_BW_ = this->fmaps_out_;
  this->K_BW_ = this->fmaps_out_ / this->group_;

  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    this->K_BW_ *= this->kernel_shape_[i];
    this->KG_BW_ *= this->kernel_shape_[i];
    this->N_BW_ *= this->im_in_shape_[i];
  }

  // GEMM definitions
  LibDNN<Dtype>::add_def(ss, "MG", this->MG_BW_);
  LibDNN<Dtype>::add_def(ss, "M", this->M_BW_);
  LibDNN<Dtype>::add_def(ss, "N", this->N_BW_);
  LibDNN<Dtype>::add_def(ss, "KG", this->KG_BW_);
  LibDNN<Dtype>::add_def(ss, "K", this->K_BW_);

  // Local memory padding
  LibDNN<Dtype>::add_def(ss, "v_pad_A",
                         this->bw_tuner_->template
                         get_param<int>("lmem_this->pad_A"));
  LibDNN<Dtype>::add_def(ss, "v_pad_B",
                         this->bw_tuner_->template
                         get_param<int>("lmem_this->pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  LibDNN<Dtype>::add_def(
      ss, "TSM", this->bw_tuner_->template get_param<int>("WPTM")
          * this->bw_tuner_->template
          get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  LibDNN<Dtype>::add_def(
      ss, "TSN", this->bw_tuner_->template get_param<int>("WPTN")
          * this->bw_tuner_->template get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  LibDNN<Dtype>::add_def(ss, "TSK", this->bw_tuner_->template
                         get_param<int>("TSK"));
  // TSK unrolling
  LibDNN<Dtype>::add_def(ss, "TSK_UNROLL",
                         this->bw_tuner_->template
                         get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  LibDNN<Dtype>::add_def(ss, "WPTM", this->bw_tuner_->template
                         get_param<int>("WPTM"));
  LibDNN<Dtype>::add_def(ss, "VWM", this->bw_tuner_->template
                         get_param<int>("VWM"));
  // The work-per-thread in dimension N
  LibDNN<Dtype>::add_def(ss, "WPTN", this->bw_tuner_->template
                         get_param<int>("WPTN"));
  LibDNN<Dtype>::add_def(ss, "VWN", this->bw_tuner_->template
                         get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  LibDNN<Dtype>::add_def(ss, "RTSM",
                         this->bw_tuner_->template
                         get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  LibDNN<Dtype>::add_def(ss, "RTSN",
                         this->bw_tuner_->template
                         get_param<int>("workgroup_size_0"));
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
std::string LibDNNDeconv<Dtype>::generate_fw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", this->num_axes_);

  // Groups
  LibDNN<Dtype>::add_def(ss, "v_g", this->group_);

  int_tp A_off = this->fmaps_in_ * this->fmaps_out_;
  int_tp B_off = this->fmaps_in_;
  int_tp C_off = this->fmaps_out_;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    A_off *= this->kernel_shape_[i];
    B_off *= this->im_in_shape_[i];
    C_off *= this->im_out_shape_[i];
  }

  // Weight offset (only used for groups)
  LibDNN<Dtype>::add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  LibDNN<Dtype>::add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  LibDNN<Dtype>::add_def(ss, "v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    LibDNN<Dtype>::add_def(ss, "v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_imsi", imsi);
  LibDNN<Dtype>::add_def(ss, "v_imso", imso);

  int_tp v_ks = 1;
  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
    v_ks *= this->kernel_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_ks", v_ks);

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Set padding to account for padding loss (backward),
    // remove forward padding
    for (int_tp i = 0; i < this->pad_.size(); ++i) {
      LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i),
            (this->kernel_shape_[i] - 1) * this->dilation_[i] - this->pad_[i]);
    }
  }

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    for (int_tp i = 0; i < this->pad_.size(); ++i) {
      LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i), this->pad_[i]);
    }
  }

  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_d_" + std::to_string(i), this->dilation_[i]);
  }

  LibDNN<Dtype>::add_def(ss, "v_fin", this->fmaps_in_);
  LibDNN<Dtype>::add_def(ss, "v_fout", this->fmaps_out_);

  if (this->bias_term_) {
    LibDNN<Dtype>::add_def(ss, "v_bmul", this->bias_multiplier_);
  }

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    this->MG_FW_ = this->fmaps_out_;
    this->M_FW_ = this->fmaps_out_ / this->group_;
    this->N_FW_ = 1;
    this->KG_FW_ = this->fmaps_in_;
    this->K_FW_ = this->fmaps_in_ / this->group_;

    for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
      this->K_FW_ *= this->kernel_shape_[i];
      this->KG_FW_ *= this->kernel_shape_[i];
      this->N_FW_ *= this->im_out_shape_[i];
    }
  }

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    this->MG_FW_ = this->fmaps_out_;
    this->M_FW_ = this->fmaps_out_ / this->group_;
    this->N_FW_ = 1;
    this->KG_FW_ = this->fmaps_in_;
    this->K_FW_ = this->fmaps_in_ / this->group_;

    for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
      this->MG_FW_ *= this->kernel_shape_[i];
      this->M_FW_ *= this->kernel_shape_[i];
      this->N_FW_ *= this->im_in_shape_[i];
    }
  }

  // GEMM definitions
  LibDNN<Dtype>::add_def(ss, "MG", this->MG_FW_);
  LibDNN<Dtype>::add_def(ss, "M", this->M_FW_);
  LibDNN<Dtype>::add_def(ss, "N", this->N_FW_);
  LibDNN<Dtype>::add_def(ss, "KG", this->KG_FW_);
  LibDNN<Dtype>::add_def(ss, "K", this->K_FW_);

  // Local memory padding
  LibDNN<Dtype>::add_def(ss, "v_pad_A",
                         this->fw_tuner_->template
                         get_param<int>("lmem_this->pad_A"));
  LibDNN<Dtype>::add_def(ss, "v_pad_B",
                         this->fw_tuner_->template
                         get_param<int>("lmem_this->pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  LibDNN<Dtype>::add_def(
      ss,
      "TSM",
      this->fw_tuner_->template get_param<int>("WPTM")
          * this->fw_tuner_->template get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  LibDNN<Dtype>::add_def(
      ss,
      "TSN",
      this->fw_tuner_->template get_param<int>("WPTN")
          * this->fw_tuner_->template get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  LibDNN<Dtype>::add_def(ss, "TSK", this->fw_tuner_->template
                         get_param<int>("TSK"));
  // TSK unrolling
  LibDNN<Dtype>::add_def(ss, "TSK_UNROLL",
                         this->fw_tuner_->template
                         get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  LibDNN<Dtype>::add_def(ss, "WPTM", this->fw_tuner_->template
                         get_param<int>("WPTM"));
  LibDNN<Dtype>::add_def(ss, "VWM", this->fw_tuner_->template
                         get_param<int>("VWM"));
  // The work-per-thread in dimension N
  LibDNN<Dtype>::add_def(ss, "WPTN", this->fw_tuner_->template
                         get_param<int>("WPTN"));
  LibDNN<Dtype>::add_def(ss, "VWN", this->fw_tuner_->template
                         get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  LibDNN<Dtype>::add_def(ss, "RTSM",
                         this->fw_tuner_->template
                         get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  LibDNN<Dtype>::add_def(ss, "RTSN",
                         this->fw_tuner_->template
                         get_param<int>("workgroup_size_0"));
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
std::string LibDNNDeconv<Dtype>::generate_wg_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", this->num_axes_);

  // Groups
  LibDNN<Dtype>::add_def(ss, "v_g", this->group_);

  int_tp A_off = this->fmaps_in_;
  int_tp B_off = this->fmaps_out_;
  int_tp C_off = this->fmaps_in_ * this->fmaps_out_;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    A_off *= this->im_in_shape_[i];
    B_off *= this->im_out_shape_[i];
    C_off *= this->kernel_shape_[i];
  }
  // Output image batch offset
  LibDNN<Dtype>::add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  LibDNN<Dtype>::add_def(ss, "v_B_off", B_off);
  // Weights offset
  LibDNN<Dtype>::add_def(ss, "v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    LibDNN<Dtype>::add_def(ss, "v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_imsi", imsi);
  LibDNN<Dtype>::add_def(ss, "v_imso", imso);

  int_tp v_ks = 1;
  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
    v_ks *= this->kernel_shape_[i];
  }
  LibDNN<Dtype>::add_def(ss, "v_ks", v_ks);

  // Set padding to account for padding loss (backward), remove forward padding
  for (int_tp i = 0; i < this->pad_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_p_" + std::to_string(i), this->pad_[i]);
  }

  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    LibDNN<Dtype>::add_def(ss, "v_d_" + std::to_string(i), this->dilation_[i]);
  }

  LibDNN<Dtype>::add_def(ss, "v_fin", this->fmaps_in_);
  LibDNN<Dtype>::add_def(ss, "v_fout", this->fmaps_out_);

  LibDNN<Dtype>::add_def(ss, "v_bmul", this->bias_multiplier_);

  this->MG_WG_ = this->fmaps_in_;
  this->M_WG_ = this->fmaps_in_ / this->group_;
  this->NG_WG_ = this->fmaps_out_;
  this->N_WG_ = this->fmaps_out_ / this->group_;
  this->K_WG_ = 1;

  this->MG_BG_ = this->fmaps_out_;
  this->M_BG_ = this->fmaps_out_ / this->group_;
  this->NG_BG_ = this->fmaps_in_;
  this->N_BG_ = this->fmaps_in_ / this->group_;
  this->K_BG_ = 1;

  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    this->N_WG_ *= this->kernel_shape_[i];
    this->NG_WG_ *= this->kernel_shape_[i];
    this->K_WG_ *= this->im_in_shape_[i];
    this->N_BG_ *= this->kernel_shape_[i];
    this->NG_BG_ *= this->kernel_shape_[i];
    this->K_BG_ *= this->im_out_shape_[i];
  }

  // GEMM definitions
  LibDNN<Dtype>::add_def(ss, "MG", this->MG_WG_);
  LibDNN<Dtype>::add_def(ss, "M", this->M_WG_);
  LibDNN<Dtype>::add_def(ss, "N", this->N_WG_);
  LibDNN<Dtype>::add_def(ss, "NG", this->NG_WG_);
  LibDNN<Dtype>::add_def(ss, "K", this->K_WG_);
  LibDNN<Dtype>::add_def(ss, "MGB", this->MG_BG_);
  LibDNN<Dtype>::add_def(ss, "MB", this->M_BG_);
  LibDNN<Dtype>::add_def(ss, "NB", this->N_WG_);
  LibDNN<Dtype>::add_def(ss, "NGB", this->NG_WG_);
  LibDNN<Dtype>::add_def(ss, "KB", this->K_BG_);

  // Local memory padding
  LibDNN<Dtype>::add_def(ss, "v_pad_A",
                         this->wg_tuner_->template
                         get_param<int>("lmem_this->pad_A"));
  LibDNN<Dtype>::add_def(ss, "v_pad_B",
                         this->wg_tuner_->template
                         get_param<int>("lmem_this->pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  LibDNN<Dtype>::add_def(
      ss,
      "TSM",
      this->wg_tuner_->template get_param<int>("WPTM")
          * this->wg_tuner_->template get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  LibDNN<Dtype>::add_def(
      ss,
      "TSN",
      this->wg_tuner_->template get_param<int>("WPTN")
          * this->wg_tuner_->template get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  LibDNN<Dtype>::add_def(ss, "TSK", this->wg_tuner_->template
                         get_param<int>("TSK"));
  // TSK unrolling
  LibDNN<Dtype>::add_def(ss, "TSK_UNROLL",
                         this->wg_tuner_->template
                         get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  LibDNN<Dtype>::add_def(ss, "WPTM", this->wg_tuner_->template
                         get_param<int>("WPTM"));
  LibDNN<Dtype>::add_def(ss, "VWM", this->wg_tuner_->template
                         get_param<int>("VWM"));
  // The work-per-thread in dimension N
  LibDNN<Dtype>::add_def(ss, "WPTN", this->wg_tuner_->template
                         get_param<int>("WPTN"));
  LibDNN<Dtype>::add_def(ss, "VWN", this->wg_tuner_->template
                         get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  LibDNN<Dtype>::add_def(ss, "RTSM",
                         this->wg_tuner_->template
                         get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  LibDNN<Dtype>::add_def(ss, "RTSN",
                         this->wg_tuner_->template
                         get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  LibDNN<Dtype>::add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  LibDNN<Dtype>::add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  LibDNN<Dtype>::add_def(ss, "v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");
  LibDNN<Dtype>::add_def(ss, "v_num_tiles_B", "(((KB - 1)/(TSK*2) + 1)*2)");


  return ss.str();
}

template<typename Dtype>
std::string LibDNNDeconv<Dtype>::generate_bw_kernels(std::string name) {
  std::stringstream ss;

  int wptn = this->bw_tuner_->template get_param<int>("WPTN");
  int wptm = this->bw_tuner_->template get_param<int>("WPTM");
  int tsk = this->bw_tuner_->template get_param<int>("TSK");
  int rtsn = this->bw_tuner_->template get_param<int>("workgroup_size_0");
  int rtsm = this->bw_tuner_->template get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = this->bw_tuner_->template get_param<int>("VWM");
  int vwn = this->bw_tuner_->template get_param<int>("VWN");
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
  if (this->bias_term_) {
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
  if (this->group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (this->group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (M * K);" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch + group * (M * N);"
       << std::endl;
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(this->bw_tuner_, false, false);

  ss << "{" << std::endl;  // Scoping for load & compute block
  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp t = 0; t < v_num_tiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "{" << std::endl;  // Scoping for loading A
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
  for (int_tp i = 0; i < this->num_axes_; ++i) {
    ss << "int_tp d_iter_" << i << ";" << std::endl;
    ss << "int_tp d_temp_" << i << ";" << std::endl;
  }

  ss << "int_tp imageIndex = offN + col;" << std::endl;
  for (int_tp i = this->num_axes_ - 1; i >= 0; --i) {
    // Compute d_iter, final tiledIndex becomes input feature map ID
    // Scale d_iter by the dilation factor
    ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
       << ";" << std::endl;
    ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

    // Compute d_temp
    // Scale d_temp by the stride and subtract the padding
    ss << "d_temp_" << i << " = (imageIndex % v_imsi_" << i << ") * v_s_" << i
       << " - v_p_" << i << ";" << std::endl;
    ss << "imageIndex = imageIndex / v_imsi_" << i << ";" << std::endl;
  }

  // Recombine final index, compute in-range
  if (!this->skip_range_check_) {
    ss << "bool in_range = true;" << std::endl;
  }
  ss << "int_tp d_iter_im;" << std::endl;
  for (int_tp i = 0; i < this->num_axes_; ++i) {
    // Here, d_temp_ represents the column shift,
    // while d_iter_ is the kernel shift
    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
    ss << "tiledIndex = tiledIndex * v_imso_" << i << " + d_iter_im;"
       << std::endl;
    if (!this->skip_range_check_) {
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imso_" << i << ";"
         << std::endl;
    }
  }

  if (!this->skip_range_check_) {
    ss << "if (in_range) {" << std::endl;
  }
  // tiledIndex now holds the memory offset for the input image
  ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
  if (!this->skip_range_check_) {
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

  ss << this->generate_gemm_core(this->bw_tuner_, false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  ss << "Cptr[globalRow * N + globalCol] = "
     << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
std::string LibDNNDeconv<Dtype>::generate_wg_kernels(std::string name) {
  std::stringstream ss;

  int wptn = this->wg_tuner_->template get_param<int>("WPTN");
  int wptm = this->wg_tuner_->template get_param<int>("WPTM");
  int tsk = this->wg_tuner_->template get_param<int>("TSK");
  int rtsn = this->wg_tuner_->template get_param<int>("workgroup_size_0");
  int rtsm = this->wg_tuner_->template get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = this->wg_tuner_->template get_param<int>("VWM");
  int vwn = this->wg_tuner_->template get_param<int>("VWN");
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
  if (this->bias_term_) {
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
  if (this->group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (this->group_ > 1) {
    ss << "__global const Dtype* Aptr = im_in + batch * v_A_off"
       << " + group * (v_A_off / v_g);" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + batch * v_B_off"
       << " + group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = wg + group * (M * N);" << std::endl;
  } else {
    ss << "__global const Dtype* Aptr = im_in + batch * v_A_off;" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + batch * v_B_off;" << std::endl;
    ss << "__global Dtype* Cptr = wg;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(this->wg_tuner_, false,
                            this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT);

  ss << "{" << std::endl;  // Scoping for load & compute block
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
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
  for (int_tp i = 0; i < this->num_axes_; ++i) {
    ss << "int_tp d_iter_" << i << ";" << std::endl;
    ss << "int_tp d_temp_" << i << ";" << std::endl;
  }

  ss << "int_tp imageIndex = offN + col;" << std::endl;
  for (int_tp i = this->num_axes_ - 1; i >= 0; --i) {
    // Compute d_iter, final imageIndex becomes input feature map ID
    // Scale d_iter by the dilation factor
    ss << "d_iter_" << i << " = (imageIndex % v_k_" << i << ") * v_d_" << i
       << ";" << std::endl;
    ss << "imageIndex = imageIndex / v_k_" << i << ";" << std::endl;

    // Compute d_temp
    // Scale d_temp by the stride and subtract the padding
    ss << "d_temp_" << i << " = (tiledIndex % v_imsi_" << i << ") * v_s_" << i
       << " - v_p_" << i << ";" << std::endl;
    ss << "tiledIndex = tiledIndex / v_imsi_" << i << ";" << std::endl;
  }

  // Recombine final index, compute in-range
  if (!this->skip_range_check_) {
    ss << "bool in_range = true;" << std::endl;
  }
  ss << "int_tp d_iter_im;" << std::endl;
  for (int_tp i = 0; i < this->num_axes_; ++i) {
    // Here, d_temp_ represents the column shift,
    // while d_iter_ is the kernel shift
    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
    ss << "imageIndex = imageIndex * v_imso_" << i << " + d_iter_im;"
       << std::endl;
    if (!this->skip_range_check_) {
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imso_" << i << ";"
         << std::endl;
    }
  }

  if (!this->skip_range_check_) {
    ss << "if (in_range) {" << std::endl;
  }
  // imageIndex now holds the memory offset for the input image
  ss << "Bsub[row][col] = Bptr[imageIndex];" << std::endl;
  if (!this->skip_range_check_) {
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

  ss << this->generate_gemm_core(this->wg_tuner_, false)
     << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;

  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Shift batch
    ss << "Aptr += v_A_off;" << std::endl;
    ss << "Bptr += v_B_off;" << std::endl;
    // The batch loop
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;"
     << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    ss << "Cptr[globalRow * N + globalCol] = "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  }
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
    ss << "atomicAdd(&(Cptr[globalRow * N + globalCol]), "
       << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]);" << std::endl;
  }
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;


  // Bias kernel
  ss << "__kernel" << std::endl;
  ss << "__attribute__((reqd_work_group_size("
     << rtsn << ", " << rtsm << ", 1)))" << std::endl;
  ss << "__attribute__((vec_type_hint(Dtype"
     << std::min(vwm, vwn) << ")))" << std::endl;
  ss << "void " + name + "_bias(";
  ss << "__global const Dtype* __restrict im_in, ";
  ss << "__global const Dtype* __restrict im_out, ";
  ss << "__global Dtype* __restrict bias, ";
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

  // Batch and group
  if (this->group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (this->group_ > 1) {
    ss << "__global const Dtype* Aptr = im_out + batch * v_B_off"
       << " + group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Dptr = bias + group * (v_fout / v_g);"
       << std::endl;
  } else {
    ss << "__global const Dtype* Aptr = im_out + batch * v_B_off;" << std::endl;
    ss << "__global Dtype* Dptr = bias;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for D registers

  bool unroll = this->wg_tuner_->template get_param<bool>("vector_unroll");

  ss << "Dtype" << vwm << " Dreg[WPTM/VWM];" << std::endl;

  // Initialize the accumulation registers
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Load
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
    ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
       << std::endl;
    ss << "((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM] = Dptr[globalRow];"
       << std::endl;
    ss << "}" << std::endl;
  } else {
    // Zero init
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

  ss << "{" << std::endl;  // Scoping for load & compute block
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Additional batch loop, keep the same accumulator for the weight gradient
    ss << "for (batch = 0; batch < batch_size; ++batch) {" << std::endl;
  }

  // Loop over all tiles
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp t = 0; t < v_num_tiles_B; ++t) {" << std::endl;

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
  ss << "if ((offM + row) < MB && tiledIndex < KB) {" << std::endl;
  ss << "Asub[row][col] = Aptr[(offM + row) * KB + tiledIndex];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "Asub[row][col] = 0.0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for loading A

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  ss << "Dtype" << vwm << " Areg;" << std::endl;
  // Loop over the values of a single tile
  ss << "#pragma unroll 1" << std::endl;
  ss << "for (int_tp kt=0; kt<TSK; kt+=TSK_UNROLL) {" << std::endl;
  ss << "#pragma unroll "
     << this->wg_tuner_->template get_param<int>("TSK_UNROLL") << std::endl;
  ss << "for (int_tp ku=0; ku<TSK_UNROLL; ++ku) {" << std::endl;
  ss << "int_tp k = kt + ku;" << std::endl;

  // Perform the computation
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
  ss << "int_tp row = tidm + wm*VWM*RTSM;" << std::endl;
  for (int i = 0; i < vwm; ++i) {
    ss << "VEC_" << vwm << "_" << i << "(Areg)" << " = Asub[row + " << (i*rtsm)
       << "][k];" << std::endl;
  }
  if (unroll) {
    for (int i = 0; i < vwm; ++i) {
      ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) " << "+= VEC_" << vwm
         << "_" << i << "(Areg) * v_bmul;" << std::endl;
    }
  } else {
    ss << "Dreg[wm] += Areg * v_bmul;" << std::endl;
  }
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;

  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Shift batch
    ss << "Aptr += v_B_off;" << std::endl;
    // The batch loop
    ss << "}" << std::endl;
  }
  ss << "}" << std::endl;  // Scoping for load & compute block


  // Store the final results in D
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
     << std::endl;
  ss << "if (tidn == 0 && offN == 0 && globalRow < MB) {" << std::endl;
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    ss << "Dptr[globalRow] = ((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM];"
       << std::endl;
  }
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
    ss << "atomicAdd(&(Dptr[globalRow]), "
       << "((Dtype*)(&(Dreg[wm/VWM])))[wm%VWM]);" << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for D registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
std::string LibDNNDeconv<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;

  int wptn = this->fw_tuner_->template get_param<int>("WPTN");
  int wptm = this->fw_tuner_->template get_param<int>("WPTM");
  int tsk = this->fw_tuner_->template get_param<int>("TSK");
  int rtsn = this->fw_tuner_->template get_param<int>("workgroup_size_0");
  int rtsm = this->fw_tuner_->template get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = this->fw_tuner_->template get_param<int>("VWM");
  int vwn = this->fw_tuner_->template get_param<int>("VWN");
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
  if (this->bias_term_) {
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
  if (this->group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (this->group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (v_A_off / (v_g * v_g));"
       << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch "
       << "+ group * (v_B_off / v_g);" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch "
       << "+ group * (v_C_off / v_g);" << std::endl;
    if (this->bias_term_) {
      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);"
          << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch;" << std::endl;
    if (this->bias_term_) {
      ss << "__global const Dtype* Dptr = bias;" << std::endl;
    }
  }


  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(this->fw_tuner_, false, false);

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

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Load weights (wg) into Asub, flip fin/fout and inverse spatially
    // Compute kidx and midx, the column and row index of the
    // weights in the original A (weights) matrix
    ss << "int_tp kidx = (v_ks - 1 - tiledIndex % v_ks) + (offM + row) * v_ks;"
       << std::endl;
    ss << "int_tp midx = tiledIndex / v_ks;" << std::endl;
    // Check range of the spatially flipped, fin/fout inverted weights
    ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
    // Access weights with the original (translated) weight indices
    ss << "Asub[row][col] = Aptr[kidx + (v_fout / v_g * v_ks) * midx];"
       << std::endl;
    ss << "} else {" << std::endl;  // M-K-Guard
    ss << "Asub[row][col] = 0.0;" << std::endl;
    ss << "}" << std::endl;
  }

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
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

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Load from B with im2col transformation

    // Define temporary registers
    for (int_tp i = 0; i < this->num_axes_; ++i) {
      ss << "int_tp d_iter_" << i << ";" << std::endl;
      ss << "int_tp d_temp_" << i << ";" << std::endl;
    }

    // Compute in-range
    ss << "bool in_range = true;" << std::endl;

    ss << "int_tp imageIndex = offN + col;" << std::endl;
    for (int_tp i = this->num_axes_ - 1; i >= 0; --i) {
      // Compute d_iter, final tiledIndex becomes input feature map ID
      // Scale d_iter by the dilation factor
      ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

      // Compute d_temp
      // Subtract the padding from d_temp, note v_p_i can be negative
      ss << "d_temp_" << i << " = (imageIndex % v_imso_" << i << ")"
         << " - v_p_" << i << ";" << std::endl;
      ss << "imageIndex = imageIndex / v_imso_" << i << ";" << std::endl;
    }

    ss << "int_tp d_iter_im;" << std::endl;
    for (int_tp i = 0; i < this->num_axes_; ++i) {
      // Here, d_temp_ represents the column shift,
      // while d_iter_ is the kernel shift
      ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
      ss << "tiledIndex = tiledIndex * v_imsi_" << i << " + d_iter_im / v_s_"
         << i << ";" << std::endl;
      // In range: Not before or after actual image data
      // and not between image strides
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_" << i
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

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
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

  ss << this->generate_gemm_core(this->fw_tuner_, false) << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;" <<std::endl;
  if (this->bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;" << std::endl;

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "Cptr[globalRow * N + globalCol] = ";
    if (this->bias_term_) {
      ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN]"
         << " + v_bmul * biasval;" << std::endl;
    } else {
      ss << "((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
    }
    ss << "}" << std::endl;
  }

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    // Define temporary registers
    for (int_tp i = 0; i < this->num_axes_; ++i) {
      ss << "int_tp d_iter_" << i << ";" << std::endl;
      ss << "int_tp d_temp_" << i << ";" << std::endl;
    }

    // Compute in-range
    ss << "bool in_range = true;" << std::endl;
    ss << "int_tp tiledIndex = globalRow;" << std::endl;
    ss << "int_tp imageIndex = globalCol;" << std::endl;
    for (int_tp i = this->num_axes_ - 1; i >= 0; --i) {
      // Compute d_iter, final tiledIndex becomes input feature map ID
      // Scale d_iter by the dilation factor
      ss << "d_iter_" << i << " = (tiledIndex % v_k_" << i << ") * v_d_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex / v_k_" << i << ";" << std::endl;

      // Compute d_temp
      // Scale d_temp by the stride
      ss << "d_temp_" << i << " = (imageIndex % v_imsi_" << i << ") * v_s_" << i
         << ";" << std::endl;
      ss << "imageIndex = imageIndex / v_imsi_" << i << ";" << std::endl;
    }

    ss << "in_range &= tiledIndex < v_fout && globalRow < M && globalCol < N;"
       << std::endl;
    ss << "int_tp d_iter_im;" << std::endl;
    for (int_tp i = 0; i < this->num_axes_; ++i) {
      // Here, d_temp_ represents the column shift,
      // while d_iter_ is the kernel shift
      // d_iter_im is the combined offset in the current dimension i
      ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << " - v_p_" << i
         << ";" << std::endl;
      ss << "tiledIndex = tiledIndex * v_imso_" << i << " + d_iter_im;"
         << std::endl;
      // In range: Not before or after actual image data
      ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imso_" << i << ";"
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
void LibDNNDeconv<Dtype>::GenerateKernels() {
  std::stringstream ss;

  ss << LibDNN<Dtype>::generate_header();
  ss << generate_fw_defs();
  ss << generate_fw_kernels("deconv_forward");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("deconv_backward");
  ss << generate_wg_defs();
  ss << generate_wg_kernels("deconv_weights");

  // Write complete kernel string
  LibDNN<Dtype>::kernel_ = ss.str();
}

template<typename Dtype>
void LibDNNDeconv<Dtype>::Forward(const Dtype* bottom_data, const Dtype* weight,
                                const Dtype* bias, Dtype* top_data,
                                int_tp batch_size) {
  int fw_wptn = this->fw_tuner_->template get_param<int>("WPTN");
  int fw_wptm = this->fw_tuner_->template get_param<int>("WPTM");
  int fw_wgs0 = this->fw_tuner_->template get_param<int>("workgroup_size_0");
  int fw_wgs1 = this->fw_tuner_->template get_param<int>("workgroup_size_1");
  int fw_div_N = fw_wptn * fw_wgs0;
  int fw_div_M = fw_wptm * fw_wgs1;

  if (this->bwalgo_
      == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    int_tp ims = batch_size * this->fmaps_out_;
    for (int_tp i = 0; i < this->im_out_shape_.size(); ++i) {
      ims *= this->im_out_shape_[i];
    }
    LibDNN<Dtype>::SetMemory(top_data, ims, 0, (Dtype) 0);
  }

#ifdef USE_GREENTEA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
    viennacl::ocl::kernel &kernel =
        LibDNN<Dtype>::ocl_program_.get_kernel("deconv_forward");
    viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

    kernel.local_work_size(0, fw_wgs0);
    kernel.local_work_size(1, fw_wgs1);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((this->N_FW_ - 1) / fw_div_N + 1) * fw_wgs0);
    kernel.global_work_size(1, ((this->M_FW_ - 1) / fw_div_M + 1) * fw_wgs1);
    kernel.global_work_size(2, batch_size * this->group_);

    if (this->bias_term_) {
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
    cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_, "deconv_forward");

    if (this->bias_term_) {
      void *args[] = { &bottom_data, &weight, &bias, &top_data };
      cuLaunchKernel(kernel, (this->N_FW_ - 1) / fw_div_N + 1,  // Grid X
                     (this->M_FW_ - 1) / fw_div_M + 1,  // Grid Y
                     batch_size * this->group_,               // Grid Z
                     fw_wgs0, fw_wgs1, 1,               // Local
                     0, NULL, args, 0);                 // Arguments
    } else {
      void *args[] = { &bottom_data, &weight, &top_data };
      cuLaunchKernel(kernel, (this->N_FW_ - 1) / fw_div_N + 1,  // Grid X
                     (this->M_FW_ - 1) / fw_div_M + 1,  // Grid Y
                     batch_size * this->group_,               // Grid Z
                     fw_wgs0, fw_wgs1, 1,               // Local
                     0, NULL, args, 0);                 // Arguments
    }
    cuCtxSynchronize();
  }
#endif  // USE_CUDA
}

template<typename Dtype>
void LibDNNDeconv<Dtype>::Backward(bool prop_down_data, bool prop_down_weights,
                                 const Dtype* top_data, const Dtype* top_diff,
                                 const Dtype* weight, Dtype* weight_diff,
                                 const Dtype* bias, Dtype* bias_diff,
                                 const Dtype* bottom_data, Dtype* bottom_diff,
                                 int_tp batch_size) {
  int bw_wptn = this->bw_tuner_->template get_param<int>("WPTN");
  int bw_wptm = this->bw_tuner_->template get_param<int>("WPTM");
  int bw_wgs0 = this->bw_tuner_->template get_param<int>("workgroup_size_0");
  int bw_wgs1 = this->bw_tuner_->template get_param<int>("workgroup_size_1");
  int bw_div_N = bw_wptn * bw_wgs0;
  int bw_div_M = bw_wptm * bw_wgs1;

  int wg_wptn = this->wg_tuner_->template get_param<int>("WPTN");
  int wg_wptm = this->wg_tuner_->template get_param<int>("WPTM");
  int wg_wgs0 = this->wg_tuner_->template get_param<int>("workgroup_size_0");
  int wg_wgs1 = this->wg_tuner_->template get_param<int>("workgroup_size_1");
  int wg_div_N = wg_wptn * wg_wgs0;
  int wg_div_M = wg_wptm * wg_wgs1;

#ifdef USE_GREENTEA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
    // Backprop w.r.t. data
    if (prop_down_data) {
      viennacl::ocl::kernel &kernel =
          LibDNN<Dtype>::ocl_program_.get_kernel("deconv_backward");
      viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

      kernel.local_work_size(0, bw_wgs0);
      kernel.local_work_size(1, bw_wgs1);
      kernel.local_work_size(2, 1);

      kernel.global_work_size(0, ((this->N_BW_ - 1) / bw_div_N + 1) * bw_wgs0);
      kernel.global_work_size(1, ((this->M_BW_ - 1) / bw_div_M + 1) * bw_wgs1);
      kernel.global_work_size(2, batch_size * this->group_);

      if (this->bias_term_) {
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
    if (prop_down_weights &&
        (this->weights_backward_ || this->bias_backward_)) {
      viennacl::ocl::kernel &kernel =
          LibDNN<Dtype>::ocl_program_.get_kernel("deconv_weights");

      viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

      kernel.local_work_size(0, wg_wgs0);
      kernel.local_work_size(1, wg_wgs1);
      kernel.local_work_size(2, 1);

      kernel.global_work_size(0, ((this->N_WG_ - 1) / wg_div_N + 1) * wg_wgs0);
      kernel.global_work_size(1, ((this->M_WG_ - 1) / wg_div_M + 1) * wg_wgs1);

      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
        kernel.global_work_size(2, this->group_);
      }
      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
        kernel.global_work_size(2, batch_size * this->group_);
      }

      if (this->bias_term_) {
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
    // Backprop w.r.t. weights and bias
    if (prop_down_weights && this->bias_term_ &&
        (this->weights_backward_ || this->bias_backward_)) {
      viennacl::ocl::kernel &kernel =
          LibDNN<Dtype>::ocl_program_.get_kernel("deconv_weights_bias");

      viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

      kernel.local_work_size(0, wg_wgs0);
      kernel.local_work_size(1, wg_wgs1);
      kernel.local_work_size(2, 1);

      kernel.global_work_size(0, ((this->N_BG_ - 1) / wg_div_N + 1) * wg_wgs0);
      kernel.global_work_size(1, ((this->M_BG_ - 1) / wg_div_M + 1) * wg_wgs1);

      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
        kernel.global_work_size(2, this->group_);
      }
      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
        kernel.global_work_size(2, batch_size * this->group_);
      }

      viennacl::ocl::enqueue(
          kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                 WrapHandle((cl_mem) top_diff, &ctx),
                 WrapHandle((cl_mem) bias_diff, &ctx),
                 WrapHandle((cl_mem) weight_diff, &ctx), batch_size),
          ctx.get_queue());
    }
  }
#endif  // USE_GREENTEA

#ifdef USE_CUDA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    // Backprop w.r.t. data
    if (prop_down_data) {
      CUfunction kernel;
      cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                          "deconv_backward");

      if (this->bias_term_) {
        void *args[] = { &top_diff, &weight, &bias, &bottom_diff };
        cuLaunchKernel(kernel, (this->N_BW_ - 1) / bw_div_N + 1,  // Grid X
                       (this->M_BW_ - 1) / bw_div_M + 1,  // Grid Y
                       batch_size * this->group_,               // Grid Z
                       bw_wgs0, bw_wgs1, 1,               // Local
                       0, NULL, args, 0);                 // Arguments
      } else {
        void *args[] = { &top_diff, &weight, &bottom_diff };
        cuLaunchKernel(kernel, (this->N_BW_ - 1) / bw_div_N + 1,  // Grid X
                       (this->M_BW_ - 1) / bw_div_M + 1,  // Grid Y
                       batch_size * this->group_,               // Grid Z
                       bw_wgs0, bw_wgs1, 1,               // Local
                       0, NULL, args, 0);                 // Arguments
      }
    }

    // Backprop w.r.t. weights and bias
    if (prop_down_weights &&
        (this->weights_backward_ || this->bias_backward_)) {
      CUfunction kernel;
      cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                          "deconv_weights");

      int gws2 = 0;

      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
        gws2 = this->group_;
      }
      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
        gws2 = batch_size * this->group_;
      }

      if (this->bias_term_) {
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
    if (prop_down_weights && this->bias_term_ &&
        (this->weights_backward_ || this->bias_backward_)) {
      CUfunction kernel;
      cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                          "deconv_weights_bias");
      int gws2 = 0;
      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
        gws2 = this->group_;
      }
      if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
        gws2 = batch_size * this->group_;
      }
      void *args[] = { &bottom_data, &top_diff, &bias_diff, &weight_diff,
          &batch_size };
      cuLaunchKernel(kernel, (this->N_BG_ - 1) / wg_div_N + 1,  // Grid X
                     (this->M_BG_ - 1) / wg_div_M + 1,  // Grid Y
                     gws2,                              // Grid Z
                     wg_wgs0, wg_wgs1, 1,               // Local
                     0, NULL, args, 0);                 // Arguments
    }
  }
#endif  // USE_CUDA
}

template<typename Dtype>
void LibDNNDeconv<Dtype>::Tune(Dtype* top_data, Dtype* top_diff, Dtype* weight,
                             Dtype* weight_diff, Dtype* bias, Dtype* bias_diff,
                             Dtype* bottom_data, Dtype* bottom_diff,
                             int_tp batch_size) {
  LibDNNDeconv* self = this;
  // Autotune forward kernel
  this->fw_tuner_->set_setup_routine([&]() -> bool {
    try {
      self->GenerateKernels();
      return self->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  this->fw_tuner_->set_benchmark_routine([&]() -> double {
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
  this->fw_tuner_->Tune(LIBDNN_TUNER_METHOD_ANNEALING);

  // Autotune backward kernel
  this->bw_tuner_->set_setup_routine([&]() -> bool {
    try {
      self->GenerateKernels();
      return self->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  this->bw_tuner_->set_benchmark_routine([&]() -> double {
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
  this->bw_tuner_->Tune(LIBDNN_TUNER_METHOD_ANNEALING);

  // Autotune weight/bias error kernel
  this->wg_tuner_->set_setup_routine([&]() -> bool {
    try {
      self->GenerateKernels();
      return self->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  this->wg_tuner_->set_benchmark_routine([&]() -> double {
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
  this->wg_tuner_->Tune(LIBDNN_TUNER_METHOD_ANNEALING);
}

INSTANTIATE_CLASS(LibDNNDeconv);

}  // namespace caffe

#endif  // USE_LIBDNN
