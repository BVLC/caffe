#ifdef USE_LIBDNN

#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/libdnn/libdnn_conv.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
LibDNNDeconv<MItype, MOtype>::LibDNNDeconv(LibDNNDeconvConfig config)
      : LibDNNConv<MItype, MOtype>(config.dev_ptr) {
  this->config_ = config;
  this->program_ = this->dev_ptr_->CreateProgram();
  this->bias_term_ = config.bias_term;
  this->fast_unsafe_math_ = config.fast_unsafe_math;
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

  this->bw_tuner_ = shared_ptr<LibDNNTuner>(new LibDNNTuner());
  this->fw_tuner_ = shared_ptr<LibDNNTuner>(new LibDNNTuner());
  this->wg_tuner_ = shared_ptr<LibDNNTuner>(new LibDNNTuner());

  // Setup tuning parameters

  // Work groups
  for (int id = 0; id < 2; ++id) {
    vector<int_tp> workgroup_sizes;
    workgroup_sizes.push_back(1);
    workgroup_sizes.push_back(2);
    for (int_tp i = 4; i < this->dev_ptr_->workgroup_size(id);
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

  this->bw_tuner_->template add_set_param<int_tp>("VWM", 4, vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->fw_tuner_->template add_set_param<int_tp>("VWM", 4, vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->wg_tuner_->template add_set_param<int_tp>("VWM", 4, vector<int_tp>(
      {1, 2, 4, 8, 16 }));

  this->bw_tuner_->template add_range_param<int_tp>("WPTN", 4, 2, 16, 2);
  this->fw_tuner_->template add_range_param<int_tp>("WPTN", 4, 2, 16, 2);
  this->wg_tuner_->template add_range_param<int_tp>("WPTN", 4, 2, 16, 2);

  this->bw_tuner_->template add_set_param<int_tp>("VWN", 4, vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->fw_tuner_->template add_set_param<int_tp>("VWN", 4, vector<int_tp>(
      {1, 2, 4, 8, 16 }));
  this->wg_tuner_->template add_set_param<int_tp>("VWN", 4, vector<int_tp>(
      {1, 2, 4, 8, 16 }));

  // Constraint using TSK, TSM, RTSM and RTSN. Adapt TSK if constraint fails.
  this->bw_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "WPTM", "workgroup_size_1"}),
    vector<string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "WPTM", "workgroup_size_1"}), vector<
    string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "WPTM", "workgroup_size_1"}), vector<
    string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  // Constraint using TSK, TSN, RTSN and RTSM. Adapt TSK if constraint fails.
  this->bw_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "WPTN", "workgroup_size_0"}),
    vector<string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "WPTN", "workgroup_size_0"}),
    vector<string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "WPTN", "workgroup_size_0"}),
    vector<string>({"TSK"}), [](vector<int64_t> args) -> bool {
      return (args[0] * args[1]) % (args[2]) == 0;
    });
  this->bw_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "TSK_UNROLL"}),
    vector<string>({"TSK_UNROLL"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "TSK_UNROLL"}),
    vector<string>({"TSK_UNROLL"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    vector<string>({"TSK", "TSK_UNROLL"}),
    vector<string>({"TSK_UNROLL"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->bw_tuner_->template add_constraint<int64_t>(
    vector<string>({"WPTM", "VWM"}),
    vector<string>({"WPTM"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    vector<string>({"WPTM", "VWM"}),
    vector<string>({"WPTM"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    vector<string>({"WPTM", "VWM"}),
    vector<string>({"WPTM"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->bw_tuner_->template add_constraint<int64_t>(
    vector<string>({"WPTN", "VWN"}),
    vector<string>({"WPTN"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->fw_tuner_->template add_constraint<int64_t>(
    vector<string>({"WPTN", "VWN"}),
    vector<string>({"WPTN"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });
  this->wg_tuner_->template add_constraint<int64_t>(
    vector<string>({"WPTN", "VWN"}),
    vector<string>({"WPTN"}),
    [](vector<int64_t> args) -> bool {
      return args[0] % args[1] == 0;
    });

  // this->pad_A, this->pad_B
  this->bw_tuner_->template
      add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  this->fw_tuner_->template
      add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  this->wg_tuner_->template
      add_range_param<int_tp>("lmem_pad_A", 0, 0, 8, 1);
  this->bw_tuner_->template
      add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);
  this->fw_tuner_->template
      add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);
  this->wg_tuner_->template
      add_range_param<int_tp>("lmem_pad_B", 0, 0, 8, 1);

  if (this->dev_ptr_->backend() == BACKEND_CUDA) {
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

  this->GenerateKernels();
  this->CompileKernels();
}

template<typename MItype, typename MOtype>
const LibDNNDeconvConfig LibDNNDeconv<MItype, MOtype>::get_config() {
  return config_;
}

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::string_identifier() {
  stringstream ss;
  ss << "DECONV_";
  // Type names
  ss << safe_type_name<MItype>() << "_";
  ss << safe_type_name<MItype>() << "_";
  ss << safe_type_name<MOtype>() << "_";
  // Device name
  ss << this->dev_ptr_->name();
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

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::generate_bw_defs() {
  stringstream ss;

  // Number of spatial axes
  ss << this->program_->define("v_nax", this->num_axes_);

  // Groups
  ss << this->program_->define("v_g", this->group_);

  int_tp B_off = this->fmaps_out_;
  int_tp C_off = this->fmaps_in_;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    B_off *= this->im_out_shape_[i];
    C_off *= this->im_in_shape_[i];
  }
  // Input image batch offset
  ss << this->program_->define("v_B_off", B_off);
  // Output image batch offset
  ss << this->program_->define("v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    ss << this->program_->define("v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    ss << this->program_->define("v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  ss << this->program_->define("v_imsi", imsi);
  ss << this->program_->define("v_imso", imso);

  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    ss << this->program_->define("v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
  }

  for (int_tp i = 0; i < this->pad_.size(); ++i) {
    ss << this->program_->define("v_p_" + std::to_string(i), this->pad_[i]);
  }

  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    ss << this->program_->define("v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    ss << this->program_->define("v_d_" + std::to_string(i), this->dilation_[i]);
  }

  ss << this->program_->define("v_fin", this->fmaps_in_);
  ss << this->program_->define("v_fout", this->fmaps_out_);

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
  ss << this->program_->define("MG", this->MG_BW_);
  ss << this->program_->define("M", this->M_BW_);
  ss << this->program_->define("N", this->N_BW_);
  ss << this->program_->define("KG", this->KG_BW_);
  ss << this->program_->define("K", this->K_BW_);

  // Local memory padding
  ss << this->program_->define("v_pad_A",
                         this->bw_tuner_->template
                         get_param<int>("lmem_pad_A"));
  ss << this->program_->define("v_pad_B",
                         this->bw_tuner_->template
                         get_param<int>("lmem_pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  ss << this->program_->define("TSM",
          this->bw_tuner_->template get_param<int>("WPTM")
          * this->bw_tuner_->template get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  ss << this->program_->define("TSN",
          this->bw_tuner_->template get_param<int>("WPTN")
          * this->bw_tuner_->template get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  ss << this->program_->define("TSK", this->bw_tuner_->template
                         get_param<int>("TSK"));
  // TSK unrolling
  ss << this->program_->define("TSK_UNROLL",
                         this->bw_tuner_->template
                         get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  ss << this->program_->define("WPTM", this->bw_tuner_->template
                         get_param<int>("WPTM"));
  ss << this->program_->define("VWM", this->bw_tuner_->template
                         get_param<int>("VWM"));
  // The work-per-thread in dimension N
  ss << this->program_->define("WPTN", this->bw_tuner_->template
                         get_param<int>("WPTN"));
  ss << this->program_->define("VWN", this->bw_tuner_->template
                         get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  ss << this->program_->define("RTSM",
                         this->bw_tuner_->template
                         get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  ss << this->program_->define("RTSN",
                         this->bw_tuner_->template
                         get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  ss << this->program_->define("LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  ss << this->program_->define("LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  ss << this->program_->define("v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::generate_fw_defs() {
  stringstream ss;

  // Number of spatial axes
  ss << this->program_->define("v_nax", this->num_axes_);

  // Groups
  ss << this->program_->define("v_g", this->group_);

  int_tp A_off = this->fmaps_in_ * this->fmaps_out_;
  int_tp B_off = this->fmaps_in_;
  int_tp C_off = this->fmaps_out_;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    A_off *= this->kernel_shape_[i];
    B_off *= this->im_in_shape_[i];
    C_off *= this->im_out_shape_[i];
  }

  // Weight offset (only used for groups)
  ss << this->program_->define("v_A_off", A_off);
  // Input image batch offset
  ss << this->program_->define("v_B_off", B_off);
  // Output image batch offset
  ss << this->program_->define("v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    ss << this->program_->define("v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    ss << this->program_->define("v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  ss << this->program_->define("v_imsi", imsi);
  ss << this->program_->define("v_imso", imso);

  int_tp v_ks = 1;
  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    ss << this->program_->define("v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
    v_ks *= this->kernel_shape_[i];
  }
  ss << this->program_->define("v_ks", v_ks);

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    // Set padding to account for padding loss (backward),
    // remove forward padding
    for (int_tp i = 0; i < this->pad_.size(); ++i) {
      ss << this->program_->define("v_p_" + std::to_string(i),
            (this->kernel_shape_[i] - 1) * this->dilation_[i] - this->pad_[i]);
    }
  }

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC) {
    for (int_tp i = 0; i < this->pad_.size(); ++i) {
      ss << this->program_->define("v_p_" + std::to_string(i), this->pad_[i]);
    }
  }

  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    ss << this->program_->define("v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    ss << this->program_->define("v_d_" + std::to_string(i), this->dilation_[i]);
  }

  ss << this->program_->define("v_fin", this->fmaps_in_);
  ss << this->program_->define("v_fout", this->fmaps_out_);

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
  ss << this->program_->define("MG", this->MG_FW_);
  ss << this->program_->define("M", this->M_FW_);
  ss << this->program_->define("N", this->N_FW_);
  ss << this->program_->define("KG", this->KG_FW_);
  ss << this->program_->define("K", this->K_FW_);

  // Local memory padding
  ss << this->program_->define("v_pad_A",
                         this->fw_tuner_->template
                         get_param<int>("lmem_pad_A"));
  ss << this->program_->define("v_pad_B",
                         this->fw_tuner_->template
                         get_param<int>("lmem_pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  ss << this->program_->define("TSM",
                this->fw_tuner_->template get_param<int>("WPTM")
                * this->fw_tuner_->template get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  ss << this->program_->define("TSN",
                this->fw_tuner_->template get_param<int>("WPTN")
                * this->fw_tuner_->template get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  ss << this->program_->define("TSK", this->fw_tuner_->template
                         get_param<int>("TSK"));
  // TSK unrolling
  ss << this->program_->define("TSK_UNROLL",
                         this->fw_tuner_->template
                         get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  ss << this->program_->define("WPTM", this->fw_tuner_->template
                         get_param<int>("WPTM"));
  ss << this->program_->define("VWM", this->fw_tuner_->template
                         get_param<int>("VWM"));
  // The work-per-thread in dimension N
  ss << this->program_->define("WPTN", this->fw_tuner_->template
                         get_param<int>("WPTN"));
  ss << this->program_->define("VWN", this->fw_tuner_->template
                         get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  ss << this->program_->define("RTSM",
                         this->fw_tuner_->template
                         get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  ss << this->program_->define("RTSN",
                         this->fw_tuner_->template
                         get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  ss << this->program_->define("LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  ss << this->program_->define("LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  ss << this->program_->define("v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::generate_wg_defs() {
  stringstream ss;

  // Number of spatial axes
  ss << this->program_->define("v_nax", this->num_axes_);

  // Groups
  ss << this->program_->define("v_g", this->group_);

  int_tp A_off = this->fmaps_in_;
  int_tp B_off = this->fmaps_out_;
  int_tp C_off = this->fmaps_in_ * this->fmaps_out_;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    A_off *= this->im_in_shape_[i];
    B_off *= this->im_out_shape_[i];
    C_off *= this->kernel_shape_[i];
  }
  // Output image batch offset
  ss << this->program_->define("v_A_off", A_off);
  // Input image batch offset
  ss << this->program_->define("v_B_off", B_off);
  // Weights offset
  ss << this->program_->define("v_C_off", C_off);

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < this->im_in_shape_.size(); ++i) {
    ss << this->program_->define("v_imsi_" + std::to_string(i),
                           this->im_in_shape_[i]);
    imsi *= this->im_in_shape_[i];
    ss << this->program_->define("v_imso_" + std::to_string(i),
                           this->im_out_shape_[i]);
    imso *= this->im_out_shape_[i];
  }
  ss << this->program_->define("v_imsi", imsi);
  ss << this->program_->define("v_imso", imso);

  int_tp v_ks = 1;
  for (int_tp i = 0; i < this->kernel_shape_.size(); ++i) {
    ss << this->program_->define("v_k_" + std::to_string(i),
                           this->kernel_shape_[i]);
    v_ks *= this->kernel_shape_[i];
  }
  ss << this->program_->define("v_ks", v_ks);

  // Set padding to account for padding loss (backward), remove forward padding
  for (int_tp i = 0; i < this->pad_.size(); ++i) {
    ss << this->program_->define("v_p_" + std::to_string(i), this->pad_[i]);
  }

  for (int_tp i = 0; i < this->stride_.size(); ++i) {
    ss << this->program_->define("v_s_" + std::to_string(i), this->stride_[i]);
  }

  for (int_tp i = 0; i < this->dilation_.size(); ++i) {
    ss << this->program_->define("v_d_" + std::to_string(i), this->dilation_[i]);
  }

  ss << this->program_->define("v_fin", this->fmaps_in_);
  ss << this->program_->define("v_fout", this->fmaps_out_);

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
  ss << this->program_->define("MG", this->MG_WG_);
  ss << this->program_->define("M", this->M_WG_);
  ss << this->program_->define("N", this->N_WG_);
  ss << this->program_->define("NG", this->NG_WG_);
  ss << this->program_->define("K", this->K_WG_);
  ss << this->program_->define("MGB", this->MG_BG_);
  ss << this->program_->define("MB", this->M_BG_);
  ss << this->program_->define("NB", this->N_WG_);
  ss << this->program_->define("NGB", this->NG_WG_);
  ss << this->program_->define("KB", this->K_BG_);

  // Local memory padding
  ss << this->program_->define("v_pad_A",
                         this->wg_tuner_->template
                         get_param<int>("lmem_pad_A"));
  ss << this->program_->define("v_pad_B",
                         this->wg_tuner_->template
                         get_param<int>("lmem_pad_B"));

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  ss << this->program_->define("TSM",
                this->wg_tuner_->template get_param<int>("WPTM")
                * this->wg_tuner_->template get_param<int>("workgroup_size_1"));
  // The tile-size in dimension N
  ss << this->program_->define("TSN",
                this->wg_tuner_->template get_param<int>("WPTN")
                * this->wg_tuner_->template get_param<int>("workgroup_size_0"));
  // The tile-size in dimension K
  ss << this->program_->define("TSK", this->wg_tuner_->template
                         get_param<int>("TSK"));
  // TSK unrolling
  ss << this->program_->define("TSK_UNROLL",
                         this->wg_tuner_->template
                         get_param<int>("TSK_UNROLL"));
  // The work-per-thread in dimension M
  ss << this->program_->define("WPTM", this->wg_tuner_->template
                         get_param<int>("WPTM"));
  ss << this->program_->define("VWM", this->wg_tuner_->template
                         get_param<int>("VWM"));
  // The work-per-thread in dimension N
  ss << this->program_->define("WPTN", this->wg_tuner_->template
                         get_param<int>("WPTN"));
  ss << this->program_->define("VWN", this->wg_tuner_->template
                         get_param<int>("VWN"));
  // The reduced tile-size in dimension M
  ss << this->program_->define("RTSM",
                         this->wg_tuner_->template
                         get_param<int>("workgroup_size_1"));
  // The reduced tile-size in dimension N
  ss << this->program_->define("RTSN",
                         this->wg_tuner_->template
                         get_param<int>("workgroup_size_0"));
  // Loads-per-thread for A
  ss << this->program_->define("LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  ss << this->program_->define("LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  // Num tiles needs to be next higher even integer
  // (due to some quirky bug in AMD OpenCL 2.0 on Windows)
  ss << this->program_->define("v_num_tiles", "(((K - 1)/(TSK*2) + 1)*2)");
  ss << this->program_->define("v_num_tiles_B", "(((KB - 1)/(TSK*2) + 1)*2)");


  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::generate_bw_kernels(string name) {
  stringstream ss;

  int wptn = this->bw_tuner_->template get_param<int>("WPTN");
  int wptm = this->bw_tuner_->template get_param<int>("WPTM");
  int tsk = this->bw_tuner_->template get_param<int>("TSK");
  int rtsn = this->bw_tuner_->template get_param<int>("workgroup_size_0");
  int rtsm = this->bw_tuner_->template get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = this->bw_tuner_->template get_param<int>("VWM");
  int vwn = this->bw_tuner_->template get_param<int>("VWN");
  // int lpta = (tsm * tsk) / (rtsm * rtsn);
  // int lptb = (tsn * tsk) / (rtsm * rtsn);

  // Backward kernel
  KernelArgs args;
  args.push_back(this->program_->template create_kernel_arg<MOtype>("im_in",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  args.push_back(this->program_->template create_kernel_arg<MItype>("wg",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  if (this->bias_term_) {
    args.push_back(this->program_->template create_kernel_arg<MItype>("bias",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  }
  args.push_back(this->program_->template create_kernel_arg<MItype>("im_out",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  ss << this->program_->function(name, args);

  // Thread identifiers
  // Local row ID (max: RTSM=TSM/WPTM)
  ss << "const int_tp tidn = " << this->program_->local_id(0) << ";"
     << std::endl;
  // Local col ID (max: RTSN=TSN/WPTN)
  ss << "const int_tp tidm = " << this->program_->local_id(1) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN * " << this->program_->group_id(0) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM * " << this->program_->group_id(1) << ";"
     << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << this->program_->local_mem("MItype",
                      "Asub[" + std::to_string(tsm) + "]"
                      + "[" + std::to_string(tsk) + " + v_pad_A]") << ";"
                    << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << this->program_->local_mem("MItype",
                      "Bsub[" + std::to_string(tsk) + "]"
                      + "[" + std::to_string(tsn) + " + v_pad_B]") << ";"
                    << std::endl;

  // Batch and group
  if (this->group_ > 1) {
    ss << "int_tp group = " << this->program_->global_id(2) << " % v_g;"
       << std::endl;
    ss << "int_tp batch = " << this->program_->global_id(2) << " / v_g;"
       << std::endl;
  } else {
    ss << "int_tp batch = " << this->program_->global_id(2) << ";" << std::endl;
  }

  if (this->group_ > 1) {
    ss << this->program_->global_ptr("const MItype", "Aptr")
       << " = wg + group * (M * K);" << std::endl;
    ss << this->program_->global_ptr("const MItype", "Bptr")
       << " = im_in + v_B_off * batch + group * (v_B_off / v_g);" << std::endl;
    ss << this->program_->global_ptr("MItype", "Cptr")
       << " = im_out + v_C_off * batch + group * (M * N);" << std::endl;
  } else {
    ss << this->program_->global_ptr("const MItype", "Aptr") << " = wg;"
       << std::endl;
    ss << this->program_->global_ptr("const MItype", "Bptr")
       << " = im_in + v_B_off * batch;" << std::endl;
    ss << this->program_->global_ptr("MItype", "Cptr")
       << " = im_out + v_C_off * batch;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  ss << this->generate_accreg_init(this->bw_tuner_, false, false, false, false);

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
  ss << this->program_->local_barrier() << std::endl;

  // FIXME
  ss << this->generate_gemm_core(this->bw_tuner_, false, false, false)
     << std::endl;

  // Synchronize before loading the next tile
  ss << this->program_->local_barrier() << std::endl;

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
     << "((MItype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::generate_wg_kernels(string name) {
  stringstream ss;

  int wptn = this->wg_tuner_->template get_param<int>("WPTN");
  int wptm = this->wg_tuner_->template get_param<int>("WPTM");
  int tsk = this->wg_tuner_->template get_param<int>("TSK");
  int rtsn = this->wg_tuner_->template get_param<int>("workgroup_size_0");
  int rtsm = this->wg_tuner_->template get_param<int>("workgroup_size_1");
  int tsm = wptm * rtsm;
  int tsn = wptn * rtsn;
  int vwm = this->wg_tuner_->template get_param<int>("VWM");
  int vwn = this->wg_tuner_->template get_param<int>("VWN");
  // int lpta = (tsm * tsk) / (rtsm * rtsn);
  // int lptb = (tsn * tsk) / (rtsm * rtsn);

  // Weight kernel
  KernelArgs wg_args;
  wg_args.push_back(this->program_->template create_kernel_arg<MOtype>("im_in",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  wg_args.push_back(this->program_->template create_kernel_arg<MItype>("im_out",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  if (this->bias_term_) {
    wg_args.push_back(this->program_->template create_kernel_arg<MItype>("bias",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  }
  wg_args.push_back(this->program_->template create_kernel_arg<MItype>("wg",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  wg_args.push_back(this->program_->template create_kernel_arg<int_tp>(
                                               "batch_size", KERNEL_ARG_CONST));
  ss << this->program_->function(name, wg_args);

  // Thread identifiers
  // Local row ID (max: TSM/WPTM)
  ss << "const int_tp tidn = " << this->program_->local_id(0) << ";"
     << std::endl;
  // Local col ID (max: TSN/WPTN)
  ss << "const int_tp tidm = " << this->program_->local_id(1) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN * " << this->program_->group_id(0) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM * " << this->program_->group_id(1) << ";"
     << std::endl;

  // Local tile memory
  ss << this->program_->local_mem("MItype", "Asub["
                       + std::to_string(tsm) + "][" + std::to_string(tsk)
                       + " + v_pad_A]") << ";" << std::endl;
  ss << this->program_->local_mem("MItype", "Bsub["
                       + std::to_string(tsk) + "][" + std::to_string(tsn)
                       + " + v_pad_B]") << ";" << std::endl;

  // Batch and group
  if (this->group_ > 1) {
    ss << "int_tp group = " << this->program_->global_id(2) << " % v_g;"
       << std::endl;
    ss << "int_tp batch = " << this->program_->global_id(2) << " / v_g;"
       << std::endl;
  } else {
    ss << "int_tp batch = " << this->program_->global_id(2) << ";" << std::endl;
  }

  if (this->group_ > 1) {
    ss << this->program_->global_ptr("const MItype", "Aptr")
       << " = im_in + batch * v_A_off + group * (v_A_off / v_g);" << std::endl;
    ss << this->program_->global_ptr("const MItype", "Bptr")
       << " = im_out + batch * v_B_off + group * (v_B_off / v_g);" << std::endl;
    ss << this->program_->global_ptr("MItype", "Cptr")
       << " = wg + group * (M * N);" << std::endl;
  } else {
    ss << this->program_->global_ptr("const MItype", "Aptr")
       << " = im_in + batch * v_A_off;" << std::endl;
    ss << this->program_->global_ptr("const MItype", "Bptr")
       << " = im_out + batch * v_B_off;" << std::endl;
    ss << this->program_->global_ptr("MItype", "Cptr") << " = wg;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  // FIXME
  ss << this->generate_accreg_init(this->wg_tuner_, false,
                             this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT,
                             false, false);

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
  ss << this->program_->local_barrier() << std::endl;

  // FIXME
  ss << this->generate_gemm_core(this->wg_tuner_, false, false, false)
     << std::endl;

  // Synchronize before loading the next tile
  ss << this->program_->local_barrier() << std::endl;

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
       << "((MItype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
  }
  if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
    ss << this->program_->template atomic_add<MItype>(
        "&(Cptr[globalRow * N + globalCol])",
        "((MItype*)(&(Creg[wm][wn/VWN])))[wn%VWN]") << std::endl;
  }
  ss << "}" << std::endl;   // M-N-Guard
  ss << "}" << std::endl;   // For (N)
  ss << "}" << std::endl;   // For (M)
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;


  if (this->bias_term_) {
    // Bias kernel
    KernelArgs b_args;
    b_args.push_back(this->program_->template create_kernel_arg<MItype>("im_in",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    b_args.push_back(this->program_->template create_kernel_arg<MItype>(
     "im_out", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    if (this->bias_term_) {
      b_args.push_back(this->program_->template create_kernel_arg<MItype>(
                                                "bias_mult", KERNEL_ARG_CONST));
      b_args.push_back(this->program_->template create_kernel_arg<MItype>(
          "bias", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    }
    b_args.push_back(this->program_->template create_kernel_arg<MItype>("wg",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    b_args.push_back(this->program_->template create_kernel_arg<int_tp>(
                                               "batch_size", KERNEL_ARG_CONST));
    ss << this->program_->function(name + "_bias", b_args);

    // Thread identifiers
    // Local row ID (max: TSM/WPTM)
    ss << "const int_tp tidn = " << this->program_->local_id(0) << ";"
       << std::endl;
    // Local col ID (max: TSN/WPTN)
    ss << "const int_tp tidm = " << this->program_->local_id(1) << ";"
       << std::endl;
    // Work-group offset
    ss << "const int_tp offN = TSN * " << this->program_->group_id(0) << ";"
       << std::endl;
    // Work-group offset
    ss << "const int_tp offM = TSM * " << this->program_->group_id(1) << ";"
       << std::endl;

    // Local tile memory
    ss << this->program_->local_mem("MItype", "Asub["
                         + std::to_string(tsm) + "][" + std::to_string(tsk)
                         + " + v_pad_A]") << ";" << std::endl;

    // Batch and group
    if (this->group_ > 1) {
      ss << "int_tp group = " << this->program_->global_id(2) << " % v_g;"
         << std::endl;
      ss << "int_tp batch = " << this->program_->global_id(2) << " / v_g;"
         << std::endl;
    } else {
      ss << "int_tp batch = " << this->program_->global_id(2) << ";"
         << std::endl;
    }

    if (this->group_ > 1) {
      ss << this->program_->global_ptr("const MItype", "Aptr")
         << " = im_out + batch * v_B_off + group * (v_B_off / v_g);"
         << std::endl;
      if (this->bias_term_) {
        ss << this->program_->global_ptr("MItype", "Dptr")
           << " = bias + group * (v_fout / v_g);" << std::endl;
      }
    } else {
      ss << this->program_->global_ptr("const MItype", "Aptr")
         << " = im_out + batch * v_B_off;" << std::endl;
      if (this->bias_term_) {
        ss << this->program_->global_ptr("MItype", "Dptr") << " = bias;"
           << std::endl;
      }
    }

    // Initialize the accumulation registers
    ss << "{" << std::endl;  // Scoping for D registers

    bool unroll = this->wg_tuner_->template get_param<bool>("vector_unroll");

    ss << "MItype" << vwm << " Dreg[WPTM/VWM];" << std::endl;

    // Initialize the accumulation registers
    if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
      // Load
      ss << "#pragma unroll" << std::endl;
      ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
      ss << "int_tp globalRow = offM + tidm + wm * RTSM;"
         << std::endl;
      ss << "((MItype*)(&(Dreg[wm/VWM])))[wm%VWM] = Dptr[globalRow];"
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
      // Additional batch loop, keep the same accumulator for the
      // weight gradient
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
    ss << this->program_->local_barrier() << std::endl;

    ss << "MItype" << vwm << " Areg;" << std::endl;
    // Loop over the values of A single tile
    ss << "#pragma unroll 1" << std::endl;
    ss << "for (int_tp kt = 0; kt < TSK; kt += TSK_UNROLL) {" << std::endl;
    ss << "#pragma unroll "
       << this->wg_tuner_->template get_param<int>("TSK_UNROLL") << std::endl;
    ss << "for (int_tp ku=0; ku<TSK_UNROLL; ++ku) {" << std::endl;
    ss << "int_tp k = kt + ku;" << std::endl;

    // Perform the computation
    ss << "#pragma unroll" << std::endl;
    ss << "for (int_tp wm=0; wm<WPTM/VWM; ++wm) {" << std::endl;
    ss << "int_tp row = tidm + wm*VWM*RTSM;" << std::endl;
    for (int i = 0; i < vwm; ++i) {
      ss << "VEC_" << vwm << "_" << i << "(Areg)"
         << " = Asub[row + " << (i*rtsm) << "][k];" << std::endl;
    }
    if (unroll) {
      for (int i = 0; i < vwm; ++i) {
        ss << "VEC_" << vwm << "_" << i << "(Dreg[wm]) " << "+= VEC_" << vwm
           << "_" << i << "(Areg) * bias_mult;" << std::endl;
      }
    } else {
      ss << "Dreg[wm] += Areg * bias_mult;" << std::endl;
    }
    ss << "}" << std::endl;

    // Loop over A single tile
    ss << "}" << std::endl;
    ss << "}" << std::endl;

    // Synchronize before loading the next tile
    ss << this->program_->local_barrier() << std::endl;

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
      ss << "Dptr[globalRow] = ((MItype*)(&(Dreg[wm/VWM])))[wm%VWM];"
         << std::endl;
    }
    if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
      ss << this->program_->template atomic_add<MItype>("&(Dptr[globalRow])",
                             "((MItype*)(&(Dreg[wm/VWM])))[wm%VWM]") << std::endl;
    }
    ss << "}" << std::endl;
    ss << "}" << std::endl;   // For (M)
    ss << "}" << std::endl;   // Scoping for D registers

    // Kernel
    ss << "}" << std::endl;
  }

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNDeconv<MItype, MOtype>::generate_fw_kernels(string name) {
  stringstream ss;

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

  // Forward kernel
  KernelArgs args;
  args.push_back(this->program_->template create_kernel_arg<MItype>("im_out",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  args.push_back(this->program_->template create_kernel_arg<MItype>("wg",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  if (this->bias_term_) {
    args.push_back(this->program_->template create_kernel_arg<MItype>(
                                                "bias_mult", KERNEL_ARG_CONST));
    args.push_back(this->program_->template create_kernel_arg<MItype>("bias",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  }
  args.push_back(this->program_->template create_kernel_arg<MOtype>("im_in",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  ss << this->program_->function(name, args);

  // Thread identifiers
  // Local row ID (max: TSM/WPTM)
  ss << "const int_tp tidn = " << this->program_->local_id(0) << ";"
     << std::endl;
  // Local col ID (max: TSN/WPTN)
  ss << "const int_tp tidm = " << this->program_->local_id(1) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offN = TSN * " << this->program_->group_id(0) << ";"
     << std::endl;
  // Work-group offset
  ss << "const int_tp offM = TSM * " << this->program_->group_id(1) << ";"
     << std::endl;

  // Local tile memory
  // Asub for loading weights & shuffling the output
  ss << this->program_->local_mem("MItype",
                      "Asub[" + std::to_string(tsm) + "]"
                      + "[" + std::to_string(tsk) + " + v_pad_A]") << ";"
                    << std::endl;
  // Bsub for loading the input image and shuffling the output image
  ss << this->program_->local_mem("MItype",
                      "Bsub[" + std::to_string(tsk) + "]"
                      + "[" + std::to_string(tsn) + " + v_pad_B]") << ";"
                    << std::endl;

  // Batch and group
  if (this->group_ > 1) {
    ss << "int_tp group = " << this->program_->global_id(2) << " % v_g;"
       << std::endl;
    ss << "int_tp batch = " << this->program_->global_id(2) << " / v_g;"
       << std::endl;
  } else {
    ss << "int_tp batch = " << this->program_->global_id(2) << ";" << std::endl;
  }

  if (this->group_ > 1) {
    ss << this->program_->global_ptr("const MItype", "Aptr")
       << " = wg + group * (v_A_off / (v_g * v_g));" << std::endl;
    ss << this->program_->global_ptr("const MItype", "Bptr")
       << "= im_out + v_B_off * batch + group * (v_B_off / v_g);" << std::endl;
    ss << this->program_->global_ptr("MItype", "Cptr")
       << " = im_in + v_C_off * batch + group * (v_C_off / v_g);" << std::endl;
    if (this->bias_term_) {
      ss << this->program_->global_ptr("const MItype", "Dptr")
         << " = bias + group * (v_fout / v_g);" << std::endl;
    }
  } else {
    ss << this->program_->global_ptr("const MItype", "Aptr") << " = wg;"
       << std::endl;
    ss << this->program_->global_ptr("const MItype", "Bptr")
       << " = im_out + v_B_off * batch;" << std::endl;
    ss << this->program_->global_ptr("MItype", "Cptr")
       << " = im_in + v_C_off * batch;" << std::endl;
    if (this->bias_term_) {
      ss << this->program_->global_ptr("const MItype", "Dptr") << " = bias;"
         << std::endl;
    }
  }


  // Initialize the accumulation registers
  ss << "{" << std::endl;  // Scoping for C registers
  // FIXME
  ss << this->generate_accreg_init(this->fw_tuner_, false, false, false, false);

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
  ss << this->program_->local_barrier() << std::endl;
  // FIXME
  ss << this->generate_gemm_core(this->fw_tuner_, false, false, false)
     << std::endl;

  // Synchronize before loading the next tile
  ss << this->program_->local_barrier() << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;
  ss << "}" << std::endl;  // Scoping for load & compute block

  // Store the final results in C
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm * RTSM;" <<std::endl;
  if (this->bias_term_) {
    ss << "MItype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "#pragma unroll" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn * RTSN;" << std::endl;

  if (this->bwalgo_ == LIBDNN_CONVOLUTION_BW_ALGO_IM2COL) {
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "Cptr[globalRow * N + globalCol] = ";
    if (this->bias_term_) {
      ss << "((MItype*)(&(Creg[wm][wn/VWN])))[wn%VWN]"
         << " + bias_mult * biasval;" << std::endl;
    } else {
      ss << "((MItype*)(&(Creg[wm][wn/VWN])))[wn%VWN];" << std::endl;
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
    ss << this->program_->template atomic_add<MItype>("(&(Cptr[tiledIndex])",
                       "((MItype*)(&(Creg[wm][wn/VWN])))[wn%VWN]") << std::endl;
    ss << "}" << std::endl;
  }

  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;   // Scoping for C registers

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename MItype, typename MOtype>
void LibDNNDeconv<MItype, MOtype>::GenerateKernels() {
  this->program_ = this->dev_ptr_->CreateProgram();

  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int16_t,
          typename std::conditional<sizeof(MItype) == 2, int32_t,
                                    int64_t>::type>::type>::type Difftype;
  typedef typename std::conditional<float_is_same<MItype>::value, MItype,
          typename std::conditional<sizeof(MItype) == 1, int32_t,
                                    int64_t>::type>::type Acctype;

  stringstream ss;
  ss << this->program_->setup();
  ss << this->program_->template define_vector_type<MItype>("MItype", 0, 16);
  ss << this->program_->template define_vector_type<MItype>("MItype", 0, 16);
  ss << this->program_->template define_vector_type<MItype>("MOtype", 0, 16);
  ss << this->program_->template define_vector_type<Acctype>("Acctype", 0, 16);
  ss << this->program_->template define_vector_type<Difftype>("Difftype",
                                                              0, 16);
  if (is_integer_type<MItype>()) {
    if (this->dev_ptr_->template preferred_vector_width<int64_t>() > 0) {
      ss << this->program_->template define_vector_type<int64_t>("Multtype",
                                                                0, 16);
    } else {
      ss << this->program_->template define_vector_type<int32_t>("Multtype",
                                                                0, 16);
    }
  }

  ss << this->program_->atomics();
  ss << this->program_->vector_accessors();

  // Deconvolution kernels only for float types
  // TODO: Forward deconvolution quantized
  if (is_float_type<MItype>()) {
    ss << generate_fw_defs();
    ss << generate_fw_kernels("deconv_forward");
    ss << generate_bw_defs();
    ss << generate_bw_kernels("deconv_backward");
    ss << generate_wg_defs();
    ss << generate_wg_kernels("deconv_weights");
  }

  // Write complete kernel string
  this->program_->set_source(ss.str());
}

template<typename MItype, typename MOtype>
bool LibDNNDeconv<MItype, MOtype>::CompileKernels() {
  return this->program_->Compile(true, true);
}

template<typename MItype, typename MOtype>
void LibDNNDeconv<MItype, MOtype>::Forward(
    vptr<const MItype> bottom_data, vptr<const MItype> weight, MItype bias_mult,
    vptr<const MItype> bias, vptr<MOtype> top_data, int_tp batch_size) {
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
    this->dev_ptr_->template set<MItype>(ims, (MItype)0, top_data);
  }

  shared_ptr<DeviceKernel> kernel =
      this->program_->GetKernel("deconv_forward");
  vector<size_t> group = {static_cast<size_t>(
                                            ((this->N_FW_ - 1) / fw_div_N + 1)),
                          static_cast<size_t>(
                                            ((this->M_FW_ - 1) / fw_div_M + 1)),
                          static_cast<size_t>(batch_size * this->group_)};
  vector<size_t> local = {static_cast<size_t>(fw_wgs0),
                          static_cast<size_t>(fw_wgs1), 1};

  kernel->add_arg(&bottom_data);
  kernel->add_arg(&weight);
  if (this->bias_term_) {
    kernel->add_arg(&bias_mult);
    kernel->add_arg(&bias);
  }
  kernel->add_arg(&top_data);
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
void LibDNNDeconv<MItype, MOtype>::Backward(
                       bool prop_down_data, bool prop_down_weights,
                       vptr<const MOtype> top_data, vptr<const MOtype> top_diff,
                       vptr<const MItype> weight, vptr<MItype> weight_diff,
                       MItype bias_mult,
                       vptr<const MItype> bias, vptr<MItype> bias_diff,
                       vptr<const MItype> bottom_data, vptr<MItype> bottom_diff,
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

  // Backprop w.r.t. data
  if (prop_down_data) {
    shared_ptr<DeviceKernel> kernel =
        this->program_->GetKernel("deconv_backward");
    vector<size_t> group = {static_cast<size_t>(
                                            ((this->N_BW_ - 1) / bw_div_N + 1)),
                            static_cast<size_t>(
                                             ((this->M_BW_ - 1) / bw_div_M + 1)),
                            static_cast<size_t>(batch_size * this->group_)};
    vector<size_t> local = {static_cast<size_t>(bw_wgs0),
                            static_cast<size_t>(bw_wgs1), 1};


    kernel->add_arg(&top_diff);
    kernel->add_arg(&weight);
    if (this->bias_term_) {
      kernel->add_arg(&bias_mult);
      kernel->add_arg(&bias);
    }
    kernel->add_arg(&bottom_diff);
    kernel->Execute(group, local);
  }

  // Backprop w.r.t. weights and bias
  if (prop_down_weights && (this->weights_backward_ || this->bias_backward_)) {
    shared_ptr<DeviceKernel> kernel =
        this->program_->GetKernel("deconv_weights");
    vector<size_t> group = {static_cast<size_t>(
                                            ((this->N_WG_ - 1) / wg_div_N + 1)),
                            static_cast<size_t>(
                                            ((this->M_WG_ - 1) / wg_div_M + 1)),
                            static_cast<size_t>(batch_size * this->group_)};
    vector<size_t> local = {static_cast<size_t>(wg_wgs0),
                            static_cast<size_t>(wg_wgs1), 1};

    if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
      group[2] = this->group_;
    }
    if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
      group[2] = batch_size * this->group_;
    }


    kernel->add_arg(&bottom_data);
    kernel->add_arg(&top_diff);
    if (this->bias_term_) {
      kernel->add_arg(&bias_diff);
    }
    kernel->add_arg(&weight_diff);
    kernel->add_arg(&batch_size);
    kernel->Execute(group, local);
  }

  // Backprop w.r.t. weights and bias
  if (prop_down_weights && this->bias_term_
      && (this->weights_backward_ || this->bias_backward_)) {
    shared_ptr<DeviceKernel> kernel =
        this->program_->GetKernel("deconv_weights_bias");
    vector<size_t> group = {static_cast<size_t>(
                                            ((this->N_BG_ - 1) / wg_div_N + 1)),
                            static_cast<size_t>(
                                            ((this->M_BG_ - 1) / wg_div_M + 1)),
                            static_cast<size_t>(batch_size * this->group_)};
    vector<size_t> local = {static_cast<size_t>(wg_wgs0),
                            static_cast<size_t>(wg_wgs1), 1};

    if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
      group[2] = this->group_;
    }
    if (this->wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC) {
      group[2] = batch_size * this->group_;
    }

    kernel->add_arg(&bottom_data);
    kernel->add_arg(&top_diff);
    kernel->add_arg(&bias_mult);
    kernel->add_arg(&bias_diff);
    kernel->add_arg(&weight_diff);
    kernel->add_arg(&batch_size);
    kernel->Execute(group, local);
  }
}

template<typename MItype, typename MOtype>
void LibDNNDeconv<MItype, MOtype>::Tune(
                  vptr<MOtype> top_data, vptr<MOtype> top_diff,
                  vptr<MItype> weight, vptr<MItype> weight_diff,
                  const MItype bias_mult,
                  vptr<MItype> bias, vptr<MItype> bias_diff,
                  vptr<MItype> bottom_data, vptr<MItype> bottom_diff,
                  int_tp batch_size) {
  // Autotune forward kernel
  this->fw_tuner_->set_setup_routine([&]() -> bool {
    try {
      this->GenerateKernels();
      return this->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  this->fw_tuner_->set_benchmark_routine([&]() -> double {
    try {
      Timer timer;
      timer.Start();
      this->Forward(bottom_data, weight, bias_mult, bias, top_data, batch_size);
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
      this->GenerateKernels();
      return this->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  this->bw_tuner_->set_benchmark_routine([&]() -> double {
    try {
      Timer timer;
      timer.Start();
      this->Backward(true, false,
          top_data, top_diff,
          weight, weight_diff,
          bias_mult,
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
      this->GenerateKernels();
      return this->CompileKernels();
    } catch(...) {
      return false;
    }
  });
  this->wg_tuner_->set_benchmark_routine([&]() -> double {
    try {
      Timer timer;
      timer.Start();
      this->Backward(false, true,
          top_data, top_diff,
          weight, weight_diff,
          bias_mult, bias, bias_diff,
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

INSTANTIATE_CLASS_2T_GUARDED(LibDNNDeconv, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
