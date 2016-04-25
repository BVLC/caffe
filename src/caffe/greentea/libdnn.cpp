#include <string>

#include "caffe/common.hpp"
#ifdef USE_GREENTEA
#include "caffe/device.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/libdnn.hpp"


namespace caffe {

template<typename Dtype>
libdnn_conv<Dtype>::libdnn_conv(libdnn_config config) {
  dev_ptr_ = config.dev_ptr;
  bias_term_ = config.bias_term;
  bias_multiplier_ = config.bias_term ? 1.0 : 0.0;
  fast_unsafe_math_ = config.fast_unsafe_math;
  int_tp dims = config.in_shape.size();
  int_tp spatial_dims = config.kernel.size();

  num_axes_ = spatial_dims;
  fmaps_in_ = config.in_shape[dims - spatial_dims - 1];
  fmaps_out_ = config.out_shape[dims - spatial_dims - 1];
  group_ = config.group;

  wgalgo_ = config.wgalgo;

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

  generate_kernels();
  compile_kernels(&(viennacl::ocl::get_context(dev_ptr_->id())));
}

template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_header() {
  std::stringstream ss;
  if (std::is_same<Dtype, double>::value) {
    // Test/enable KHR 64 bit (double)
    ss << "#if defined(cl_khr_fp64)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
    ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;

    // Test/enable AMD 64 bit (double)
    ss << "#elif defined(cl_amd_fp64)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable" << std::endl;
    ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;
    ss << "#endif" << std::endl;
  }

  // 64 bit integers
  if (sizeof(int_tp) == 8) {
    // Test/enable 64 bit atomics
    ss << "#if defined(cl_khr_int64_base_atomics)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
       << std::endl;
    ss << "#define ATOMICS_64_AVAILABLE" << std::endl;
    ss << "#endif" << std::endl;
  }

  if (std::is_same<Dtype, double>::value) {
    ss << "#define Dtype double" << std::endl;
  } else {
    ss << "#define Dtype float" << std::endl;
  }

  if (sizeof(int_tp) == 8) {
    ss << "#define int_tp long" << std::endl;
    ss << "#define uint_tp unsigned long" << std::endl;
    ss << "#define int_tpc long" << std::endl;
    ss << "#define uint_tpc unsigned long" << std::endl;
  } else {
    ss << "#define int_tp int" << std::endl;
    ss << "#define uint_tp unsigned int" << std::endl;
    ss << "#define int_tpc int" << std::endl;
    ss << "#define uint_tpc unsigned int" << std::endl;
  }

  return ss.str();
}

template<typename Dtype>
template<class T>
inline void libdnn_conv<Dtype>::add_def(std::stringstream& ss,  // NOLINT
                                        const char* name, T value) {
  ss << "#ifdef " << name << std::endl;
  ss << "#undef " << name << std::endl;
  ss << "#endif" << std::endl;
  if (std::is_same<T, float>::value) {
    ss << "#define " << name << " (float) "
        << std::setprecision(32) << value << std::endl;
  } else if (std::is_same<T, double>::value) {
    ss << "#define " << name << " (double) "
        << std::setprecision(32) << value << std::endl;
  } else {
    ss << "#define " << name << " " << value << std::endl;
  }
}

template<typename Dtype>
template<class T>
inline void libdnn_conv<Dtype>::add_def(std::stringstream& ss,  // NOLINT
                                        const std::string name, T value) {
  add_def(ss, name.c_str(), value);
}



template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_fw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  add_def(ss, "v_nax", num_axes_);

  // Groups
  add_def(ss, "v_g", group_);

  int_tp B_off = fmaps_in_;
  int_tp C_off = fmaps_out_;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    B_off *= im_in_shape_[i];
    C_off *= im_out_shape_[i];
  }
  // Input image batch offset
  add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  add_def(ss, "v_C_off", C_off);

  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
    add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
  }

  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
  }

  for (int_tp i = 0; i < pad_.size(); ++i) {
    add_def(ss, "v_p_" + std::to_string(i), pad_[i]);
  }

  for (int_tp i = 0; i < stride_.size(); ++i) {
    add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
  }

  for (int_tp i = 0; i < dilation_.size(); ++i) {
    add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
  }

  add_def(ss, "v_fin", fmaps_in_);
  add_def(ss, "v_fout", fmaps_out_);

  if (bias_term_) {
    add_def(ss, "v_bmul", bias_multiplier_);
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
  add_def(ss, "MG", MG_FW_);
  add_def(ss, "M", M_FW_);
  add_def(ss, "N", N_FW_);
  add_def(ss, "KG", KG_FW_);
  add_def(ss, "K", K_FW_);

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  add_def(ss, "TSM", 64);
  // The tile-size in dimension N
  add_def(ss, "TSN", 64);
  // The tile-size in dimension K
  add_def(ss, "TSK", 16);
  // The work-per-thread in dimension M
  add_def(ss, "WPTM", 4);
  // The work-per-thread in dimension N
  add_def(ss, "WPTN", 4);
  // The reduced tile-size in dimension M
  add_def(ss, "RTSM", "(TSM/WPTM)");
  // The reduced tile-size in dimension N
  add_def(ss, "RTSN", "(TSN/WPTN)");
  // Loads-per-thread for A
  add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  return ss.str();
}

template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_bw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  add_def(ss, "v_nax", num_axes_);

  // Groups
  add_def(ss, "v_g", group_);

  int_tp B_off = fmaps_out_;
  int_tp C_off = fmaps_in_;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    B_off *= im_out_shape_[i];
    C_off *= im_in_shape_[i];
  }
  // Input image batch offset
  add_def(ss, "v_B_off", B_off);
  // Output image batch offset
  add_def(ss, "v_C_off", C_off);

  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
    add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
  }

  int_tp v_ks = 1;
  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
    v_ks *= kernel_shape_[i];
  }
  add_def(ss, "v_ks", v_ks);

  // Set padding to account for padding loss (backward), remove forward padding
  for (int_tp i = 0; i < pad_.size(); ++i) {
    add_def(ss, "v_p_" + std::to_string(i),
            (kernel_shape_[i] - 1) * dilation_[i] - pad_[i]);
  }

  for (int_tp i = 0; i < stride_.size(); ++i) {
    add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
  }

  for (int_tp i = 0; i < dilation_.size(); ++i) {
    add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
  }

  add_def(ss, "v_fin", fmaps_in_);
  add_def(ss, "v_fout", fmaps_out_);

  if (bias_term_) {
    add_def(ss, "v_bmul", bias_multiplier_);
  }

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

  // GEMM definitions
  add_def(ss, "MG", MG_BW_);
  add_def(ss, "M", M_BW_);
  add_def(ss, "N", N_BW_);
  add_def(ss, "KG", KG_BW_);
  add_def(ss, "K", K_BW_);

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  add_def(ss, "TSM", 64);
  // The tile-size in dimension N
  add_def(ss, "TSN", 64);
  // The tile-size in dimension K
  add_def(ss, "TSK", 16);
  // The work-per-thread in dimension M
  add_def(ss, "WPTM", 4);
  // The work-per-thread in dimension N
  add_def(ss, "WPTN", 4);
  // The reduced tile-size in dimension M
  add_def(ss, "RTSM", "(TSM/WPTM)");
  // The reduced tile-size in dimension N
  add_def(ss, "RTSN", "(TSN/WPTN)");
  // Loads-per-thread for A
  add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  return ss.str();
}


template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_wg_defs() {
  std::stringstream ss;

  // Number of spatial axes
  add_def(ss, "v_nax", num_axes_);

  // Groups
  add_def(ss, "v_g", group_);

  int_tp A_off = fmaps_out_;
  int_tp B_off = fmaps_in_;
  int_tp C_off = fmaps_in_ * fmaps_out_;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    A_off *= im_out_shape_[i];
    B_off *= im_in_shape_[i];
    C_off *= kernel_shape_[i];
  }
  // Output image batch offset
  add_def(ss, "v_A_off", A_off);
  // Input image batch offset
  add_def(ss, "v_B_off", B_off);
  // Weights offset
  add_def(ss, "v_C_off", C_off);

  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    add_def(ss, "v_imsi_" + std::to_string(i), im_in_shape_[i]);
    add_def(ss, "v_imso_" + std::to_string(i), im_out_shape_[i]);
  }

  int_tp v_ks = 1;
  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    add_def(ss, "v_k_" + std::to_string(i), kernel_shape_[i]);
    v_ks *= kernel_shape_[i];
  }
  add_def(ss, "v_ks", v_ks);

  // Set padding to account for padding loss (backward), remove forward padding
  for (int_tp i = 0; i < pad_.size(); ++i) {
    add_def(ss, "v_p_" + std::to_string(i), pad_[i]);
  }

  for (int_tp i = 0; i < stride_.size(); ++i) {
    add_def(ss, "v_s_" + std::to_string(i), stride_[i]);
  }

  for (int_tp i = 0; i < dilation_.size(); ++i) {
    add_def(ss, "v_d_" + std::to_string(i), dilation_[i]);
  }

  add_def(ss, "v_fin", fmaps_in_);
  add_def(ss, "v_fout", fmaps_out_);

  if (bias_term_) {
    add_def(ss, "v_bmul", bias_multiplier_);
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
  add_def(ss, "MG", MG_WG_);
  add_def(ss, "M", M_WG_);
  add_def(ss, "N", N_WG_);
  add_def(ss, "NG", NG_WG_);
  add_def(ss, "K", K_WG_);

  // Definitions as on http://www.cedricnugteren.nl/tutorial.php?page=8
  // The tile-size in dimension M
  add_def(ss, "TSM", 64);
  // The tile-size in dimension N
  add_def(ss, "TSN", 64);
  // The tile-size in dimension K
  add_def(ss, "TSK", 16);
  // The work-per-thread in dimension M
  add_def(ss, "WPTM", 4);
  // The work-per-thread in dimension N
  add_def(ss, "WPTN", 4);
  // The reduced tile-size in dimension M
  add_def(ss, "RTSM", "(TSM/WPTM)");
  // The reduced tile-size in dimension N
  add_def(ss, "RTSN", "(TSN/WPTN)");
  // Loads-per-thread for A
  add_def(ss, "LPTA", "((TSK*TSM)/(RTSM*RTSN))");
  // Loads-per-thread for B
  add_def(ss, "LPTB", "((TSK*TSN)/(RTSM*RTSN))");

  return ss.str();
}



template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;

  // Forward kernel
  ss << "__kernel void " + name + "(";
  ss << "__global const Dtype* im_in, ";
  ss << "__global const Dtype* wg, ";
  if (bias_term_) {
    ss << "__global const Dtype* bias, ";
  }
  ss << "__global Dtype* im_out";
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
  ss << "__local Dtype Asub[TSM][TSK];" << std::endl;
  ss << "__local Dtype Bsub[TSK][TSN];" << std::endl;

  // Register memory
  ss << "Dtype Areg;" << std::endl;
  ss << "Dtype Breg[WPTN];" << std::endl;
  ss << "Dtype Creg[WPTM][WPTN];" << std::endl;

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
       << "+ group * (v_B_off / v_g);"
       << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch + group * (M * N);"
       << std::endl;
    if (bias_term_) {
      ss << "__global const Dtype* Dptr = bias + group * (v_fout / v_g);";
    }
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_out + v_C_off * batch;" << std::endl;
    if (bias_term_) {
      ss << "__global const Dtype* Dptr = bias;";
    }
  }

  // Initialize the accumulation registers
  ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "Creg[wm][wn] = 0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over all tiles
  ss << "int_tp numTiles = ((K - 1)/TSK) + 1;" << std::endl;
  ss << "for (int_tp t = 0; t < numTiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int_tp tid = tidn * RTSM + tidm;" << std::endl;
  ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id % TSM;" << std::endl;
  ss << "int_tp col = id / TSM;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;

  // Load weights (wg) into Asub
  ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
  ss << "Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "Asub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Load one tile of B into local memory
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidn * RTSM + tidm;" << std::endl;
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
    ss << "Bsub[row][col] = 0;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;

  ss << "}" << std::endl;

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over the values of a single tile
  ss << "for (int_tp k=0; k<TSK; k++) {" << std::endl;

  // Cache the values of Bsub in registers
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "int_tp col = tidn + wn*RTSN;" << std::endl;
  ss << "Breg[wn] = Bsub[k][col];" << std::endl;
  ss << "}" << std::endl;

  // Perform the computation
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp row = tidm + wm*RTSM;" << std::endl;
  ss << "Areg = Asub[row][k];" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "Creg[wm][wn] += Areg * Breg[wn];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;

  // Store the final results in C
  ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm*RTSM;" << std::endl;
  if (bias_term_) {
    ss << "Dtype biasval = Dptr[globalRow];" << std::endl;
  }
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn*RTSN;" << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  if (bias_term_) {
    ss << "Cptr[globalRow * N + globalCol] = Creg[wm][wn] + v_bmul * biasval;"
        << std::endl;
  } else {
    ss << "Cptr[globalRow * N + globalCol] = Creg[wm][wn];" << std::endl;
  }
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}


template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_wg_kernels(std::string name) {
  std::stringstream ss;

  // Forward kernel
  ss << "__kernel void " + name + "(";
  ss << "__global const Dtype* im_in, ";
  ss << "__global const Dtype* im_out, ";
  if (bias_term_) {
    ss << "__global Dtype* bias, ";
  }
  ss << "__global Dtype* wg, ";
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
  ss << "__local Dtype Asub[TSM][TSK];" << std::endl;
  ss << "__local Dtype Bsub[TSK][TSN];" << std::endl;

  // Register memory
  ss << "Dtype Areg;" << std::endl;
  ss << "Dtype Breg[WPTN];" << std::endl;
  ss << "Dtype Creg[WPTM][WPTN];" << std::endl;

  if (bias_term_) {
    ss << "Dtype Dreg[WPTM];" << std::endl;
  }

  // Batch and group
  if (group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (group_ > 1) {
    ss << "__global const Dtype* Aptr = im_out + group * (M * K);"
        << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch "
        << "+ group * (v_B_off / v_g);"
        << std::endl;
    ss << "__global Dtype* Cptr = wg + v_C_off * batch + group * (M * N);"
        << std::endl;
    if (bias_term_) {
      ss << "__global Dtype* Dptr = bias + v_fout * batch "
          << "+ group * (v_fout / v_g);"
          << std::endl;
    }
  } else {
    ss << "__global const Dtype* Aptr = im_out;" << std::endl;
    ss << "__global const Dtype* Bptr = im_in + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = wg + v_C_off * batch;" << std::endl;
    if (bias_term_) {
      ss << "__global Dtype* Dptr = bias + v_fout * batch;"
          << std::endl;
    }
  }

  if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Initialize the accumulation registers
    // Load, add, store pattern
    ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
    ss << "int_tp globalRow = offM + tidm + wm*RTSM;" << std::endl;
    if (bias_term_) {
      ss << "Dreg[wm] = Dptr[globalRow];" << std::endl;
    }
    ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
    ss << "int_tp globalCol = offN + tidn + wn*RTSN;" << std::endl;
    ss << "if (globalRow < M && globalCol < N) {" << std::endl;
    ss << "Creg[wm][wn] = Cptr[globalRow * N + globalCol];" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  } else {
    // Zero initialize, atomic-add or reduce-add (intermediate buffer) pattern
    ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
    if (bias_term_) {
      ss << "Dreg[wm] = 0;" << std::endl;
    }
    ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
    ss << "Creg[wm][wn] = 0;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
    // Additional batch loop, keep the same accumulator for the weight gradient
    ss << "for (batch = 0; batch < batch_size; ++batch) {" << std::endl;
  }

  // Loop over all tiles
  ss << "int_tp numTiles = ((K - 1)/TSK) + 1;" << std::endl;

  ss << "for (int_tp t = 0; t < numTiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int_tp tid = tidn * RTSM + tidm;" << std::endl;
  ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id % TSM;" << std::endl;
  ss << "int_tp col = id / TSM;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;

  // Load weights (wg) into Asub
  ss << "if ((offM + row) < M && tiledIndex < K) {" << std::endl;
  ss << "Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "Asub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Load one tile of B into local memory
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidn * RTSM + tidm;" << std::endl;
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
    ss << "Bsub[row][col] = 0;" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;

  ss << "}" << std::endl;

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over the values of a single tile
  ss << "for (int_tp k=0; k<TSK; k++) {" << std::endl;

  // Cache the values of Bsub in registers
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "int_tp col = tidn + wn*RTSN;" << std::endl;
  ss << "Breg[wn] = Bsub[k][col];" << std::endl;
  ss << "}" << std::endl;

  // Perform the computation
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp row = tidm + wm*RTSM;" << std::endl;
  ss << "Areg = Asub[row][k];" << std::endl;
  if (bias_term_) {
    ss << "Dreg[wm] += Areg * v_bmul;" << std::endl;
  }
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "Creg[wm][wn] += Areg * Breg[wn];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;

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

  // Store the final results in C
  ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm*RTSM;" << std::endl;
  if (bias_term_) {
    ss << "if (tidn == 0 && offN == 0 && globalRow < M) {" << std::endl;
    ss << "Dptr[globalRow] = Dreg[wm];" << std::endl;
    ss << "}" << std::endl;
  }
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn*RTSN;" << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  ss << "Cptr[globalRow * N + globalCol] = Creg[wm][wn];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}


template<typename Dtype>
std::string libdnn_conv<Dtype>::generate_bw_kernels(std::string name) {
  std::stringstream ss;

  // Backward kernel
  ss << generate_bw_defs();

  ss << "__kernel void conv_backward(";
  ss << "__global const Dtype* im_out, ";
  ss << "__global Dtype* wg, ";
  if (bias_term_) {
    ss << "__global Dtype* bias, ";
  }
  ss << "__global Dtype* im_in";
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
  ss << "__local Dtype Asub[TSM][TSK];" << std::endl;
  ss << "__local Dtype Bsub[TSK][TSN];" << std::endl;

  // Register memory
  ss << "Dtype Areg;" << std::endl;
  ss << "Dtype Breg[WPTN];" << std::endl;
  ss << "Dtype Creg[WPTM][WPTN];" << std::endl;

  // Batch and group
  if (group_ > 1) {
    ss << "int_tp group = get_global_id(2) % v_g;" << std::endl;
    ss << "int_tp batch = get_global_id(2) / v_g;" << std::endl;
  } else {
    ss << "int_tp batch = get_global_id(2);" << std::endl;
  }

  if (group_ > 1) {
    ss << "__global const Dtype* Aptr = wg + group * (M * K);" << std::endl;
    ss
        << "__global const Dtype* Bptr = im_out + v_B_off * batch "
        << "+ group * (v_B_off / v_g);"
        << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch + group * (M * N);"
        << std::endl;
  } else {
    ss << "__global const Dtype* Aptr = wg;" << std::endl;
    ss << "__global const Dtype* Bptr = im_out + v_B_off * batch;" << std::endl;
    ss << "__global Dtype* Cptr = im_in + v_C_off * batch;" << std::endl;
  }

  // Initialize the accumulation registers
  ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "Creg[wm][wn] = 0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over all tiles
  ss << "int_tp numTiles = ((K - 1)/TSK) + 1;" << std::endl;
  ss << "for (int_tp t = 0; t < numTiles; ++t) {" << std::endl;

  // Load one tile of A into local memory
  ss << "for (int_tp la = 0; la < LPTA; ++la) {" << std::endl;
  ss << "int_tp tid = tidn * RTSM + tidm;" << std::endl;
  ss << "int_tp id = la * RTSN * RTSM + tid;" << std::endl;
  ss << "int_tp row = id % TSM;" << std::endl;
  ss << "int_tp col = id / TSM;" << std::endl;
  ss << "int_tp tiledIndex = TSK * t + col;" << std::endl;

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
  ss << "} else {" << std::endl;
  ss << "Asub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Load one tile of B into local memory
  ss << "for (int_tp lb = 0; lb < LPTB; ++lb) {" << std::endl;
  ss << "int_tp tid = tidn * RTSM + tidm;" << std::endl;
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
    ss << "d_temp_" << i << " = (imageIndex % v_imsi_" << i << ")" << " - v_p_"
        << i << ";" << std::endl;
    ss << "imageIndex = imageIndex / v_imsi_" << i << ";" << std::endl;
  }

  ss << "int_tp d_iter_im;" << std::endl;
  for (int_tp i = 0; i < num_axes_; ++i) {
    // Here, d_temp_ represents the column shift,
    // while d_iter_ is the kernel shift
    ss << "d_iter_im = d_temp_" << i << " + d_iter_" << i << ";" << std::endl;
    ss << "tiledIndex = tiledIndex * v_imso_" << i << " + d_iter_im / v_s_" << i
        << ";" << std::endl;
    // In range: Not before or after actual image data
    // and not between image strides
    ss << "in_range &= d_iter_im >= 0 && d_iter_im < v_imso_" << i << " * v_s_"
        << i << " && d_iter_im % v_s_" << i << " == 0;" << std::endl;
  }

  ss << "if (in_range) {" << std::endl;
  // tiledIndex now holds the memory offset for the input image
  ss << "Bsub[row][col] = Bptr[tiledIndex];" << std::endl;
  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;

  ss << "} else {" << std::endl;
  ss << "Bsub[row][col] = 0;" << std::endl;
  ss << "}" << std::endl;

  ss << "}" << std::endl;

  // Synchronize to make sure the tile is loaded
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over the values of a single tile
  ss << "for (int_tp k=0; k<TSK; k++) {" << std::endl;

  // Cache the values of Bsub in registers
  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "int_tp col = tidn + wn*RTSN;" << std::endl;
  ss << "Breg[wn] = Bsub[k][col];" << std::endl;
  ss << "}" << std::endl;

  // Perform the computation
  ss << "for (int_tp wm=0; wm<WPTM; ++wm) {" << std::endl;
  ss << "int_tp row = tidm + wm*RTSM;" << std::endl;
  ss << "Areg = Asub[row][k];" << std::endl;
  ss << "for (int_tp wn=0; wn<WPTN; ++wn) {" << std::endl;
  ss << "Creg[wm][wn] += Areg * Breg[wn];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Loop over a single tile
  ss << "}" << std::endl;

  // Synchronize before loading the next tile
  ss << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  // Loop over all tiles
  ss << "}" << std::endl;

  // Store the final results in C
  ss << "for (int_tp wm=0; wm<WPTM; wm++) {" << std::endl;
  ss << "int_tp globalRow = offM + tidm + wm*RTSM;" << std::endl;

  ss << "for (int_tp wn=0; wn<WPTN; wn++) {" << std::endl;
  ss << "int_tp globalCol = offN + tidn + wn*RTSN;" << std::endl;
  ss << "if (globalRow < M && globalCol < N) {" << std::endl;
  ss << "Cptr[globalRow * N + globalCol] = Creg[wm][wn];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  // Kernel
  ss << "}" << std::endl;

  return ss.str();
}

template<typename Dtype>
void libdnn_conv<Dtype>::generate_kernels() {
  std::stringstream ss;

  ss << generate_header();
  ss << generate_fw_defs();
  ss << generate_fw_kernels("conv_forward");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("conv_backward");
  ss << generate_wg_defs();
  ss << generate_wg_kernels("conv_weights");

  // Write complete kernel string
  kernel_ = ss.str();
}

template<typename Dtype>
viennacl::ocl::program libdnn_conv<Dtype>::compile_kernels(
    viennacl::ocl::context *ctx) {

  std::string build_opts = "";

  if (fast_unsafe_math_) {
    build_opts += "-cl-fast-relaxed-math -cl-mad-enable ";
  }

  ctx->build_options(build_opts);

  // std::cout << kernel_ << std::endl;

  program_ = ctx->add_program(kernel_.c_str(), "kernel_program");
  return program_;
}

template<typename Dtype>
void libdnn_conv<Dtype>::forward(cl_mem bottom_data, cl_mem weight, cl_mem bias,
                                 cl_mem top_data, int_tp batch_size) {
  viennacl::ocl::kernel &kernel = program_.get_kernel("conv_forward");

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_ptr_->id());

  kernel.local_work_size(0, 16);
  kernel.local_work_size(1, 16);
  kernel.local_work_size(2, 1);

  kernel.global_work_size(0, ((this->N_FW_ - 1) / 64 + 1) * 16);
  kernel.global_work_size(1, ((this->M_FW_ - 1) / 64 + 1) * 16);
  kernel.global_work_size(2, batch_size * group_);

  // for (int i = 0; i < 3; ++i) {
  // std::cout << i << "; local: "
  //           << kernel.local_work_size(i) << ", global: "
  //           << kernel.global_work_size(i) << std::endl;
  // }

  if (bias_term_) {
    viennacl::ocl::enqueue(
        kernel(WrapHandle(bottom_data, &ctx), WrapHandle(weight, &ctx),
               WrapHandle(bias, &ctx), WrapHandle(top_data, &ctx)),
        ctx.get_queue());
  } else {
    viennacl::ocl::enqueue(
        kernel(WrapHandle(bottom_data, &ctx), WrapHandle(weight, &ctx),
               WrapHandle(top_data, &ctx)),
        ctx.get_queue());
  }
}

template<typename Dtype>
void libdnn_conv<Dtype>::backward(bool prop_down_data,
                                  cl_mem top_data, cl_mem top_diff,
                                  cl_mem weight, cl_mem weight_diff,
                                  cl_mem bias, cl_mem bias_diff,
                                  cl_mem bottom_data, cl_mem bottom_diff,
                                  int_tp batch_size) {
  // Backprop w.r.t. data
  if (prop_down_data) {
    viennacl::ocl::kernel &kernel = program_.get_kernel("conv_backward");

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_ptr_->id());

    kernel.local_work_size(0, 16);
    kernel.local_work_size(1, 16);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((this->N_BW_ - 1) / 64 + 1) * 16);
    kernel.global_work_size(1, ((this->M_BW_ - 1) / 64 + 1) * 16);
    kernel.global_work_size(2, batch_size * group_);

    // for (int i = 0; i < 3; ++i) {
    // std::cout << i << "; local: "
    //           << kernel.local_work_size(i) << ", global: "
    //           << kernel.global_work_size(i) << std::endl;
    // }

    if (bias_term_) {
      viennacl::ocl::enqueue(
          kernel(WrapHandle(top_diff, &ctx), WrapHandle(weight, &ctx),
                 WrapHandle(bias, &ctx), WrapHandle(bottom_diff, &ctx)),
          ctx.get_queue());
    } else {
      viennacl::ocl::enqueue(
          kernel(WrapHandle(top_diff, &ctx), WrapHandle(weight, &ctx),
                 WrapHandle(bottom_diff, &ctx)),
          ctx.get_queue());
    }
  }

  // Backprop w.r.t. weights and bias
  if (this->weights_backward_ || this->bias_backward_) {
    viennacl::ocl::kernel &kernel = program_.get_kernel("conv_weights");

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_ptr_->id());

    kernel.local_work_size(0, 16);
    kernel.local_work_size(1, 16);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((this->N_WG_ - 1) / 64 + 1) * 16);
    kernel.global_work_size(1, ((this->M_WG_ - 1) / 64 + 1) * 16);

    if (wgalgo_ == LIBDNN_CONVOLUTION_WG_ALGO_DIRECT) {
      kernel.global_work_size(2, group_);
    } else {
      kernel.global_work_size(2, batch_size * group_);
    }

    // for (int i = 0; i < 3; ++i) {
    // std::cout << i << "; local: "
    //           << kernel.local_work_size(i) << ", global: "
    //           << kernel.global_work_size(i) << std::endl;
    // }

    if (bias_term_) {
      viennacl::ocl::enqueue(
          kernel(WrapHandle(bottom_data, &ctx), WrapHandle(top_diff, &ctx),
                 WrapHandle(bias_diff, &ctx), WrapHandle(weight_diff, &ctx),
                 batch_size),
          ctx.get_queue());
    } else {
      viennacl::ocl::enqueue(
          kernel(WrapHandle(bottom_data, &ctx), WrapHandle(top_diff, &ctx),
                 WrapHandle(weight_diff, &ctx), batch_size),
          ctx.get_queue());
    }
  }
}

INSTANTIATE_CLASS(libdnn_conv);

}  // namespace caffe

#endif  // USE_GREENTEA
