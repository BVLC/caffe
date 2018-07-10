#ifdef USE_LIBDNN

#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/libdnn/libdnn_pool.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
LibDNNPool<MItype, MOtype>::LibDNNPool(LibDNNPoolConfig config)
        : LibDNN<MItype, MOtype>(config.dev_ptr)  {
  config_ = config;
  this->fast_unsafe_math_ = config.fast_unsafe_math;
  int_tp dims = config.in_shape.size();
  int_tp spatial_dims = config.kernel.size();

  num_axes_ = spatial_dims;

  pool_method_ = config.pool_method;
  bwalgo_ = config.bwalgo;
  use_top_mask_ = config.use_top_mask;

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

  fw_tuner_ = shared_ptr<LibDNNTuner>(new LibDNNTuner());
  bw_tuner_ = shared_ptr<LibDNNTuner>(new LibDNNTuner());

  fw_tuner_->add_range_param<int_tp>("LW0", 8, 4, 16, 4);
  bw_tuner_->add_range_param<int_tp>("LW0", 8, 4, 16, 4);
  fw_tuner_->add_range_param<int_tp>("LW1", 8, 4, 16, 4);
  bw_tuner_->add_range_param<int_tp>("LW1", 8, 4, 16, 4);

  this->GenerateKernels();
  this->CompileKernels();
}

template<typename MItype, typename MOtype>
const LibDNNPoolConfig LibDNNPool<MItype, MOtype>::get_config() {
  return config_;
}


template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::string_identifier() {
  stringstream ss;
  ss << "POOL_";
  // Type names
  ss << safe_type_name<MItype>() << "_";
  ss << safe_type_name<MItype>() << "_";
  ss << safe_type_name<MOtype>() << "_";
  switch (pool_method_) {
    case LIBDNN_POOLING_METHOD_MAX:
      ss << "MAX_";
      break;
    case LIBDNN_POOLING_METHOD_AVE:
      ss << "AVE_";
      break;
    case LIBDNN_POOLING_METHOD_STO:
      ss << "STO_";
      break;
  }
  // Device name
  ss << this->dev_ptr_->name();
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
  ss << "]";
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::generate_fw_defs() {
  stringstream ss;

  // Number of spatial axes
  ss << this->program_->define("v_nax", num_axes_);

  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    ss << this->program_->define("v_k_" + std::to_string(i), kernel_shape_[i]);
  }
  for (int_tp i = 0; i < pad_.size(); ++i) {
    ss << this->program_->define("v_p_" + std::to_string(i), pad_[i]);
  }
  for (int_tp i = 0; i < stride_.size(); ++i) {
    ss << this->program_->define("v_s_" + std::to_string(i), stride_[i]);
  }
  for (int_tp i = 0; i < dilation_.size(); ++i) {
    ss << this->program_->define("v_d_" + std::to_string(i), dilation_[i]);
  }

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    ss << this->program_->define("v_imsi_" + std::to_string(i),
                                 im_in_shape_[i]);
    imsi *= im_in_shape_[i];
    ss << this->program_->define("v_imso_" + std::to_string(i),
                                 im_out_shape_[i]);
    imso *= im_out_shape_[i];
  }
  ss << this->program_->define("v_imsi", imsi);
  ss << this->program_->define("v_imso", imso);

  return ss.str();
}


template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::generate_bw_defs() {
  stringstream ss;

  // Number of spatial axes
  ss << this->program_->define("v_nax", num_axes_);
  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    ss << this->program_->define("v_k_" + std::to_string(i), kernel_shape_[i]);
  }
  for (int_tp i = 0; i < pad_.size(); ++i) {
    ss << this->program_->define("v_p_" + std::to_string(i), pad_[i]);
  }
  for (int_tp i = 0; i < stride_.size(); ++i) {
    ss << this->program_->define("v_s_" + std::to_string(i), stride_[i]);
  }
  for (int_tp i = 0; i < dilation_.size(); ++i) {
    ss << this->program_->define("v_d_" + std::to_string(i), dilation_[i]);
  }

  int_tp imsi = 1;
  int_tp imso = 1;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    ss << this->program_->define("v_imsi_" + std::to_string(i),
                                 im_in_shape_[i]);
    imsi *= im_in_shape_[i];
    ss << this->program_->define("v_imso_" + std::to_string(i),
                                 im_out_shape_[i]);
    imso *= im_out_shape_[i];
  }
  ss << this->program_->define("v_imsi", imsi);
  ss << this->program_->define("v_imso", imso);

  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::generate_fw_kernels(string name,
                                                       bool test_mode) {
  stringstream ss;
#ifdef USE_HALF
  if (std::is_same<MItype, half_fp>::value) {
    ss << "#define DTYPE_MAX HALF_MAX" << std::endl;
    ss << "#define DTYPE_MIN HALF_MIN" << std::endl;
  } else if (std::is_same<MItype, float>::value
        || std::is_same<MItype, double>::value) {
#endif
    ss << "#define DTYPE_MAX FLT_MAX" << std::endl;
    ss << "#define DTYPE_MIN FLT_MIN" << std::endl;
#ifdef USE_HALF
  } else {
    ss << "#define DTYPE_MAX " << 0 << std::endl;
    ss << "#define DTYPE_MIN " << 0 << std::endl;
  }
#endif

  KernelArgs args;
  args.push_back(this->program_->template create_kernel_arg<MItype>(
      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
      | KERNEL_ARG_RESTRICT));
  args.push_back(this->program_->template create_kernel_arg<MOtype>("top_data",
                                  KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    if (use_top_mask_) {
      args.push_back(this->program_->template create_kernel_arg<MItype>(
                      "top_mask", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    } else {
      args.push_back(this->program_->template create_kernel_arg<int_tp>(
                          "mask", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    }
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_STO && !test_mode) {
    args.push_back(this->program_->template create_kernel_arg<MItype>(
             "rand_idx", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  }
  args.push_back(this->program_->template create_kernel_arg<int_tp>(
                                                 "channels", KERNEL_ARG_CONST));
  args.push_back(this->program_->template create_kernel_arg<int_tp>(
                                               "batch_size", KERNEL_ARG_CONST));
  ss << this->program_->function(name, args);

  ss << "int_tp out_idx = " << this->program_->global_id(0) << ";" << std::endl;
  ss << "if (" << this->program_->global_id(1)
     << " >= channels * batch_size) {return;}" << std::endl;
  ss << "int_tp idx_0 = " << this->program_->global_id(0) << ";" << std::endl;
  for (int_tp i = num_axes_ - 1; i >= 1; --i) {
    ss << "int_tp idx_" << i << " = (idx_0 % v_imso_" << i << ");" << std::endl;
    ss << "idx_" << i << " = idx_" << i
       << " * v_s_" << i << " - v_p_" << i << ";" << std::endl;
    ss << "idx_0 /= v_imso_" << i << ";" << std::endl;
  }
  ss << "if (idx_0 >= v_imso_0) {return;}" << std::endl;
  ss << "idx_0 = idx_0 * v_s_0 - v_p_0;" << std::endl;
  ss << "int_tp in_idx = idx_0;" << std::endl;
  for (int_tp i = 1; i < num_axes_; ++i) {
    ss << "in_idx = in_idx * v_imsi_" << i
       << " + " << "idx_" << i << ";" << std::endl;
  }
  ss << this->program_->global_ptr("const MItype", "in_ptr")
     << " = bottom_data + " << this->program_->global_id(1)
     << " * v_imsi + in_idx;" << std::endl;
  ss << this->program_->global_ptr("MOtype", "out_ptr") << " = top_data + "
     << this->program_->global_id(1) << " * v_imso;" << std::endl;

  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    if (use_top_mask_) {
      ss << this->program_->global_ptr("MOtype", "mask_ptr") << " = top_mask + "
         << this->program_->global_id(1) << " * v_imso;" << std::endl;
    } else {
      ss << this->program_->global_ptr("int_tp", "mask_ptr") << " = mask + "
         << this->program_->global_id(1) << " * v_imso;"  << std::endl;
    }
    ss << "MItype val = -DTYPE_MAX;" << std::endl;
    ss << "int_tp maxidx = -1;" << std::endl;
  }

  if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
    ss << "MItype val = 0;" << std::endl;
  }

  if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
    if (test_mode) {
      ss << "MItype cumsum = DTYPE_MIN;" << std::endl;
      ss << "MItype cumvalues = 0;" << std::endl;
    } else {
      ss << this->program_->global_ptr("MItype", "rand_ptr") << " = rand_idx + "
         << this->program_->global_id(1) << " * v_imso;" << std::endl;
      ss << "MItype val = 0;" << std::endl;
      ss << "MItype cumsum = 0;" << std::endl;
      ss << "int_tp stoidx = -1;" << std::endl;
    }
  }

  vector<int_tp> d_iter;

  for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
    d_iter.push_back(0);
  }

  if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
    int_tp ave = std::accumulate(kernel_shape_.begin(),
                                 kernel_shape_.end(),
                                  1, std::multiplies<int_tp>());
    ss << "int_tp ave = " << ave << ";" << std::endl;
  }

  for (int_tp sto_idx = 0;
       sto_idx < ((pool_method_ == LIBDNN_POOLING_METHOD_STO && !test_mode)
       ? 2 : 1); ++sto_idx) {
    if (pool_method_ == LIBDNN_POOLING_METHOD_STO && sto_idx == 1) {
      ss << "MItype thres = rand_ptr[out_idx] * cumsum;" << std::endl;
      ss << "cumsum = 0;" << std::endl;
    }
    // Loop over the kernel
    bool incremented;
    do {
      int_tp kernel_offset = 0;
      int_tp size_prod = 1;
      for (int_tp i = num_axes_ - 1; i >= 0; --i) {
        kernel_offset += size_prod * d_iter[i] * dilation_[i];
        size_prod *= im_in_shape_[i];
      }

      bool max_guard = false;
      bool pad_guard = false;
      bool overspill_guard = false;
      for (int_tp i = 0; i < num_axes_; ++i) {
        if (d_iter[i] * dilation_[i] < pad_[i]) {
          pad_guard = true;
        }
        if (d_iter[i] * dilation_[i] >=
            ((kernel_shape_[i] - 1) * dilation_[i] + 1) - pad_[i] ||
            (im_out_shape_[i] - 1) * stride_[i] + d_iter[i]
                         * dilation_[i] - pad_[i] >= im_in_shape_[i] ) {
          pad_guard = true;
        }
        if ((im_out_shape_[i] - 1) * stride_[i] + d_iter[i]
             * dilation_[i] - pad_[i] >= im_in_shape_[i]) {
          overspill_guard = true;
        }
      }
      if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
        max_guard = true;
      }

      if (max_guard || pad_guard || overspill_guard) {
        ss << "if (";
      }
      if (pad_guard || overspill_guard) {
        for (int_tp i = 0; i < num_axes_; ++i) {
          if (d_iter[i] * dilation_[i] < pad_[i]) {
            ss << "idx_" << i << " >= -" << (d_iter[i] * dilation_[i])
               << " && ";
          }
          if ((d_iter[i] * dilation_[i] >= ((kernel_shape_[i] - 1)
              * dilation_[i] + 1) - pad_[i]) ||
              ((im_out_shape_[i] - 1) * stride_[i]
              + d_iter[i] * dilation_[i] - pad_[i]
              >= im_in_shape_[i])) {
            ss << "idx_" << i << " < v_imsi_" << i << " - "
               << (d_iter[i] * dilation_[i]) << " && ";
          }
        }
      }
      if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
        if (max_guard  || pad_guard || overspill_guard) {
          ss << "in_ptr[" << kernel_offset << "] > val) {" << std::endl;
        }
        ss << "maxidx = in_idx + " << kernel_offset << ";" << std::endl;
        ss << "val = in_ptr[" << kernel_offset << "];" << std::endl;
        if (max_guard  || pad_guard || overspill_guard) {
          ss << "}" << std::endl;
        }
      }
      if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
        if (pad_guard || overspill_guard) {
          ss << "true) {" << std::endl;
        }
        ss << "val += in_ptr[" << kernel_offset << "];" << std::endl;
        if (pad_guard || overspill_guard) {
          ss << "}" << std::endl;
        }
        if (overspill_guard) {
          ss << "if (";
          for (int_tp i = 0; i < num_axes_; ++i) {
            if ((im_out_shape_[i] - 1) * stride_[i]
                + d_iter[i] * dilation_[i] - pad_[i]
                >= im_in_shape_[i]) {
              ss << "idx_" << i << " + " << d_iter[i] * dilation_[i]
                 << " >= v_imsi_" << i << " + "
                 << pad_[i] << " || ";
            }
          }
          ss << "false) {--ave;}" << std::endl;
        }
      }
      if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
        if (pad_guard || overspill_guard) {
          ss << "true) {" << std::endl;
        }
        ss << "cumsum += in_ptr[" << kernel_offset << "];" << std::endl;
        if (test_mode) {
          ss << "cumvalues += in_ptr[" << kernel_offset << "]"
             << " * in_ptr[" << kernel_offset << "];" << std::endl;
        } else {
          if (sto_idx == 1) {
            // Second pass
            ss << "if (cumsum > thres) {" << std::endl;
            ss << "stoidx = in_idx + " << kernel_offset << ";" << std::endl;
            ss << "val = in_ptr[" << kernel_offset << "];" << std::endl;
            ss << "thres = DTYPE_MAX;" << std::endl;
            ss << "}" << std::endl;
          }
        }
        if (pad_guard || overspill_guard) {
          ss << "}" << std::endl;
        }
      }

      incremented = false;
      for (int_tp i = num_axes_ - 1; i >= 0; --i) {
        if (d_iter[i] >= kernel_shape_[i] - 1) {
          d_iter[i] = 0;
        } else {
          d_iter[i] += 1;
          incremented = true;
          break;
        }
      }
    } while (incremented);
  }

  // Write out the pooling result
  if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
    ss << "out_ptr[out_idx] = val / ((MItype)ave);" << std::endl;
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    ss << "out_ptr[out_idx] = val;" << std::endl;
    if (use_top_mask_) {
      ss << "mask_ptr[out_idx] = (MItype)maxidx;" << std::endl;
    } else {
      ss << "mask_ptr[out_idx] = maxidx;" << std::endl;
    }
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
    if (test_mode) {
      ss << "out_ptr[out_idx] = cumvalues / cumsum;" << std::endl;
    } else {
      ss << "out_ptr[out_idx] = val;" << std::endl;
      ss << "rand_ptr[out_idx] = (MItype)stoidx;" << std::endl;
    }
  }

  ss << "}" << std::endl;  // Kernel
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::generate_fwtr_kernels(string name) {
  stringstream ss;
  ss << generate_fw_kernels(name, false);
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::generate_fwte_kernels(string name) {
  stringstream ss;
  ss << generate_fw_kernels(name, true);
  return ss.str();
}

template<typename MItype, typename MOtype>
string LibDNNPool<MItype, MOtype>::generate_bw_kernels(string name) {
  stringstream ss;

  KernelArgs args;
  args.push_back(this->program_->template create_kernel_arg<MOtype>("top_diff",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  args.push_back(this->program_->template create_kernel_arg<MItype>(
      "bottom_diff", KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    if (use_top_mask_) {
      args.push_back(this->program_->template create_kernel_arg<MOtype>(
          "top_mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
          | KERNEL_ARG_RESTRICT));
    } else {
      args.push_back(this->program_->template create_kernel_arg<int_tp>("mask",
               KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM | KERNEL_ARG_RESTRICT));
    }
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
    args.push_back(this->program_->template create_kernel_arg<MItype>(
        "rand_idx", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM
        | KERNEL_ARG_RESTRICT));
  }
  args.push_back(this->program_->template create_kernel_arg<int_tp>("channels",
           KERNEL_ARG_CONST));
  args.push_back(this->program_->template create_kernel_arg<int_tp>(
           "batch_size", KERNEL_ARG_CONST));
  ss << this->program_->function(name, args);

  if (bwalgo_ == LIBDNN_POOLING_BW_ALGO_ATOMIC) {
    // Atomic kernel
    ss << "int_tp in_idx = " << this->program_->global_id(0) << ";"
       << std::endl;
    ss << "if (" << this->program_->global_id(1)
       << " >= channels * batch_size) {return;}" << std::endl;
    ss << "int_tp idx_0 = " << this->program_->global_id(0) << ";" << std::endl;
    for (int_tp i = num_axes_ - 1; i >= 1; --i) {
      ss << "int_tp idx_" << i << " = (idx_0 % v_imso_" << i << ");"
         << std::endl;
      ss << "idx_" << i << " = idx_" << i << " * v_s_"
         << i << " - v_p_" << i << ";" << std::endl;
      ss << "idx_0 /= v_imso_" << i << ";" << std::endl;
    }
    ss << "if (idx_0 >= v_imso_0) {return;}" << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      ss << "idx_0 = idx_0 * v_s_0 - v_p_0;" << std::endl;
      ss << "int_tp out_idx = idx_0;" << std::endl;
      for (int_tp i = 1; i < num_axes_; ++i) {
        ss << "out_idx = out_idx * v_imsi_" << i
           << " + " << "idx_" << i << ";" << std::endl;
      }
      ss << this->program_->global_ptr("MItype", "out_ptr") << " = bottom_diff "
         << "+ " << this->program_->global_id(1) << " * v_imsi + out_idx;"
         << std::endl;
    } else {
      ss << this->program_->global_ptr("MItype", "out_ptr") << " = bottom_diff "
         << "+ " << this->program_->global_id(1) << " * v_imsi;" << std::endl;
    }
    ss << this->program_->global_ptr("const MOtype", "in_ptr") << " = top_diff "
       << "+ " << this->program_->global_id(1) << " * v_imso + in_idx;"
       << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
      if (use_top_mask_) {
        ss << this->program_->global_ptr("const MOtype", "mask_ptr")
           << "= top_mask + " << this->program_->global_id(1)
           << " * v_imso + in_idx;" << std::endl;
      } else {
        ss << this->program_->global_ptr("const int_tp", "mask_ptr")
           << " = mask + " << this->program_->global_id(1)
           << " * v_imso + in_idx;" << std::endl;
      }
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << this->program_->global_ptr("const MItype", "rand_ptr")
         << " = rand_idx + " << this->program_->global_id(1)
         << " * v_imso + in_idx;" << std::endl;
    }

    vector<int_tp> d_iter;

    for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
      d_iter.push_back(0);
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      int_tp ave = std::accumulate(kernel_shape_.begin(),
                                   kernel_shape_.end(),
                                    1, std::multiplies<int_tp>());
      ss << "int_tp ave = " << ave << ";" << std::endl;
      ss << "MItype val = in_ptr[0];" << std::endl;
    }

    for (int_tp ave_idx = 0;
         ave_idx < ((pool_method_ == LIBDNN_POOLING_METHOD_AVE)
         ? 2 : 0); ++ave_idx) {
      if (ave_idx == 1) {
        ss << "val /= ((MItype)ave);" << std::endl;
      }
      // Loop over the kernel
      bool incremented;
      do {
        int_tp kernel_offset = 0;
        int_tp size_prod = 1;
        for (int_tp i = num_axes_ - 1; i >= 0; --i) {
          kernel_offset += size_prod * d_iter[i] * dilation_[i];
          size_prod *= im_in_shape_[i];
        }

        bool pad_guard = false;
        bool overspill_guard = false;
        for (int_tp i = 0; i < num_axes_; ++i) {
          if (d_iter[i] * dilation_[i] < pad_[i]) {
            pad_guard = true;
          }
          if (d_iter[i] * dilation_[i] >=
              ((kernel_shape_[i] - 1) * dilation_[i] + 1) - pad_[i] ||
              (im_out_shape_[i] - 1) * stride_[i] + d_iter[i]
                           * dilation_[i] - pad_[i] >= im_in_shape_[i] ) {
            pad_guard = true;
          }
          if ((im_out_shape_[i] - 1) * stride_[i] + d_iter[i]
               * dilation_[i] - pad_[i] >= im_in_shape_[i]) {
            overspill_guard = true;
          }
        }

        if ((ave_idx == 1) && (pad_guard || overspill_guard)) {
          ss << "if (";
        }
        if ((ave_idx == 1) && (pad_guard || overspill_guard)) {
          for (int_tp i = 0; i < num_axes_; ++i) {
            if (d_iter[i] * dilation_[i] < pad_[i]) {
              ss << "idx_" << i << " >= -" << (d_iter[i] * dilation_[i])
                 << " && ";
            }
            if ((d_iter[i] * dilation_[i] >= ((kernel_shape_[i] - 1)
                * dilation_[i] + 1) - pad_[i]) ||
                ((im_out_shape_[i] - 1) * stride_[i]
                + d_iter[i] * dilation_[i] - pad_[i]
                >= im_in_shape_[i])) {
              ss << "idx_" << i << " < v_imsi_" << i << " - "
                 << (d_iter[i] * dilation_[i]) << " && ";
            }
          }
        }
        if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
          if ((ave_idx == 1) && (pad_guard || overspill_guard)) {
            ss << "true) {" << std::endl;
          }
          if (ave_idx == 1) {
            ss << this->program_->template atomic_add<MItype>("(&out_ptr["
               + std::to_string(kernel_offset) + "])", "val") << std::endl;
          }
          if ((ave_idx == 1) && (pad_guard || overspill_guard)) {
            ss << "}" << std::endl;
          }
          if (overspill_guard && ave_idx == 0) {
            ss << "if (";
            for (int_tp i = 0; i < num_axes_; ++i) {
              if ((im_out_shape_[i] - 1) * stride_[i]
                  + d_iter[i] * dilation_[i] - pad_[i]
                  >= im_in_shape_[i]) {
                ss << "idx_" << i << " + " << d_iter[i] * dilation_[i]
                   << " >= v_imsi_" << i << " + "
                   << pad_[i] << " || ";
              }
            }
            ss << "false) {--ave;}" << std::endl;
          }
        }

        incremented = false;
        for (int_tp i = num_axes_ - 1; i >= 0; --i) {
          if (d_iter[i] >= kernel_shape_[i] - 1) {
            d_iter[i] = 0;
          } else {
            d_iter[i] += 1;
            incremented = true;
            break;
          }
        }
      } while (incremented);
    }
    if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
      ss << "if (mask_ptr[0] >= 0 && mask_ptr[0] < v_imsi) {" << std::endl;
      ss << this->program_->template atomic_add<MItype>(
          "&out_ptr[(int_tp)(mask_ptr[0])]",
          "in_ptr[0]") << std::endl;
      ss << "}" << std::endl;
    }
    if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << "if (mask_ptr[0] >= 0 && mask_ptr[0] < v_imsi) {" << std::endl;
      ss << this->program_->template atomic_add<MItype>(
          "&out_ptr[(int_tp)(rand_ptr[0])]",
          "in_ptr[0]") << std::endl;
      ss << "}" << std::endl;
    }

  } else {
    // Direct, deterministic kernel
    ss << "int_tp d_start[" << num_axes_ << "];" << std::endl;
    ss << "int_tp d_end[" << num_axes_ << "];" << std::endl;
    ss << "int_tp d_iter[" << num_axes_ << "];" << std::endl;

    ss << "int_tp out_idx = " << this->program_->global_id(0)
       << ";" << std::endl;
    ss << "int_tp idx_0 = " << this->program_->global_id(0)
       << ";" << std::endl;
    ss << "if (" << this->program_->global_id(1)
       << " >= channels * batch_size) {return;}" << std::endl;

    for (int_tp i = num_axes_ - 1; i >= 1; --i) {
      ss << "int_tp idx_" << i << " = (idx_0 % v_imsi_" << i << ");"
         << std::endl;
      ss << "idx_0 /= v_imsi_" << i << ";" << std::endl;
    }
    ss << "if (idx_0 >= v_imsi_0) {return;}" << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      ss << this->program_->global_ptr("MItype", "out_ptr") << " = bottom_diff "
         << "+ " << this->program_->global_id(1) << " * v_imsi + out_idx;"
         << std::endl;
    } else {
      ss << this->program_->global_ptr("MItype", "out_ptr") << " = bottom_diff "
         << "+ " << this->program_->global_id(1) << " * v_imsi + out_idx;"
         << std::endl;
    }
    ss << this->program_->global_ptr("const MOtype", "in_ptr") << " = top_diff "
       << "+ " << this->program_->global_id(1) << " * v_imso;" << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
      if (use_top_mask_) {
        ss << this->program_->global_ptr("const MOtype", "mask_ptr")
           << " = top_mask + " << this->program_->global_id(1) << " * v_imso;"
           << std::endl;
      } else {
        ss << this->program_->global_ptr("const int_tp", "mask_ptr")
           << " = mask + " << this->program_->global_id(1) << " * v_imso;"
           << std::endl;
      }
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << this->program_->global_ptr("const MItype", "rand_ptr")
         << " = rand_idx + " << this->program_->global_id(1) << " * v_imso;"
         << std::endl;
    }

    for (int_tp i = 0; i < num_axes_; ++i) {
      ss << "d_start[" << i << "] = (idx_" << i << " + v_p_" << i << " < "
         << "((v_k_" << i << " - 1) * v_d_" << i << " + 1)) ? 0 : (idx_" << i
         << " + v_p_" << i
         << " - ((v_k_" << i << " - 1) * v_d_" << i << " + 1))"
         << " / v_s_" << i << " + 1;" << std::endl;
      ss << "d_end[" << i << "] = min(v_imso_" << i << " - 1, "
         << "(idx_" << i << " + v_p_" << i << ")"
         << " / v_s_" << i << ");" << std::endl;
      ss << "d_iter[" << i << "] = d_start[" << i << "];" << std::endl;
      ss << "if (d_start[" << i << "] > d_end[" << i << "]) {" << std::endl;
      ss << "out_ptr[0] = 0;" << std::endl;
      ss << "return;" << std::endl;
      ss << "}" << std::endl;
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      ss << "int_tp av_start[" << num_axes_ << "];" << std::endl;
      ss << "int_tp av_end[" << num_axes_ << "];" << std::endl;
    }
    // ss << "printf(\"%f\\N\", (float)ave);" << std::endl;
    ss << "MItype gradient = 0.0;" << std::endl;
    ss << "bool incremented;" << std::endl;
    ss << "do {" << std::endl;
    ss << "int_tp offset = 0;" << std::endl;
    for (int_tp i = 0; i < num_axes_; ++i) {
      ss << "offset += d_iter[" << i << "];" << std::endl;
      if (i < num_axes_ - 1) {
        ss << "offset *= v_imso_" << (i + 1) << ";" << std::endl;
      }
    }
    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      ss << "int_tp ave = 1;" << std::endl;
      for (int_tp i = 0; i < num_axes_; ++i) {
        ss << "av_start[" << i << "] = d_iter[" << i << "] * v_s_" << i
        << " - v_p_" << i << ";" << std::endl;
        ss << "av_end[" << i << "] = min(av_start[" << i << "] + ((v_k_"
           << i << " - 1) * v_d_"
           << i << " + 1), v_imsi_" << i << " + v_p_" << i << ");"
           << std::endl;
        ss << "ave *= ((av_end[" << i << "] - av_start[" << i << "] - 1) / v_d_"
           << i << " + 1);"
           << std::endl;
      }
    }
    // Dilation filters
    bool has_dilation = false;
    for (int_tp i = 0; i < num_axes_; ++i) {
      if (dilation_[i] > 1) {
        has_dilation = true;
      }
    }
    if (has_dilation &&
        (pool_method_ == LIBDNN_POOLING_METHOD_AVE ||
        pool_method_ == LIBDNN_POOLING_METHOD_STO)) {
      ss << "if (";
      for (int i  = 0; i < num_axes_; ++i) {
        ss << "idx_" << i << " >= av_start[" << i << "] && ";
        ss << "idx_" << i << " < av_end[" << i << "] && ";
        ss << "(idx_" << i <<" - av_start[" << i << "]) % v_d_" << i << " == 0"
           << " && ";
      }
      ss << "true) {" << std::endl;
    }
    if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
      ss << "if ((int_tp)mask_ptr[offset] == out_idx) {" << std::endl;
    } else if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << "if ((int_tp)rand_ptr[offset] == out_idx) {" << std::endl;
    } else {
      ss << "{" << std::endl;
    }
    ss << "gradient += in_ptr[offset]";
    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      ss << " / (MItype)ave;" << std::endl;
    } else {
      ss << ";" << std::endl;
    }
    ss << "}" << std::endl;
    if (has_dilation &&
        (pool_method_ == LIBDNN_POOLING_METHOD_AVE ||
        pool_method_ == LIBDNN_POOLING_METHOD_STO)) {
      ss << "}" << std::endl;
    }
    // Increment
    ss << "incremented = false;" << std::endl;
    ss << "for (int_tp i = v_nax - 1; i >= 0; --i) {" << std::endl;
    ss << "if (d_iter[i] >= d_end[i]) {" << std::endl;
    ss << "d_iter[i] = d_start[i];" << std::endl;
    ss << "} else {" << std::endl;
    ss << "++d_iter[i];" << std::endl;
    ss << "incremented = true;" << std::endl;
    ss << "break;" << std::endl;
    ss << "}}} while (incremented);" << std::endl;

    ss << "out_ptr[0] = gradient;" << std::endl;
  }  // Deterministic kernel
  ss << "}" << std::endl;  // Kernel

  return ss.str();
}

template<typename MItype, typename MOtype>
void LibDNNPool<MItype, MOtype>::GenerateKernels() {
  this->program_ = this->dev_ptr_->CreateProgram();

  stringstream ss;
  ss << this->program_->setup();
  ss << this->program_->template define_vector_type<MItype>("MItype", 0, 16);
  ss << this->program_->template define_vector_type<MItype>("MItype", 0, 16);
  ss << this->program_->template define_vector_type<MOtype>("MOtype", 0, 16);
  ss << this->program_->atomics();
  ss << generate_fw_defs();
  ss << generate_fwtr_kernels("pool_forward_train");
  ss << generate_fwte_kernels("pool_forward_test");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("pool_backward");

  // Write complete kernel string
  this->program_->set_source(ss.str());
}

template<typename MItype, typename MOtype>
bool LibDNNPool<MItype, MOtype>::CompileKernels() {
  return this->program_->Compile(true, true);
}

template<typename MItype, typename MOtype>
void LibDNNPool<MItype, MOtype>::Forward(
    vptr<const MItype> bottom_data, vptr<MOtype> top_data,
    int_tp channels, int_tp batch_size,
    bool test_mode, vptr<int_tp> mask,
    vptr<MOtype> top_mask, vptr<MItype> rand_idx) {
  int_tp imsi = std::accumulate(im_in_shape_.begin(), im_in_shape_.end(),
                                1, std::multiplies<int_tp>());
  int_tp imso = std::accumulate(im_out_shape_.begin(), im_out_shape_.end(),
                                1, std::multiplies<int_tp>());

  size_t lw0 = static_cast<size_t>(fw_tuner_->get_param<int_tp>("LW0"));
  size_t lw1 = static_cast<size_t>(fw_tuner_->get_param<int_tp>("LW1"));

  shared_ptr<DeviceKernel> kernel =
      this->program_->GetKernel(test_mode ? "pool_forward_test"
                                          : "pool_forward_train");
  vector<size_t> group = {static_cast<size_t>((imso - 1) / lw0 + 1),
                 static_cast<size_t>((channels * batch_size - 1) / lw1 + 1), 1};
  vector<size_t> local = {lw0, lw1, 1};

  kernel->add_arg(&bottom_data);
  kernel->add_arg(&top_data);
  switch (pool_method_) {
    case LIBDNN_POOLING_METHOD_MAX: {
      if (use_top_mask_) {
        kernel->add_arg(&top_mask);
      } else {
        kernel->add_arg(&mask);
      }
      break;
    }
    case LIBDNN_POOLING_METHOD_AVE: {
      break;
    }
    case LIBDNN_POOLING_METHOD_STO: {
      kernel->add_arg(&rand_idx);
      break;
    }
  }
  kernel->add_arg(&channels);
  kernel->add_arg(&batch_size);
  kernel->Execute(group, local);
}

template<typename MItype, typename MOtype>
void LibDNNPool<MItype, MOtype>::Backward(
                      vptr<const MOtype> top_diff, vptr<MItype> bottom_diff,
                      int_tp channels, int_tp batch_size,
                      vptr<const int_tp> mask, vptr<const MOtype> top_mask,
                      vptr<const MItype> rand_idx) {
  int_tp ims = batch_size * channels;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    ims *= im_in_shape_[i];
  }
  this->dev_ptr_->template set<MItype>(ims, (MItype)0, bottom_diff);

  int_tp imsi = std::accumulate(im_in_shape_.begin(), im_in_shape_.end(),
                                1, std::multiplies<int_tp>());
  int_tp imso = std::accumulate(im_out_shape_.begin(), im_out_shape_.end(),
                                1, std::multiplies<int_tp>());

  int_tp imsw = 0;
  if (bwalgo_ == LIBDNN_POOLING_BW_ALGO_DIRECT) {
    // Direct kernel iterates over input size
    imsw = imsi;
  } else {
    // Atomic kernel iterates over output size
    imsw = imso;
  }

  size_t lw0 = static_cast<size_t>(bw_tuner_->get_param<int_tp>("LW0"));
  size_t lw1 = static_cast<size_t>(bw_tuner_->get_param<int_tp>("LW1"));

  shared_ptr<DeviceKernel> kernel =
      this->program_->GetKernel("pool_backward");
  vector<size_t> group = {static_cast<size_t>((imsw - 1) / lw0 + 1),
                 static_cast<size_t>((channels * batch_size - 1) / lw1 + 1), 1};
  vector<size_t> local = {lw0, lw1, 1};

  kernel->add_arg(&top_diff);
  kernel->add_arg(&bottom_diff);
  switch (pool_method_) {
    case LIBDNN_POOLING_METHOD_MAX: {
      if (use_top_mask_) {
        kernel->add_arg(&top_mask);
      } else {
        kernel->add_arg(&mask);
      }
      break;
    }
    case LIBDNN_POOLING_METHOD_AVE: {
      break;
    }
    case LIBDNN_POOLING_METHOD_STO: {
      kernel->add_arg(&rand_idx);
      break;
    }
  }
  kernel->add_arg(&channels);
  kernel->add_arg(&batch_size);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASS_2T_GUARDED(LibDNNPool, PROTO_TYPES, PROTO_TYPES);

}  // namespace caffe

#endif  // USE_LIBDNN
