#include <functional>
#include <numeric>
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
LibDNNPool<Dtype>::LibDNNPool(LibDNNPoolConfig config) {
  config_ = config;
  LibDNN<Dtype>::dev_ptr_ = config.dev_ptr;
  LibDNN<Dtype>::fast_unsafe_math_ = config.fast_unsafe_math;
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

  fw_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());
  bw_tuner_ = std::shared_ptr<LibDNNTuner>(new LibDNNTuner());

  fw_tuner_->add_range_param<int_tp>("LW0", 8, 4, 16, 4);
  bw_tuner_->add_range_param<int_tp>("LW0", 8, 4, 16, 4);
  fw_tuner_->add_range_param<int_tp>("LW1", 8, 4, 16, 4);
  bw_tuner_->add_range_param<int_tp>("LW1", 8, 4, 16, 4);


  GenerateKernels();
  LibDNN<Dtype>::CompileKernels();
}

template<typename Dtype>
const LibDNNPoolConfig LibDNNPool<Dtype>::get_config() {
  return config_;
}


template<typename Dtype>
std::string LibDNNPool<Dtype>::string_identifier() {
  std::stringstream ss;
  ss << "POOL_";
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
  ss << "]";
  return ss.str();
}

template<typename Dtype>
std::string LibDNNPool<Dtype>::generate_fw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", num_axes_);

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

  return ss.str();
}


template<typename Dtype>
std::string LibDNNPool<Dtype>::generate_bw_defs() {
  std::stringstream ss;

  // Number of spatial axes
  LibDNN<Dtype>::add_def(ss, "v_nax", num_axes_);
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

  return ss.str();
}

template<typename Dtype>
std::string LibDNNPool<Dtype>::generate_fw_kernels(std::string name,
                                                   bool test_mode) {
  std::stringstream ss;
#ifdef HAS_HALF_SUPPORT
  if (std::is_same<Dtype, half_float::half>::value) {
    ss << "#define DTYPE_MAX HALF_MAX" << std::endl;
    ss << "#define DTYPE_MIN HALF_MIN" << std::endl;
  } else {
#endif
    ss << "#define DTYPE_MAX FLT_MAX" << std::endl;
    ss << "#define DTYPE_MIN FLT_MIN" << std::endl;
#ifdef HAS_HALF_SUPPORT
  }
#endif

  ss << "__kernel void " + name + "(";
  ss << "__global const Dtype* __restrict bottom_data, ";
  ss << "__global Dtype* __restrict top_data, ";
  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    if (use_top_mask_) {
      ss << "__global Dtype* __restrict top_mask, ";
    } else {
      ss << "__global int_tp* __restrict mask, ";
    }
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_STO && !test_mode) {
    ss << "__global Dtype* __restrict rand_idx, ";
  }
  ss << "int_tp channels, ";
  ss << "int_tp batch_size";
  ss << ") {" << std::endl;

  ss << "int_tp out_idx = get_global_id(0);" << std::endl;
  ss << "if (get_global_id(1) >= channels * batch_size) {return;}" << std::endl;
  ss << "int_tp idx_0 = get_global_id(0);" << std::endl;
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
  ss << "__global const Dtype* in_ptr = bottom_data + "
     << "get_global_id(1) * v_imsi + in_idx;" << std::endl;
  ss << "__global Dtype* out_ptr = top_data + "
     << "get_global_id(1) * v_imso;" << std::endl;

  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    if (use_top_mask_) {
      ss << "__global Dtype* mask_ptr = top_mask + get_global_id(1) * v_imso;"
         << std::endl;
    } else {
      ss << "__global int_tp* mask_ptr = mask + get_global_id(1) * v_imso;"
         << std::endl;
    }
    ss << "Dtype val = -DTYPE_MAX;" << std::endl;
    ss << "int_tp maxidx = -1;" << std::endl;
  }

  if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
    ss << "Dtype val = 0;" << std::endl;
  }

  if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
    if (test_mode) {
      ss << "Dtype cumsum = DTYPE_MIN;" << std::endl;
      ss << "Dtype cumvalues = 0;" << std::endl;
    } else {
      ss << "__global Dtype* rand_ptr = rand_idx + get_global_id(1) * v_imso;"
         << std::endl;
      ss << "Dtype val = 0;" << std::endl;
      ss << "Dtype cumsum = 0;" << std::endl;
      ss << "int_tp stoidx = -1;" << std::endl;
    }
  }

  std::vector<int_tp> d_iter;
  int_tp curr_idx = 0;

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
      ss << "Dtype thres = rand_ptr[out_idx] * cumsum;" << std::endl;
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
    ss << "out_ptr[out_idx] = val / ((Dtype)ave);" << std::endl;
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    ss << "out_ptr[out_idx] = val;" << std::endl;
    ss << "mask_ptr[out_idx] = (Dtype)maxidx;" << std::endl;
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
    if (test_mode) {
      ss << "out_ptr[out_idx] = cumvalues / cumsum;" << std::endl;
    } else {
      ss << "out_ptr[out_idx] = val;" << std::endl;
      ss << "rand_ptr[out_idx] = (Dtype)stoidx;" << std::endl;
    }
  }

  ss << "}" << std::endl;  // Kernel
  return ss.str();
}

template<typename Dtype>
std::string LibDNNPool<Dtype>::generate_fwtr_kernels(std::string name) {
  std::stringstream ss;
  ss << generate_fw_kernels(name, false);
  return ss.str();
}

template<typename Dtype>
std::string LibDNNPool<Dtype>::generate_fwte_kernels(std::string name) {
  std::stringstream ss;
  ss << generate_fw_kernels(name, true);
  return ss.str();
}



template<typename Dtype>
std::string LibDNNPool<Dtype>::generate_bw_kernels(std::string name) {
  std::stringstream ss;

  ss << "__kernel void " + name + "(";
  ss << "__global const Dtype* __restrict top_diff, ";
  ss << "__global Dtype* __restrict bottom_diff, ";
  if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
    if (use_top_mask_) {
      ss << "__global const Dtype* __restrict top_mask, ";
    } else {
      ss << "__global const int_tp* __restrict mask, ";
    }
  }
  if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
    ss << "__global const Dtype* __restrict rand_idx, ";
  }
  ss << "int_tp channels, ";
  ss << "int_tp batch_size";
  ss << ") {" << std::endl;
  if (bwalgo_ == LIBDNN_POOLING_BW_ALGO_ATOMIC) {
    // Atomic kernel
    ss << "int_tp in_idx = get_global_id(0);" << std::endl;
    ss << "if (get_global_id(1) >= channels * batch_size) {return;}"
       << std::endl;
    ss << "int_tp idx_0 = get_global_id(0);" << std::endl;
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
      ss << "__global Dtype* out_ptr = bottom_diff "
         << "+ get_global_id(1) * v_imsi + out_idx;" << std::endl;
    } else {
      ss << "__global Dtype* out_ptr = bottom_diff "
         << "+ get_global_id(1) * v_imsi;" << std::endl;
    }
    ss << "__global const Dtype* in_ptr = top_diff "
       << "+ get_global_id(1) * v_imso + in_idx;" << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
      if (use_top_mask_) {
        ss << "__global const Dtype* mask_ptr = top_mask "
           << "+ get_global_id(1) * v_imso + in_idx;" << std::endl;
      } else {
        ss << "__global const int_tp* mask_ptr = mask "
           << "+ get_global_id(1) * v_imso + in_idx;" << std::endl;
      }
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << "__global const Dtype* rand_ptr = rand_idx "
         << "+ get_global_id(1) * v_imso + in_idx;" << std::endl;
    }

    std::vector<int_tp> d_iter;

    for (int_tp i = 0; i < kernel_shape_.size(); ++i) {
      d_iter.push_back(0);
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      int_tp ave = std::accumulate(kernel_shape_.begin(),
                                   kernel_shape_.end(),
                                    1, std::multiplies<int_tp>());
      ss << "int_tp ave = " << ave << ";" << std::endl;
      ss << "Dtype val = in_ptr[0];" << std::endl;
    }

    for (int_tp ave_idx = 0;
         ave_idx < ((pool_method_ == LIBDNN_POOLING_METHOD_AVE)
         ? 2 : 0); ++ave_idx) {
      if (ave_idx == 1) {
        ss << "val /= ((Dtype)ave);" << std::endl;
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
            ss << "atomicAdd((&out_ptr[" << kernel_offset << "]), val);"
               << std::endl;
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
      ss << "atomicAdd(&out_ptr[(int_tp)(mask_ptr[0])], "
         << "in_ptr[0]);" << std::endl;
      ss << "}" << std::endl;
    }
    if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << "if (mask_ptr[0] >= 0 && mask_ptr[0] < v_imsi) {" << std::endl;
      ss << "atomicAdd(&out_ptr[(int_tp)(rand_ptr[0])], "
         << "in_ptr[0]);" << std::endl;
      ss << "}" << std::endl;
    }

  } else {
    // Direct, deterministic kernel
    ss << "int_tp d_start[" << num_axes_ << "];" << std::endl;
    ss << "int_tp d_end[" << num_axes_ << "];" << std::endl;
    ss << "int_tp d_iter[" << num_axes_ << "];" << std::endl;

    ss << "int_tp out_idx = get_global_id(0);" << std::endl;
    ss << "int_tp idx_0 = get_global_id(0);" << std::endl;
    ss << "if (get_global_id(1) >= channels * batch_size) {return;}"
       << std::endl;

    for (int_tp i = num_axes_ - 1; i >= 1; --i) {
      ss << "int_tp idx_" << i << " = (idx_0 % v_imsi_" << i << ");"
         << std::endl;
      ss << "idx_0 /= v_imsi_" << i << ";" << std::endl;
    }
    ss << "if (idx_0 >= v_imsi_0) {return;}" << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_AVE) {
      ss << "__global Dtype* out_ptr = bottom_diff "
         << "+ get_global_id(1) * v_imsi + out_idx;" << std::endl;
    } else {
      ss << "__global Dtype* out_ptr = bottom_diff "
         << "+ get_global_id(1) * v_imsi + out_idx;" << std::endl;
    }
    ss << "__global const Dtype* in_ptr = top_diff "
       << "+ get_global_id(1) * v_imso;" << std::endl;

    if (pool_method_ == LIBDNN_POOLING_METHOD_MAX) {
      if (use_top_mask_) {
        ss << "__global const Dtype* mask_ptr = top_mask "
           << "+ get_global_id(1) * v_imso;" << std::endl;
      } else {
        ss << "__global const int_tp* mask_ptr = mask "
           << "+ get_global_id(1) * v_imso;" << std::endl;
      }
    }

    if (pool_method_ == LIBDNN_POOLING_METHOD_STO) {
      ss << "__global const Dtype* rand_ptr = rand_idx "
         << "+ get_global_id(1) * v_imso;" << std::endl;
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
    // ss << "printf(\"%f\\n\", (float)ave);" << std::endl;
    ss << "Dtype gradient = 0.0;" << std::endl;
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
      ss << " / (Dtype)ave;" << std::endl;
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

template<typename Dtype>
void LibDNNPool<Dtype>::GenerateKernels() {
  std::stringstream ss;

  ss << LibDNN<Dtype>::generate_header();
  ss << generate_fw_defs();
  ss << generate_fwtr_kernels("pool_forward_train");
  ss << generate_fwte_kernels("pool_forward_test");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("pool_backward");

  // Write complete kernel string
  LibDNN<Dtype>::kernel_ = ss.str();
}

template<typename Dtype>
void LibDNNPool<Dtype>::Forward(const Dtype* bottom_data,
                                Dtype* top_data,
                                int_tp channels,
                                int_tp batch_size,
                                bool test_mode,
                                int_tp* mask,
                                Dtype* top_mask,
                                Dtype* rand_idx) {
  int_tp imsi = std::accumulate(im_in_shape_.begin(), im_in_shape_.end(),
                                1, std::multiplies<int_tp>());
  int_tp imso = std::accumulate(im_out_shape_.begin(), im_out_shape_.end(),
                                1, std::multiplies<int_tp>());

  int_tp lw0 = fw_tuner_->get_param<int_tp>("LW0");
  int_tp lw1 = fw_tuner_->get_param<int_tp>("LW1");

#ifdef USE_GREENTEA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
    viennacl::ocl::kernel &kernel =
        LibDNN<Dtype>::ocl_program_.get_kernel(
        test_mode ? "pool_forward_test" : "pool_forward_train");
    viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

    kernel.local_work_size(0, lw0);
    kernel.local_work_size(1, lw1);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((imso - 1) / lw0 + 1) * lw0);
    kernel.global_work_size(1, ((channels * batch_size - 1) / lw1 + 1) * lw1);
    kernel.global_work_size(2, 1);

    switch (pool_method_) {
      case LIBDNN_POOLING_METHOD_MAX:
        if (use_top_mask_) {
          viennacl::ocl::enqueue(
                 kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                        WrapHandle((cl_mem) top_data, &ctx),
                        WrapHandle((cl_mem) top_mask, &ctx),
                        channels,
                        batch_size),
                 ctx.get_queue());
        } else {
         viennacl::ocl::enqueue(
                kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                       WrapHandle((cl_mem) top_data, &ctx),
                       WrapHandle((cl_mem) mask, &ctx),
                       channels,
                       batch_size),
                ctx.get_queue());
        }
        break;
      case LIBDNN_POOLING_METHOD_AVE:
        viennacl::ocl::enqueue(
               kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                      WrapHandle((cl_mem) top_data, &ctx),
                      channels,
                      batch_size),
               ctx.get_queue());
        break;
      case LIBDNN_POOLING_METHOD_STO:
        viennacl::ocl::enqueue(
               kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                      WrapHandle((cl_mem) top_data, &ctx),
                      WrapHandle((cl_mem) rand_idx, &ctx),
                      channels,
                      batch_size),
               ctx.get_queue());
        break;
    }
  }
#endif  // USE_GREENTEA

#ifdef USE_CUDA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    CUfunction kernel;
    cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
               test_mode ? "pool_forward_test" : "pool_forward_train");

    switch (pool_method_) {
      case LIBDNN_POOLING_METHOD_MAX: {
        if (use_top_mask_) {
          void *args[] = { &bottom_data, &top_data, &top_mask,
              &channels, &batch_size };
          cuLaunchKernel(kernel,
                         (imso - 1) / lw0 + 1,                   // Grid X
                         (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                         1,                                      // Grid Z
                         lw0, lw1, 1,                            // Local
                         0, NULL, args, 0);                      // Arguments
        } else {
          void *args[] = { &bottom_data, &top_data, &mask,
              &channels, &batch_size };
          cuLaunchKernel(kernel,
                         (imso - 1) / lw0 + 1,                   // Grid X
                         (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                         1,                                      // Grid Z
                         lw0, lw1, 1,                            // Local
                         0, NULL, args, 0);                      // Arguments
        }
        break;
      }
      case LIBDNN_POOLING_METHOD_AVE: {
        void *args[] = { &bottom_data, &top_data,
            &channels, &batch_size };
        cuLaunchKernel(kernel,
                       (imso - 1) / lw0 + 1,                   // Grid X
                       (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                       1,                                      // Grid Z
                       lw0, lw1, 1,                            // Local
                       0, NULL, args, 0);                      // Arguments
        break;
      }
      case LIBDNN_POOLING_METHOD_STO: {
        void *args[] = { &bottom_data, &top_data, &rand_idx,
            &channels, &batch_size };
        cuLaunchKernel(kernel,
                       (imso - 1) / lw0 + 1,                   // Grid X
                       (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                       1,                                      // Grid Z
                       lw0, lw1, 1,                            // Local
                       0, NULL, args, 0);                      // Arguments
        break;
      }
    }
    cuCtxSynchronize();
  }
#endif  // USE_CUDA
}


template<typename Dtype>
void LibDNNPool<Dtype>::Backward(const Dtype* top_diff,
                                Dtype* bottom_diff,
                                int_tp channels,
                                int_tp batch_size,
                                const int_tp* mask,
                                const Dtype* top_mask,
                                const Dtype* rand_idx) {
  int_tp ims = batch_size * channels;
  for (int_tp i = 0; i < im_in_shape_.size(); ++i) {
    ims *= im_in_shape_[i];
  }
  LibDNN<Dtype>::SetMemory(bottom_diff, ims, 0, (Dtype) 0);

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

  int_tp lw0 = bw_tuner_->get_param<int_tp>("LW0");
  int_tp lw1 = bw_tuner_->get_param<int_tp>("LW1");

#ifdef USE_GREENTEA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
    viennacl::ocl::kernel &kernel =
        LibDNN<Dtype>::ocl_program_.get_kernel("pool_backward");
    viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

    kernel.local_work_size(0, lw0);
    kernel.local_work_size(1, lw1);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((imsw - 1) / lw0 + 1) * lw0);
    kernel.global_work_size(1, ((channels * batch_size - 1) / lw1 + 1) * lw1);
    kernel.global_work_size(2, 1);

    switch (pool_method_) {
      case LIBDNN_POOLING_METHOD_MAX:
        if (use_top_mask_) {
          viennacl::ocl::enqueue(
                 kernel(WrapHandle((cl_mem) top_diff, &ctx),
                        WrapHandle((cl_mem) bottom_diff, &ctx),
                        WrapHandle((cl_mem) top_mask, &ctx),
                        channels,
                        batch_size),
                 ctx.get_queue());
        } else {
         viennacl::ocl::enqueue(
                kernel(WrapHandle((cl_mem) top_diff, &ctx),
                       WrapHandle((cl_mem) bottom_diff, &ctx),
                       WrapHandle((cl_mem) mask, &ctx),
                       channels,
                       batch_size),
                ctx.get_queue());
        }
        break;
      case LIBDNN_POOLING_METHOD_AVE:
        viennacl::ocl::enqueue(
               kernel(WrapHandle((cl_mem) top_diff, &ctx),
                      WrapHandle((cl_mem) bottom_diff, &ctx),
                      channels,
                      batch_size),
               ctx.get_queue());
        break;
      case LIBDNN_POOLING_METHOD_STO:
        viennacl::ocl::enqueue(
               kernel(WrapHandle((cl_mem) top_diff, &ctx),
                      WrapHandle((cl_mem) bottom_diff, &ctx),
                      WrapHandle((cl_mem) rand_idx, &ctx),
                      channels,
                      batch_size),
               ctx.get_queue());
        break;
    }
  }
#endif  // USE_GREENTEA

#ifdef USE_CUDA
  if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
    CUfunction kernel;
    cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_, "pool_backward");

    switch (pool_method_) {
      case LIBDNN_POOLING_METHOD_MAX: {
        if (use_top_mask_) {
          void *args[] = { &top_diff, &bottom_diff, &top_mask,
              &channels, &batch_size };
          cuLaunchKernel(kernel,
                         (imsw - 1) / lw0 + 1,                   // Grid X
                         (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                         1,                                      // Grid Z
                         lw0, lw1, 1,                            // Local
                         0, NULL, args, 0);                      // Arguments
        } else {
          void *args[] = { &top_diff, &bottom_diff, &mask,
              &channels, &batch_size };
          cuLaunchKernel(kernel,
                         (imsw - 1) / lw0 + 1,                   // Grid X
                         (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                         1,                                      // Grid Z
                         lw0, lw1, 1,                            // Local
                         0, NULL, args, 0);                      // Arguments
        }
        break;
      }
      case LIBDNN_POOLING_METHOD_AVE: {
        void *args[] = { &top_diff, &bottom_diff,
            &channels, &batch_size };
        cuLaunchKernel(kernel,
                       (imsw - 1) / lw0 + 1,                   // Grid X
                       (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                       1,                                      // Grid Z
                       lw0, lw1, 1,                            // Local
                       0, NULL, args, 0);                      // Arguments
        break;
      }
      case LIBDNN_POOLING_METHOD_STO: {
        void *args[] = { &top_diff, &bottom_diff, &rand_idx,
            &channels, &batch_size };
        cuLaunchKernel(kernel,
                       (imsw - 1) / lw0 + 1,                   // Grid X
                       (channels * batch_size - 1) / lw1 + 1,  // Grid Y
                       1,                                      // Grid Z
                       lw0, lw1, 1,                            // Local
                       0, NULL, args, 0);                      // Arguments
        break;
      }
    }
    cuCtxSynchronize();
  }
#endif  // USE_CUDA
}

INSTANTIATE_CLASS(LibDNNPool);

}  // namespace caffe

#endif  // USE_LIBDNN
