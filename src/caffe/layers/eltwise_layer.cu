#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void EltwiseLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

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
    ss << "#define DTYPE_MAX " << type_max_val<MItype>() << std::endl;
    ss << "#define DTYPE_MIN " << 0 << std::endl;
  }
#endif

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_data_a", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_data_b", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "blob_idx", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "top_data", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "mask", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("MaxForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "Dtype maxval = -DTYPE_MAX;" << std::endl;
  ss << "int_tp maxidx = -1;" << std::endl;
  ss << "if (bottom_data_a[index] > bottom_data_b[index]) {" << std::endl;
  // only update for very first bottom_data blob (blob_idx == 0)
  ss << "if (blob_idx == 0) {" << std::endl;
  ss << "maxval = bottom_data_a[index];" << std::endl;
  ss << "top_data[index] = maxval;" << std::endl;
  ss << "maxidx = blob_idx;" << std::endl;
  ss << "mask[index] = maxidx;" << std::endl;
  ss << "}" << std::endl;
  ss << "} else {" << std::endl;
  ss << "maxval = bottom_data_b[index];" << std::endl;
  ss << "top_data[index] = maxval;" << std::endl;
  ss << "maxidx = blob_idx + 1;" << std::endl;
  ss << "mask[index] = maxidx;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "blob_idx", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "bottom_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("MaxBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "Dtype gradient = 0;" << std::endl;
  ss << "if (mask[index] == blob_idx) {" << std::endl;
  ss << "gradient += top_diff[index];" << std::endl;
  ss << "}" << std::endl;
  ss << "bottom_diff[index] = gradient;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void EltwiseLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  vptr<int_tp> mask;
  const int_tp count = top[0]->count();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();

  switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD: {
      this->device_->template mul<Dtype>(count, bottom[0]->gpu_data(),
                                         bottom[1]->gpu_data(),
                         top_data);
      for (int_tp i = 2; i < bottom.size(); ++i) {
        this->device_->template mul<Dtype>(count, top_data,
                                           bottom[i]->gpu_data(), top_data);
      }
      break;
    }
    case EltwiseParameter_EltwiseOp_SUM: {
      this->device_->set(count, Dtype(0.), top_data);
      // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
      for (int_tp i = 0; i < bottom.size(); ++i) {
        this->device_->axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
      }
      break;
    }
    case EltwiseParameter_EltwiseOp_MAX: {
      mask = max_idx_.mutable_gpu_data();
      shared_ptr<DeviceKernel> kernel =
                                 this->device_program_->GetKernel("MaxForward");
      vptr<const Dtype> bottom_0_data = bottom[0]->gpu_data();
      vptr<const Dtype> bottom_1_data = bottom[1]->gpu_data();
      int_tp idx = 0;

      kernel->add_arg(&count);
      kernel->add_arg(&bottom_0_data);
      kernel->add_arg(&bottom_1_data);
      kernel->add_arg(&idx);
      kernel->add_arg(&top_data);
      kernel->add_arg(&mask);

      vector<size_t> work_size(1, count);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                 true);
      kernel->Execute(group, local);

      for (int_tp i = 2; i < bottom.size(); ++i) {

        vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
        int_tp idx = i - 1;

        kernel->add_arg(&count);
        kernel->add_arg(&top_data);
        kernel->add_arg(&bottom_data);
        kernel->add_arg(&idx);
        kernel->add_arg(&top_data);
        kernel->add_arg(&mask);

        vector<size_t> work_size(1, count);
        vector<size_t> group;
        vector<size_t> local;
        this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                   true);
        kernel->Execute(group, local);
      }
      break;
    }
    default: {
      LOG(FATAL)<< "Unknown elementwise operation.";
    }
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void EltwiseLayer<Dtype, MItype, MOtype>::Backward_gpu(
                                       const vector<Blob<MOtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<MItype>*>& bottom) {
  vptr<const int_tp> mask;
  const int_tp count = top[0]->count();
  vptr<const Dtype> top_data = top[0]->gpu_data();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();

  for (int_tp i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
      vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
        case EltwiseParameter_EltwiseOp_PROD: {
          if (stable_prod_grad_) {
            bool initialized = false;
            for (int_tp j = 0; j < bottom.size(); ++j) {
              if (i == j) {
                continue;
              }
              if (!initialized) {
                this->device_->copy(count, bottom[j]->gpu_data(), bottom_diff);
                initialized = true;
              } else {
                this->device_->template mul<Dtype>(count, bottom[j]->gpu_data(),
                                                   bottom_diff, bottom_diff);
              }
            }
          } else {
            this->device_->template div<Dtype>(count, top_data, bottom_data,
                                               bottom_diff);
          }
          this->device_->template mul<Dtype>(count, bottom_diff, top_diff,
                                             bottom_diff);
          break;
        }
        case EltwiseParameter_EltwiseOp_SUM: {
          if (coeffs_[i] == Dtype(1.)) {
            this->device_->copy(count, top_diff, bottom_diff);
          } else {
            this->device_->scale(count, coeffs_[i], top_diff, bottom_diff);
          }
          break;
        }
        case EltwiseParameter_EltwiseOp_MAX: {
          mask = max_idx_.gpu_data();
          shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("MaxBackward");

          kernel->add_arg(&count);
          kernel->add_arg(&top_diff);
          kernel->add_arg(&i);
          kernel->add_arg(&mask);
          kernel->add_arg(&bottom_diff);

          vector<size_t> work_size(1, count);
          vector<size_t> group;
          vector<size_t> local;
          this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                                     true);
          kernel->Execute(group, local);
          break;
        }
        default: {
          LOG(FATAL)<< "Unknown elementwise operation.";
        }
      }
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(EltwiseLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
