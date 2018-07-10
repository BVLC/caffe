#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void AccuracyLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  {
    KernelArgs fw_args;
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<uint_tp>("nthreads", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "label", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "acc", KERNEL_ARG_GLOBAL_MEM));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dim", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num_labels", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "top_k", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                      "has_ignore_label", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ignore_label", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "counts", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("AccuracyForward", fw_args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");

    ss << "const int_tp n = index / spatial_dim;" << std::endl;
    ss << "const int_tp s = index % spatial_dim;" << std::endl;
    ss << "const int_tp label_value = (int_tp)(label[n * spatial_dim + s]);"
       << std::endl;
    ss << "const Dtype prob_of_true_class = bottom_data[n * dim"
       << " + label_value * spatial_dim + s];" << std::endl;
    ss << "int_tp num_better_predictions = -1;"
       << " // true_class also counts as \"better\"" << std::endl;
    ss << "if (has_ignore_label && label_value == ignore_label) {" << std::endl;
    ss << "acc[index] = 0;" << std::endl;
    ss << "counts[index] = 0;" << std::endl;
    ss << "} else {" << std::endl;
    ss << "for (int_tp k = 0; k < num_labels & num_better_predictions < top_k;"
       << " k++) {" << std::endl;
    ss << "num_better_predictions += "
       << "(bottom_data[n * dim + k * spatial_dim + s] >= prob_of_true_class);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "acc[index] = (num_better_predictions < top_k) ? "
       << "(MItype)1 : (MItype)0;" << std::endl;
    ss << "counts[index] = (MItype)1;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  {
    KernelArgs fw_args;
    fw_args.push_back(this->device_program_->template
                      create_kernel_arg<uint_tp>("nthreads", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "label", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "acc", KERNEL_ARG_GLOBAL_MEM));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "dim", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "spatial_dim", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "num_labels", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "top_k", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<bool>(
                      "has_ignore_label", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                      "ignore_label", KERNEL_ARG_CONST));
    fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                      "counts", KERNEL_ARG_GLOBAL_MEM));
    ss << this->device_program_->function("AccuracyForwardWithPerClass",
                                          fw_args);
    ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");

    ss << "const int_tp n = index / spatial_dim;" << std::endl;
    ss << "const int_tp s = index % spatial_dim;" << std::endl;
    ss << "const int_tp label_value = (int_tp)(label[n * spatial_dim + s]);"
       << std::endl;
    ss << "const Dtype prob_of_true_class = bottom_data[n * dim"
       << " + label_value * spatial_dim + s];" << std::endl;
    ss << "if (has_ignore_label && label_value == ignore_label) {"
       << std::endl;
    ss << "// nothing to be done." << std::endl;
    ss << "} else {" << std::endl;
    ss << "int_tp num_better_predictions = -1;"
       << "  // true_class also counts as \"better\"" << std::endl;
    ss << "for (int_tp k = 0; k < num_labels & num_better_predictions < top_k; "
       << "k++) {" << std::endl;
    ss << "num_better_predictions += "
       << "(bottom_data[n * dim + k * spatial_dim + s] >= prob_of_true_class);"
       << std::endl;
    ss << "}" << std::endl;
    ss << "acc[label_value*nthreads + index]"
       << " += (num_better_predictions < top_k) ? (MItype) 1 : (MItype) 0;"
       << std::endl;
    ss << "counts[label_value*nthreads + index] = (Dtype)1;" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
    ss << "}" << std::endl;
  }

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void AccuracyLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom, const vector<Blob<MOtype>*>& top) {
  vptr<const MItype> bottom_data = bottom[0]->gpu_data();
  vptr<const MItype> bottom_label = bottom[1]->gpu_data();
  const int_tp dim = bottom[0]->count() / outer_num_;
  const int_tp num_labels = bottom[0]->shape(label_axis_);
  const uint_tp nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  vptr<MItype> acc_data = bottom[0]->mutable_gpu_diff();
  if (top.size() == 1) {
    // simple case - report only global accuracy.

    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    vptr<MItype> counts = bottom[1]->mutable_gpu_diff();

    shared_ptr<DeviceKernel> kernel =
                          this->device_program_->GetKernel("AccuracyForward");

    kernel->add_arg(&nthreads);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_label);
    kernel->add_arg(&acc_data);
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&dim);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&num_labels);
    kernel->add_arg(&top_k_);
    kernel->add_arg(&has_ignore_label_);
    kernel->add_arg(&ignore_label_);
    kernel->add_arg(&counts);

    vector<size_t> work_size(1, nthreads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(),
                               true);
    kernel->Execute(group, local);

    MOtype acc;
    this->device_->template asum<MOtype>(nthreads, acc_data, &acc);
    MOtype valid_count;
    this->device_->template asum<MOtype>(nthreads, counts, &valid_count);
    if (valid_count > 0) {
      top[0]->mutable_cpu_data()[0] = acc / valid_count;
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }
  } else {
    // need to report per-class accuracy as well

    // allocate space for more detailed "counts"
    nums_buffer_.ReshapeLike(bottom[0]);
    vptr<Dtype> counts = nums_buffer_.mutable_gpu_data();

    this->device_->template set<MItype>(bottom[0]->count(), MItype(0),
                                        acc_data);
    this->device_->template set<Dtype>(nums_buffer_.count(), Dtype(0), counts);

    shared_ptr<DeviceKernel> kernel =
                this->device_program_->GetKernel("AccuracyForwardWithPerClass");

    kernel->add_arg(&nthreads);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&bottom_label);
    kernel->add_arg(&acc_data);
    kernel->add_arg(&outer_num_);
    kernel->add_arg(&dim);
    kernel->add_arg(&inner_num_);
    kernel->add_arg(&num_labels);
    kernel->add_arg(&top_k_);
    kernel->add_arg(&has_ignore_label_);
    kernel->add_arg(&ignore_label_);
    kernel->add_arg(&counts);

    vector<size_t> work_size(1, nthreads);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    // get the overall accuracy
    MItype acc;
    this->device_->template asum<MItype>(bottom[0]->count(), acc_data, &acc);
    Dtype valid_count;
    this->device_->template asum<Dtype>(nums_buffer_.count(), counts,
                                         &valid_count);
    if (valid_count > 0) {
      top[0]->mutable_cpu_data()[0] = static_cast<MOtype>(acc /
                                              static_cast<MItype>(valid_count));
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }

    // get per-class accuracy
    MOtype* per_class_acc = top[1]->mutable_cpu_data();
    for (int_tp l = 0; l < num_labels; l++) {
      this->device_->template asum<MOtype>(nthreads,
                                    acc_data + l * nthreads, per_class_acc + l);
      this->device_->template asum<Dtype>(nthreads,
                                           counts + l * nthreads, &valid_count);
      if (valid_count > 0) {
        per_class_acc[l] /= static_cast<MOtype>(valid_count);
      } else {
        per_class_acc[l] = MOtype(0);
      }
    }
  }
  // Clear scratch memory to prevent interfering with backward (see #6202).
  this->device_->template set<MItype>(bottom[0]->count(),
                                      MItype(0), bottom[0]->mutable_gpu_diff());
}


template<typename Dtype, typename MItype, typename MOtype>
void AccuracyLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (propagate_down[1]) {  NOT_IMPLEMENTED;  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(AccuracyLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
