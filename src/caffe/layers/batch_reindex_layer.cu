#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void BatchReindexLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "count", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "inner_dim", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "permut", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("BRForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "count");
  ss << "uint_tp n = index / (inner_dim);" << std::endl;
  ss << "uint_tp in_n = (uint_tp)(permut[n]);" << std::endl;
  ss << "out[index] = in[in_n * (inner_dim) + index % (inner_dim)];"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "count", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "inner_dim", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "top_indexes", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "begins", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "counts", KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("BRBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "count");
  ss << "uint_tp n = index / (inner_dim);" << std::endl;
  ss << "out[index] = 0;" << std::endl;
  ss << "uint_tp lower = (uint_tp)(begins[n]);" << std::endl;
  ss << "uint_tp upper = lower + (uint_tp)(counts[n]);" << std::endl;
  ss << "for (uint_tp i = lower; i < upper; ++i) {" << std::endl;
  ss << "uint_tp in_n = (uint_tp)(top_indexes[i]);" << std::endl;
  ss << "out[index] += in[in_n * (inner_dim) + index % (inner_dim)];"
     << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}

template<typename Dtype, typename MItype, typename MOtype>
void BatchReindexLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                           const vector<Blob<MItype>*>& bottom,
                                           const vector<Blob<MOtype>*>& top) {
  check_batch_reindex(bottom[0]->shape(0), bottom[1]->count(),
                      bottom[1]->cpu_data());
  if (top[0]->count() == 0) {
    return;
  }
  const uint_tp count = top[0]->count();
  const uint_tp inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  vptr<const MItype> bottom_data = bottom[0]->gpu_data();
  vptr<const MItype> perm_data = bottom[1]->gpu_data();
  vptr<MOtype> top_data = top[0]->mutable_gpu_data();

  shared_ptr<DeviceKernel> kernel =
                                  this->device_program_->GetKernel("BRForward");
  kernel->add_arg(&count);
  kernel->add_arg(&inner_dim);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&perm_data);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}


template<typename Dtype, typename MItype, typename MOtype>
void BatchReindexLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  CHECK(!propagate_down[1]) << "Cannot backprop to index.";
  if (!propagate_down[0]) {
    return;
  }

  vector<std::pair<int_tp, int_tp> > mapping;
  const MItype* perm = bottom[1]->cpu_data();
  for (int_tp i = 0; i < bottom[1]->count(); ++i) {
    mapping.push_back(pair<int_tp, int_tp>(static_cast<int_tp>(perm[i]), i));
  }
  std::sort(mapping.begin(), mapping.end(), pair_sort_first());

  // Each element of the bottom diff is potentially the sum of many top diffs.
  // However, we'd like each CUDA thread to handle exactly one output.  Hence,
  // we first pre-compute a list of lists of indices that need to be summed for
  // each output. `top_indexes` holds the data of this list of lists.  The
  // k'th element of `begins` points to the location in `top_indexes` where the
  // list for the k'th example begin, and the k'th element of `counts` is the
  // length of that list.
  vector<int_tp> shape;
  shape.push_back(bottom[1]->count());
  Blob<Dtype> top_indexes(shape, this->device_);
  shape[0] = bottom[0]->shape(0);
  Blob<Dtype> counts(shape, this->device_);
  Blob<Dtype> begins(shape, this->device_);
  Dtype* t_i_data = top_indexes.mutable_cpu_data();
  Dtype* c_data = counts.mutable_cpu_data();
  Dtype* b_data = begins.mutable_cpu_data();
  caffe_set(begins.count(), Dtype(-1), b_data);
  caffe_set(counts.count(), Dtype(0), c_data);
  for (int_tp i = 0; i < mapping.size(); ++i) {
    t_i_data[i] = mapping[i].second;
    if (b_data[mapping[i].first] == -1) {
      b_data[mapping[i].first] = i;
    }
    c_data[mapping[i].first] += 1;
  }

  const uint_tp count = bottom[0]->count();
  const uint_tp inner_dim = bottom[0]->count() / bottom[0]->shape(0);
  vptr<const MOtype> top_diff = top[0]->gpu_diff();
  vptr<const Dtype> top_indexes_data = top_indexes.gpu_data();
  vptr<const Dtype> begins_data = begins.gpu_data();
  vptr<const Dtype> counts_data = counts.gpu_data();
  vptr<MItype> bottom_diff = bottom[0]->mutable_gpu_diff();

  shared_ptr<DeviceKernel> kernel =
                                 this->device_program_->GetKernel("BRBackward");
  kernel->add_arg(&count);
  kernel->add_arg(&inner_dim);
  kernel->add_arg(&top_diff);
  kernel->add_arg(&top_indexes_data);
  kernel->add_arg(&begins_data);
  kernel->add_arg(&counts_data);
  kernel->add_arg(&bottom_diff);

  vector<size_t> work_size(1, count);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(BatchReindexLayer, Backward_gpu,
                                  (double), (double), (double));
}  // namespace caffe
