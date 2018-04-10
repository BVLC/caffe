#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void TileLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "bottom_data", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "tile_size", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num_tiles", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "bottom_tile_axis", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "top_data", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("TileForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "const int_tp d = index % tile_size;" << std::endl;
  ss << "const int_tp b = (index / tile_size / num_tiles) % bottom_tile_axis;"
     << std::endl;
  ss << "const int_tp n = index / tile_size / num_tiles / bottom_tile_axis;"
     << std::endl;
  ss << "const int_tp bottom_index = (n * bottom_tile_axis + b)"
     << " * tile_size + d;" << std::endl;
  ss << "top_data[index] = bottom_data[bottom_index];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "nthreads", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "top_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "tile_size", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "num_tiles", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<int_tp>(
                    "bottom_tile_axis", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("TileBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "nthreads");
  ss << "const int_tp d = index % tile_size;" << std::endl;
  ss << "const int_tp b = (index / tile_size) % bottom_tile_axis;" << std::endl;
  ss << "const int_tp n = index / tile_size / bottom_tile_axis;" << std::endl;
  ss << "bottom_diff[index] = 0;" << std::endl;
  ss << "int_tp top_index = (n * num_tiles * bottom_tile_axis + b)"
     << " * tile_size + d;" << std::endl;
  ss << "for (int_tp t = 0; t < num_tiles; ++t) {" << std::endl;
  ss << "bottom_diff[index] += top_diff[top_index];" << std::endl;
  ss << "top_index += bottom_tile_axis * tile_size;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void TileLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  const int_tp bottom_tile_axis = bottom[0]->shape(axis_);
  const int_tp nthreads = top[0]->count();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("TileForward");
  kernel->add_arg(&nthreads);
  kernel->add_arg(&bottom_data);
  kernel->add_arg(&inner_dim_);
  kernel->add_arg(&tiles_);
  kernel->add_arg(&bottom_tile_axis);
  kernel->add_arg(&top_data);

  vector<size_t> work_size(1, nthreads);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

template<typename Dtype, typename MItype, typename MOtype>
void TileLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
  const int_tp bottom_tile_axis = bottom[0]->shape(axis_);
  const int_tp tile_size = inner_dim_ / bottom_tile_axis;
  const int_tp nthreads = bottom[0]->count();

  shared_ptr<DeviceKernel> kernel =
                                this->device_program_->GetKernel("TileBackward");
  kernel->add_arg(&nthreads);
  kernel->add_arg(&top_diff);
  kernel->add_arg(&tile_size);
  kernel->add_arg(&tiles_);
  kernel->add_arg(&bottom_tile_axis);
  kernel->add_arg(&bottom_diff);

  vector<size_t> work_size(1, nthreads);
  vector<size_t> group;
  vector<size_t> local;
  this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
  kernel->Execute(group, local);
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, GenerateProgram,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, Forward_gpu,
                                  (double), (double), (double));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(TileLayer, Backward_gpu,
                                  (double), (double), (double));

}  // namespace caffe
