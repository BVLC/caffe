#include <boost/filesystem.hpp>
#include "caffe/backend/cuda/cuda_device_program.hpp"
#include "caffe/backend/cuda/cuda_device.hpp"

namespace caffe {

#ifdef USE_CUDA

CudaDeviceProgram::CudaDeviceProgram(Device* dev) : DeviceProgram(dev) {
}

void CudaDeviceProgram::Compile(bool load_cache, bool store_cache) {
  nvrtcCreateProgram(&cuda_program_, src_.c_str(), NULL, 0, NULL, NULL);

  vector<const char*> build_opts;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, this->device_->id());

  string arch_opt = "--gpu-architecture=compute_"
      + std::to_string(prop.major) + std::to_string(prop.minor);
  string stdcpp_opt = "--std=c++11";
  string fum_opt = "--use_fast_math";

  build_opts.push_back(arch_opt.c_str());
  build_opts.push_back(stdcpp_opt.c_str());
  if (this->device_->is_fast_unsafe_math()) {
    build_opts.push_back(fum_opt.c_str());
  }
  nvrtcCompileProgram(cuda_program_, build_opts.size(), &build_opts[0]);

  size_t ptxSize;
  nvrtcGetPTXSize(cuda_program_, &ptxSize);
  char *ptx = new char[ptxSize];
  nvrtcGetPTX(cuda_program_, ptx);

  cuModuleLoadDataEx(&cuda_module_, ptx, 0, 0, 0);

#ifndef NDEBUG
  string debug_path = ".caffe_debug";
  const char* path = debug_path.c_str();
  boost::filesystem::path dir(path);
  boost::filesystem::create_directory(dir);


  size_t log_size;
  nvrtcGetProgramLogSize(cuda_program_, &log_size);
  vector<char> log(log_size);
  nvrtcGetProgramLog(cuda_program_, log.data());

  std::cout << "CUDA compile log:" << std::endl;
  std::cout << log.data() << std::endl;

  FILE* fp = fopen((".caffe_debug/" + string_identifier() + ".cuptx").c_str(),
                   "wb");
  fwrite(ptx, sizeof(char), ptxSize, fp);
  fclose(fp);
  free(ptx);
#endif  // NDEBUG
}


shared_ptr<DeviceKernel>
              CudaDeviceProgram::GetKernel(string name) {

  CUfunction kernel;
  cuModuleGetFunction(&kernel, cuda_module_, name.c_str());

  KernelArgs args;

  std::map<string, KernelArgs>::iterator pos = this->args_.find(name);
  if (pos == this->args_.end()) {
    LOG(FATAL) << "CUDA kernel " << name << " not found";
  } else {
    args = pos->second;
  }

  return std::make_shared<CudaDeviceKernel>(device_, kernel, args);
}

string CudaDeviceProgram::function(string name,
             vector<std::tuple<string, string, uint64_t>> args) {
  args_.insert(make_pair(name, args));
  stringstream ss;
  ss << "__global__ void ";
  ss << name << "(";
  for (uint_tp i = 0; i < args.size(); ++i) {
    uint64_t flags = std::get<2>(args[i]);
    if ((flags & KERNEL_ARG_CONST) == KERNEL_ARG_CONST) {
      ss << "const ";
    }
    ss << std::get<0>(args[i]) << " " << std::get<1>(args[i]);
    if (i < args.size() - 1) {
      ss << ", ";
    }
  }
  ss << ") {" << std::endl;
  return ss.str();
}

string CudaDeviceProgram::kernel_loop(string type,
                                             string index, string n) {
  stringstream ss;
  ss << "for (" << type << " "
     << index << " = blockIdx.x * blockDim.x + threadIdx.x; "
     << index << " < " << n << "; "
     << index << " += blockDim.x * gridDim.x) {" << std::endl;
  return ss.str();
}

string CudaDeviceProgram::setup() {
  stringstream ss;
  ss << this->define_type<int_tp>("int_tp");
  ss << this->define_type<uint_tp>("uint_tp");
#ifdef USE_INDEX_64
  ss << "#ifdef " << "int_tpc" << std::endl;
  ss << "#undef " << "int_tpc" << std::endl;
  ss << "#endif  //" << "int_tpc" << std::endl;
  ss << "#define " << "int_tpc" << "long long" << std::endl;
  ss << "#ifdef " << "uint_tpc" << std::endl;
  ss << "#undef " << "uint_tpc" << std::endl;
  ss << "#endif  //" << "uint_tpc" << std::endl;
  ss << "#define " << "uint_tpc" << "unsigned long long" << std::endl;
#else
  ss << "#ifdef " << "int_tpc" << std::endl;
  ss << "#undef " << "int_tpc" << std::endl;
  ss << "#endif  //" << "int_tpc" << std::endl;
  ss << "#define " << "int_tpc" << "int" << std::endl;
  ss << "#ifdef " << "uint_tpc" << std::endl;
  ss << "#undef " << "uint_tpc" << std::endl;
  ss << "#endif  //" << "uint_tpc" << std::endl;
  ss << "#define " << "uint_tpc" << "unsigned int" << std::endl;
#endif  // USE_INDEX_64
  return ss.str();
}

string CudaDeviceProgram::global_ptr(string type, string name) {
  return type + "* " + name;
}

string CudaDeviceProgram::local_ptr(string type, string name) {
  return type + "* " + name;
}

string CudaDeviceProgram::local_mem(string type) {
  return "__shared__ " + type;
}

string CudaDeviceProgram::local_id(uint_tp fixed_index) {
  switch(fixed_index) {
    case 0:
      return "threadIdx.x";
    case 1:
      return "threadIdx.y";
    case 2:
      return "threadIdx.z";
    default:
      return "threadIdx.x";
  }
}
string CudaDeviceProgram::local_id(string runtime_index) {
  stringstream ss;
  ss << "((" << runtime_index << " == 0) ?";
  ss << "threadIdx.x : ";
  ss << "((" << runtime_index << " == 1) ?";
  ss << "threadIdx.y : threadIdx.z))";
  return ss.str();
}
string CudaDeviceProgram::local_size(uint_tp fixed_index) {
  switch(fixed_index) {
    case 0:
      return "blockDim.x";
    case 1:
      return "blockDim.y";
    case 2:
      return "blockDim.z";
    default:
      return "blockDim.x";
  }
}
string CudaDeviceProgram::local_size(string runtime_index) {
  stringstream ss;
  ss << "((" << runtime_index << " == 0) ?";
  ss << "blockDim.x : ";
  ss << "((" << runtime_index << " == 1) ?";
  ss << "blockDim.y : blockDim.z))";
  return ss.str();
}
string CudaDeviceProgram::group_id(uint_tp fixed_index) {
  switch(fixed_index) {
    case 0:
      return "blockIdx.x";
    case 1:
      return "blockIdx.y";
    case 2:
      return "blockIdx.z";
    default:
      return "blockIdx.x";
  }
}
string CudaDeviceProgram::group_id(string runtime_index) {
  stringstream ss;
  ss << "((" << runtime_index << " == 0) ?";
  ss << "blockIdx.x : ";
  ss << "((" << runtime_index << " == 1) ?";
  ss << "blockIdx.y : blockIdx.z))";
  return ss.str();
}
string CudaDeviceProgram::group_size(uint_tp fixed_index) {
  switch(fixed_index) {
    case 0:
      return "gridDim.x";
    case 1:
      return "gridDim.y";
    case 2:
      return "gridDim.z";
    default:
      return "gridDim.x";
  }
}
string CudaDeviceProgram::group_size(string runtime_index) {
  stringstream ss;
  ss << "((" << runtime_index << " == 0) ?";
  ss << "gridDim.x : ";
  ss << "((" << runtime_index << " == 1) ?";
  ss << "gridDim.y : gridDim.z))";
  return ss.str();
}
string CudaDeviceProgram::global_id(uint_tp fixed_index) {
  switch(fixed_index) {
    case 0:
      return "(blockIdx.x * blockDim.x + threadIdx.x)";
    case 1:
      return "(blockIdx.y * blockDim.y + threadIdx.y)";
    case 2:
      return "(blockIdx.z * blockDim.z + threadIdx.z)";
    default:
      return "(blockIdx.x * blockDim.x + threadIdx.x)";
  }
}
string CudaDeviceProgram::global_id(string runtime_index) {
  stringstream ss;
  ss << "((" << runtime_index << " == 0) ?";
  ss << "(blockIdx.x * blockDim.x + threadIdx.x) : ";
  ss << "((" << runtime_index << " == 1) ?";
  ss << "(blockIdx.y * blockDim.y + threadIdx.y) : ";
  ss << "(blockIdx.z * blockDim.z + threadIdx.z)))";
  return ss.str();
}
string CudaDeviceProgram::global_size(uint_tp fixed_index) {
  switch(fixed_index) {
    case 0:
      return "(blockDim.x * gridDim.x)";
    case 1:
      return "(blockDim.y * gridDim.y)";
    case 2:
      return "(blockDim.z * gridDim.z)";
    default:
      return "(blockDim.x * gridDim.x)";
  }
}
string CudaDeviceProgram::global_size(string runtime_index) {
  stringstream ss;
  ss << "((" << runtime_index << " == 0) ?";
  ss << "(blockDim.x * gridDim.x) : ";
  ss << "((" << runtime_index << " == 1) ?";
  ss << "(blockDim.y * gridDim.y) : ";
  ss << "(blockDim.z * gridDim.z)))";
  return ss.str();
}

string CudaDeviceProgram::local_barrier() {
  return "__syncthreads();";
}
string CudaDeviceProgram::global_barrier() {
  return "__threadfence();";
}

string CudaDeviceProgram::atomics() {
  stringstream ss;
  ss << "inline __device__" << std::endl;
  ss << "float caffe_gpu_atomic_float_add(const float val, float* address) {"
     << std::endl;
  ss << "return atomicAdd(address, val);" << std::endl;
  ss << "}" << std::endl;

  // double atomicAdd implementation taken from:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
  ss << "inline __device__" << std::endl;
  ss << "double caffe_gpu_atomic_double_add(const double val, "
     << "double* address) {" << std::endl;
  ss << "unsigned long long int* address_as_ull = "
     << "  reinterpret_cast<unsigned long long int*>(address);" << std::endl;
  ss << "unsigned long long int old = *address_as_ull;" << std::endl;
  ss << "unsigned long long int assumed;" << std::endl;
  ss << "do {" << std::endl;
  ss << "assumed = old;" << std::endl;
  ss << "old = atomicCAS(address_as_ull, assumed, "
     << "__double_as_longlong(val + __longlong_as_double(assumed)));"
     << std::endl;
  ss << "} while (assumed != old);" << std::endl;
  ss << "return __longlong_as_double(old);" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}


inline string pointer_suffix(uint64_t flags) {
  if ((flags & KERNEL_ARG_GLOBAL_MEM) == KERNEL_ARG_GLOBAL_MEM ||
      (flags & KERNEL_ARG_LOCAL_MEM) == KERNEL_ARG_LOCAL_MEM) {
    return "*";
  }
  return "";
}

string CudaDeviceProgram::kernel_arg_type_void(uint64_t flags) {
  return "void" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_bool(uint64_t flags) {
  return "bool" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_char(uint64_t flags) {
  return "char" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_half(uint64_t flags) {
  return "half" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_float(uint64_t flags) {
  return "float" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_double(uint64_t flags) {
  return "double" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_int8_t(uint64_t flags) {
  return "int8_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_int16_t(uint64_t flags) {
  return "int16_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_int32_t(uint64_t flags) {
  return "int32_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_int64_t(uint64_t flags) {
  return "int64_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_uint8_t(uint64_t flags) {
  return "uint8_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_uint16_t(uint64_t flags) {
  return "uint16_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_uint32_t(uint64_t flags) {
  return "uint32_t" + pointer_suffix(flags);
}
string CudaDeviceProgram::kernel_arg_type_uint64_t(uint64_t flags) {
  return "uint64_t" + pointer_suffix(flags);
}

#endif  // USE_CUDA

}

