#ifdef USE_CUDA

#include <cfloat>
#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>

#include "caffe/backend/cuda/cuda_device_program.hpp"
#include "caffe/backend/cuda/cuda_device.hpp"

namespace caffe {

CudaDeviceProgram::CudaDeviceProgram(Device* dev) : DeviceProgram(dev),
    cuda_program_(nullptr), cuda_module_(nullptr) {
  cuda_program_ = make_shared<nvrtcProgram>();
  cuda_module_ = make_shared<CUmodule>();
}

CudaDeviceProgram::~CudaDeviceProgram() {
  if (cuda_program_ != nullptr) {
    nvrtcDestroyProgram(cuda_program_.get());
  }
}

bool CudaDeviceProgram::Compile(bool load_cache, bool store_cache) {
  // CUDA_CHECK(cudaSetDevice(device_->id()));

  bool success = true;

  // Don't compile empty programs with no function declarations
  if (this->args_.size() == 0) {
    return true;
  }

  vector<const char*> build_opts;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, this->device_->id());

  string arch_opt = "--gpu-architecture=compute_"
      + std::to_string(prop.major) + std::to_string(prop.minor);
  // string stdcpp_opt = "--std=c++11";
  string fum_opt = "--use_fast_math";
  string def_dev_opt = "-default-device";

  build_opts.push_back(arch_opt.c_str());
  //build_opts.push_back(stdcpp_opt.c_str());
  if (this->device_->is_fast_unsafe_math()) {
    build_opts.push_back(fum_opt.c_str());
  }
  build_opts.push_back(def_dev_opt.c_str());

  // Pointer holding binary CUDA program
  char* ptx = nullptr;
  size_t ptx_size = 0;
  bool loaded_from_cache = false;
  string flags = this->device_->name() + arch_opt + fum_opt;

#ifdef USE_SQLITE
  if (load_cache) {
    // Try to load the kernel from cache
    int64_t id = this->device_->get_database()->GetKernelInfo(identifier(),
                                               flags.c_str(), flags.size(),
                                               src_.c_str(), src_.size(),
                                               &ptx_size);
    if (id >= 0 && ptx_size > 0) {
      ptx = new char[ptx_size];  // NOLINT
      loaded_from_cache = this->device_->get_database()->LoadKernel(id, ptx);
      if (loaded_from_cache) {
        CUresult result = cuModuleLoadDataEx(cuda_module_.get(), ptx, 0, 0, 0);
        loaded_from_cache = (result == CUDA_SUCCESS);
        success = loaded_from_cache;
        if (!loaded_from_cache) {
          LOG(WARNING) << "Failed to load CUDA binary ("
                       << this->string_identifier() << ") from cache ("
                       << cudaGetErrorString(result) << ")" << std::endl;
        }
      }
      delete ptx;  // NOLINT
      ptx = nullptr;
    }
  }
#endif  // USE_SQLITE

  if (!loaded_from_cache) {
    nvrtcCreateProgram(cuda_program_.get(), src_.c_str(), NULL,
                  static_cast<CudaDevice*>(this->device_)->get_header_count(),
                  static_cast<CudaDevice*>(this->device_)->get_header_sources(),
                  static_cast<CudaDevice*>(this->device_)->get_header_names());

    nvrtcCompileProgram(*cuda_program_.get(), build_opts.size(),
                        &build_opts[0]);

    nvrtcGetPTXSize(*cuda_program_.get(), &ptx_size);
    ptx = new char[ptx_size];
    nvrtcGetPTX(*cuda_program_.get(), ptx);
#ifndef NDEBUG
    string debug_path = ".caffe_debug";
    const char* path = debug_path.c_str();
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(dir);
    {
      FILE* fp = fopen((".caffe_debug/"
          + string_identifier() + ".cu").c_str(), "wb");
      fwrite(this->src_.c_str(), sizeof(char), this->src_.size(), fp);
      fclose(fp);
    }
    {
      size_t log_size;
      nvrtcGetProgramLogSize(*cuda_program_.get(), &log_size);
      vector<char> log(log_size);
      nvrtcGetProgramLog(*cuda_program_.get(), log.data());
      std::cout << "CUDA compile log:" << std::endl;
      std::cout << log.data() << std::endl;
      FILE* fp = fopen((".caffe_debug/"
          + string_identifier() + ".cuptx").c_str(), "wb");
      fwrite(ptx, sizeof(char), ptx_size, fp);
      fclose(fp);
    }
#endif  // NDEBUG
    CUresult result = cuModuleLoadDataEx(cuda_module_.get(), ptx, 0, 0, 0);
    if (!(result == CUDA_SUCCESS)) {
      LOG(ERROR) << "Failed to compile CUDA binary ("
                 << this->string_identifier() << ") from code ("
                 << cudaGetErrorString(result) << ")" << std::endl;
    }
#ifdef USE_SQLITE
    if (store_cache && (result == CUDA_SUCCESS)) {
      this->device_->get_database()->StoreKernel(identifier(),
                                                flags.c_str(), flags.size(),
                                                src_.c_str(), src_.size(),
                                                ptx, ptx_size);
    }
#endif  // USE_SQLITE
    delete ptx;  // NOLINT
  }
  return success;
}


shared_ptr<DeviceKernel> CudaDeviceProgram::GetKernel(string name) {
  shared_ptr<CUfunction> kernel = make_shared<CUfunction>();
  CHECK(cuda_module_.get()) << "CUDA module invalid.";
  CUresult result = cuModuleGetFunction(kernel.get(), *cuda_module_.get(),
                                        name.c_str());
  if (result != CUDA_SUCCESS) {
    LOG(FATAL) << "Loading CUDA kernel " << name << " ("
               << this->string_identifier() << ") failed ("
               << cudaGetErrorString(result) << ")" << std::endl;
  }
  CHECK(kernel.get()) << "Loading CUDA kernel " << name << " failed.";

  KernelArgs args;

  std::map<string, KernelArgs>::iterator pos = this->args_.find(name);
  if (pos == this->args_.end()) {
    LOG(FATAL) << "CUDA kernel " << name << " not found";
  } else {
    args = pos->second;
  }

  return make_shared<CudaDeviceKernel>(device_, name, kernel, args);
}

string CudaDeviceProgram::function(string name, KernelArgs args,
                                   KernelHints hints) {
  args_.insert(make_pair(name, args));
  stringstream ss;
  ss << " extern \"C\" ";
  ss << "__global__ void " << std::endl;

  int32_t max_threads_per_block = 0;
  int32_t min_blocks_per_mp = 0;
  for (uint_tp i = 0; i < hints.size(); ++i) {
    switch(std::get<0>(hints[i])) {
      case KERNEL_REQD_WORK_GROUP_X:
        max_threads_per_block = std::max(max_threads_per_block *
                                         std::stoi(std::get<1>(hints[i])),
                                         std::stoi(std::get<1>(hints[i])));
        break;
      case KERNEL_REQD_WORK_GROUP_Y:
        max_threads_per_block = std::max(max_threads_per_block *
                                         std::stoi(std::get<1>(hints[i])),
                                         std::stoi(std::get<1>(hints[i])));
        break;
      case KERNEL_REQD_WORK_GROUP_Z:
        max_threads_per_block = std::max(max_threads_per_block *
                                         std::stoi(std::get<1>(hints[i])),
                                         std::stoi(std::get<1>(hints[i])));
        break;
      case KERNEL_HINT_WORK_GROUP_X:
        max_threads_per_block = std::max(max_threads_per_block *
                                         std::stoi(std::get<1>(hints[i])),
                                         std::stoi(std::get<1>(hints[i])));
        break;
      case KERNEL_HINT_WORK_GROUP_Y:
        max_threads_per_block = std::max(max_threads_per_block *
                                         std::stoi(std::get<1>(hints[i])),
                                         std::stoi(std::get<1>(hints[i])));
        break;
      case KERNEL_HINT_WORK_GROUP_Z:
        max_threads_per_block = std::max(max_threads_per_block *
                                         std::stoi(std::get<1>(hints[i])),
                                         std::stoi(std::get<1>(hints[i])));
        break;
      case KERNEL_HINT_MIN_BLOCKS_PER_MP:
        min_blocks_per_mp = std::stoi(std::get<1>(hints[i]));
        break;
    }
  }

  if (max_threads_per_block > 0 and min_blocks_per_mp > 0) {
    ss << "__launch_bounds__(" << max_threads_per_block << ","
                               << min_blocks_per_mp << ")" << std::endl;
  }

  ss << name << "(";
  for (uint_tp i = 0; i < args.size(); ++i) {
    uint64_t flags = std::get<2>(args[i]);
    if ((flags & KERNEL_ARG_CONST) == KERNEL_ARG_CONST) {
      ss << "const ";
    }
    ss << std::get<0>(args[i]) << " ";
    if ((flags & KERNEL_ARG_RESTRICT) == KERNEL_ARG_RESTRICT) {
      ss << "__restrict__ ";
    }
    ss << std::get<1>(args[i]);
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
     << index << " < (" << n << "); "
     << index << " += blockDim.x * gridDim.x) {" << std::endl;
  return ss.str();
}

string CudaDeviceProgram::setup() {
  stringstream ss;

#ifdef USE_HALF
  ss << "#include \"cuda_fp16.h\"" << std::endl;
  ss << "#include \"cuda_fp16.hpp\"" << std::endl;

  ss << "#define HALF_SUPPORT_AVAILABLE" << std::endl;
  ss << "#define HALF_MAX " << HALF_MAX << std::endl;
  ss << "#define HALF_MIN " << HALF_MIN << std::endl;
#endif  // USE_HALF

  ss << "#define FLT_MAX " << FLT_MAX << std::endl;
  ss << "#define FLT_MIN " << FLT_MIN << std::endl;

  ss << "#define int8_t char" << std::endl;
  ss << "#define int16_t short" << std::endl;
  ss << "#define int32_t int" << std::endl;
  ss << "#define int64_t long long" << std::endl;
  ss << "#define uint8_t unsigned char" << std::endl;
  ss << "#define uint16_t unsigned short" << std::endl;
  ss << "#define uint32_t unsigned int" << std::endl;
  ss << "#define uint64_t unsigned long long" << std::endl;
  ss << "#define uchar unsigned char" << std::endl;
  ss << "#define ushort unsigned short" << std::endl;
  ss << "#define uint unsigned int" << std::endl;
  ss << "#define ulonglong unsigned long long" << std::endl;

  ss << this->define_type<int_tp>("int_tp");
  ss << this->define_type<uint_tp>("uint_tp");
#ifdef USE_INDEX_64
  ss << "#ifdef " << "int_tpc" << std::endl;
  ss << "#undef " << "int_tpc" << std::endl;
  ss << "#endif  //" << "int_tpc" << std::endl;
  ss << "#define " << "int_tpc" << " long long" << std::endl;
  ss << "#ifdef " << "uint_tpc" << std::endl;
  ss << "#undef " << "uint_tpc" << std::endl;
  ss << "#endif  //" << "uint_tpc" << std::endl;
  ss << "#define " << "uint_tpc" << " unsigned long long" << std::endl;
#else  // USE_INDEX_64
  ss << "#ifdef " << "int_tpc" << std::endl;
  ss << "#undef " << "int_tpc" << std::endl;
  ss << "#endif  //" << "int_tpc" << std::endl;
  ss << "#define " << "int_tpc" << " int" << std::endl;
  ss << "#ifdef " << "uint_tpc" << std::endl;
  ss << "#undef " << "uint_tpc" << std::endl;
  ss << "#endif  //" << "uint_tpc" << std::endl;
  ss << "#define " << "uint_tpc" << " unsigned int" << std::endl;
#endif  // USE_INDEX_64

#ifdef USE_HALF
  // Add support for wider FP16 vectors in CUDA
  ss << "typedef struct __align__(8) {" << std::endl;
  ss << "half x, y, z, w;" << std::endl;
  ss << "} half4;" << std::endl;

  ss << "typedef struct __align__(16) {" << std::endl;
  ss << "half s0, s1, s2, s3, s4, s5, s6, s7, s8;" << std::endl;
  ss << "} half8;" << std::endl;

  ss << "typedef struct __align__(32) {" << std::endl;
  ss << "half s0, s1, s2, s3, s4, s5, s6, s7, s8, "
     << "s9, sA, sB, sC, sD, sE, sF;" << std::endl;
  ss << "} half16;" << std::endl;
#endif  // USE_HALF

  return ss.str();
}

string CudaDeviceProgram::global_ptr(string type, string name) {
  return type + "* " + name;
}

string CudaDeviceProgram::local_ptr(string type, string name) {
  return type + "* " + name;
}

string CudaDeviceProgram::local_mem(string type, string name) {
  return "__shared__ " + type + " " + name;
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
  ss << "float caffe_gpu_atomic_float_add(float* address, const float val) {"
     << std::endl;
  ss << "return atomicAdd(address, val);" << std::endl;
  ss << "}" << std::endl;

  // double atomicAdd implementation taken from:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
  ss << "inline __device__" << std::endl;
  ss << "double caffe_gpu_atomic_double_add(double* address, "
     << "const double val) {" << std::endl;
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

string CudaDeviceProgram::device_type_name_void() const {
  return "void";
}
string CudaDeviceProgram::device_type_name_bool() const {
  return "bool";
}
string CudaDeviceProgram::device_type_name_char() const {
  return "char";
}
string CudaDeviceProgram::device_type_name_half() const {
  return "half";
}
string CudaDeviceProgram::device_type_name_float() const {
  return "float";
}
string CudaDeviceProgram::device_type_name_double() const {
  return "double";
}
string CudaDeviceProgram::device_type_name_int8() const {
  return "char";
}
string CudaDeviceProgram::device_type_name_int16() const {
  return "short";
}
string CudaDeviceProgram::device_type_name_int32() const {
  return "int";
}
string CudaDeviceProgram::device_type_name_int64() const {
  return "longlong";
}
string CudaDeviceProgram::device_type_name_uint8() const {
  return "uchar";
}
string CudaDeviceProgram::device_type_name_uint16() const {
  return "ushort";
}
string CudaDeviceProgram::device_type_name_uint32() const {
  return "uint";
}
string CudaDeviceProgram::device_type_name_uint64() const {
  return "ulonglong";
}

string CudaDeviceProgram::convert_type_char(int_tp vec_len,
                                            string src_val) const {
  return "((char" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_half(int_tp vec_len,
                                            string src_val) const {
  return "((half" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_float(int_tp vec_len,
                                             string src_val) const {
  return "((float" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_double(int_tp vec_len,
                                              string src_val) const {
  return "((double" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_uint8(int_tp vec_len,
                                            string src_val) const {
  return "((uchar" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_uint16(int_tp vec_len,
                                             string src_val) const {
  return "((ushort" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_uint32(int_tp vec_len,
                                             string src_val) const {
  return "((uint" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_uint64(int_tp vec_len,
                                             string src_val) const {
  return "((ulong" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_int8(int_tp vec_len,
                                            string src_val) const {
  return "((char" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_int16(int_tp vec_len,
                                             string src_val) const {
  return "((short" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_int32(int_tp vec_len,
                                             string src_val) const {
  return "((int" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}
string CudaDeviceProgram::convert_type_int64(int_tp vec_len,
                                             string src_val) const {
  return "((long" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + ")(" + src_val + "))";
}

string CudaDeviceProgram::helper_functions_half() const {
  stringstream ss;
  ss << "#ifndef CAFFE_CUDA_HELPER_HALF" << std::endl;
  ss << "#define CAFFE_CUDA_HELPER_HALF" << std::endl;
  ss << "__device__ half abs(half x) {" << std::endl;
  ss << "return (half)abs((float)x);;" << std::endl;
  ss << "}" << std::endl;
  ss << "__device__ half pow(half x, half y) {" << std::endl;
  ss << "return (half)pow((float)x, (float)y);";
  ss << "}" << std::endl;
  ss << "__device__ half signbit(half x) {" << std::endl;
  ss << "return (half)signbit((float)x);";
  ss << "}" << std::endl;
  ss << "__device__ half max(half x, half y) {" << std::endl;
  ss << "return x > y ? x : y;" << std::endl;
  ss << "}" << std::endl;
  ss << "__device__ half min(half x, half y) {" << std::endl;
  ss << "return x < y ? x : y;" << std::endl;
  ss << "}" << std::endl;
  ss << "#endif  // CAFFE_CUDA_HELPER_HALF" << std::endl;

  return ss.str();
}
string CudaDeviceProgram::helper_functions_float() const {
  return "";
}
string CudaDeviceProgram::helper_functions_double() const {
  return "";
}
string CudaDeviceProgram::helper_functions_uint8() const {
  return "";
}
string CudaDeviceProgram::helper_functions_uint16() const {
  return "";
}
string CudaDeviceProgram::helper_functions_uint32() const {
  return "";
}
string CudaDeviceProgram::helper_functions_uint64() const {
  return "";
}


}  // namespace caffe

#endif  // USE_CUDA
