#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

#include "caffe/backend/opencl/ocl_device_program.hpp"
#include "caffe/backend/backend.hpp"
#include "caffe/backend/device.hpp"

namespace caffe {

#ifdef USE_OPENCL

OclDeviceProgram::OclDeviceProgram(Device *dev) : DeviceProgram(dev) {
}

bool OclDeviceProgram::Compile(bool load_cache, bool store_cache) {
  // Don't compile empty programs with no function declarations
  if (this->args_.size() == 0) {
    return true;
  }

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(
      this->device_->id());
  cl_int err = 0;

  string build_opts = "";
  build_opts += "-cl-mad-enable ";
  build_opts += "-cl-single-precision-constant ";

  bool loaded_from_cache = false;
  string flags = this->device_->name() + build_opts;
  ctx.build_options(build_opts);

  vector<cl_device_id> device_ids(ctx.device_num());
  clGetContextInfo(ctx.handle().get(), CL_CONTEXT_DEVICES,
                   ctx.device_num() * sizeof(cl_device_id),
                   &(device_ids[0]), nullptr);

#ifdef USE_SQLITE
  // Try to load the kernel from cache
  if (load_cache) {
    size_t ptx_size = 0;
    int64_t id = this->device_->get_database()->GetKernelInfo(identifier(),
                                               flags.c_str(), flags.size(),
                                               src_.c_str(), src_.size(),
                                               &ptx_size);
    if (id >= 0 && ptx_size > 0) {
      vector<char> ptx = vector<char>(ptx_size);  // NOLINT
      loaded_from_cache = this->device_->get_database()->LoadKernel(id,
                                                                    &(ptx[0]));
      if (loaded_from_cache) {
        vector<cl_int> status(ctx.device_num());
        const unsigned char* ptx_ptr =
            reinterpret_cast<unsigned char*>(&(ptx[0]));
        cl_program cached_program = clCreateProgramWithBinary(
            ctx.handle().get(), ctx.device_num(), &(device_ids[0]), &ptx_size,
            &ptx_ptr, &(status[0]), &err);
        for (size_t i = 0; i < ctx.device_num(); ++i) {
          loaded_from_cache = loaded_from_cache && (status[i] == CL_SUCCESS);
        }
        if (err != CL_SUCCESS || !loaded_from_cache) {
          LOG(WARNING) << "Failed to load OpenCL binary ("
                       << this->string_identifier() << ") from cache ("
                       << clGetErrorString(err) << ")" << std::endl;
          loaded_from_cache = false;
        } else {
          err = clBuildProgram(cached_program, ctx.device_num(),
                               &(device_ids[0]), ctx.build_options().c_str(),
                               nullptr, nullptr);
          if (err != CL_SUCCESS || !loaded_from_cache) {
            LOG(WARNING) << "Failed to load OpenCL binary ("
                         << this->string_identifier() << ") from cache ("
                         << clGetErrorString(err) << ")" << std::endl;
            loaded_from_cache = false;
          }
        }
        if (loaded_from_cache) {
          ocl_program_ = ctx.add_program(cached_program, string_identifier());
        }
      }
    }
  }
#endif  // USE_SQLITE

  // Compile from source
  if (!loaded_from_cache) {
#ifndef NDEBUG
    string debug_path = ".caffe_debug";
    const char* path = debug_path.c_str();
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(dir);
    {
      FILE* fp = fopen((".caffe_debug/" + string_identifier() + ".cl").c_str(),
                       "wb");
      fwrite(this->src_.c_str(), sizeof(char), this->src_.size(), fp);
      fclose(fp);
    }
#endif  // NDEBUG

    size_t src_size = src_.size();
    const char* src_ptr = src_.c_str();

    cl_program compiled_program = clCreateProgramWithSource(ctx.handle().get(),
                                       1, &src_ptr, &src_size, &err);
    if (err != CL_SUCCESS) {
      LOG(ERROR) << "Failed to compile OpenCL binary ("
                 << this->string_identifier() << ") from code ("
                 << clGetErrorString(err) << ")" << std::endl;
    }
    err = clBuildProgram(compiled_program, ctx.device_num(),
                         &(device_ids[0]), ctx.build_options().c_str(),
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t len;
      clGetProgramBuildInfo(compiled_program, device_ids[0],
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
      std::vector<char> log_chars(len);
      clGetProgramBuildInfo(compiled_program, device_ids[0],
                            CL_PROGRAM_BUILD_LOG, len, &log_chars[0], NULL);
      std::string build_log(&log_chars[0]);

      LOG(ERROR) << "Failed to compile OpenCL binary ("
                 << this->string_identifier() << ") from code ("
                 << clGetErrorString(err) << ")" << std::endl
                 << build_log;
    }
    ocl_program_ = ctx.add_program(compiled_program, string_identifier());
  }

  // Add kernels to the program
  vector<cl_kernel> kernels(1024);
  cl_uint num_kernels;
  err = clCreateKernelsInProgram(ocl_program_.handle().get(),
                                 1024, &(kernels[0]), &num_kernels);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to load OpenCL kernels ("
               << this->string_identifier() << ") ("
               << clGetErrorString(err) << ")" << std::endl;
  } else {
    for (cl_uint i = 0; i < num_kernels; ++i) {
      vector<char> kernel_name(128);
      err = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 128,
                            &(kernel_name[0]), NULL);
      ocl_program_.add_kernel(kernels[i], string(&(kernel_name[0])));
    }
  }

  // Storing compiled kernels not loaded from the cache
  if (!loaded_from_cache) {
    size_t len;
    vector<size_t> ptx_sizes(ctx.devices().size(), 0);
    vector<char*> ptxs;

    if (num_kernels > 0) {
      err = clGetProgramInfo(ocl_program_.handle().get(),
                       CL_PROGRAM_BINARY_SIZES, 0, NULL, &len);
      if (len > 0 && (err == CL_SUCCESS)) {
        clGetProgramInfo(ocl_program_.handle().get(),
                         CL_PROGRAM_BINARY_SIZES, len, &(ptx_sizes[0]), NULL);
      }

      for (size_t i = 0; i < ctx.devices().size(); ++i) {
        ptxs.push_back(new char[ptx_sizes[i]]);  // NOLINT
      }
      err = clGetProgramInfo(ocl_program_.handle().get(),
                       CL_PROGRAM_BINARIES, 0, nullptr, &len);
      if (len > 0 && (err == CL_SUCCESS)) {
        clGetProgramInfo(ocl_program_.handle().get(),
                         CL_PROGRAM_BINARIES, len, &(ptxs[0]), nullptr);
      }
    }
#ifndef NDEBUG
    {
      FILE* fp = fopen((".caffe_debug/" + string_identifier()
                        + ".clptx").c_str(), "wb");
      if (ptxs.size() > 0) {
        fwrite(ptxs[0], sizeof(char), ptx_sizes[0], fp);
      }
      fclose(fp);
    }
#endif  // NDEBUG
#ifdef USE_SQLITE
    if (store_cache) {
      this->device_->get_database()->StoreKernel(identifier(),
                         flags.c_str(), flags.size(), src_.c_str(), src_.size(),
                         ptxs.size() > 0 ? ptxs[0] : nullptr,
                         ptx_sizes[0]);
    }
#endif  // USE_SQLITE
    for (size_t i = 0; i < ctx.devices().size(); ++i) {
      if (ptxs.size() > i) {
        delete[] ptxs[i];
      }
    }
  }

  return true;
}

shared_ptr<DeviceKernel> OclDeviceProgram::GetKernel(string name) {
  viennacl::ocl::kernel &kernel = this->ocl_program_.get_kernel(name);

  KernelArgs args;
  std::map<string, KernelArgs>::iterator pos = this->args_.find(name);
  if (pos == this->args_.end()) {
    LOG(FATAL) << "OpenCL kernel " << name << " ("
               << this->string_identifier() << ") not found";
  } else {
    args = pos->second;
  }

  /*
  for(int_tp i = 0; i < args.size(); ++i) {
    std::cout << i << " flags: " << std::get<0>(args[i]) << std::endl;
    std::cout << i << " flags: " << std::get<1>(args[i]) << std::endl;
    std::cout << i << " flags: " << std::get<2>(args[i]) << std::endl;
  }
  */

  return make_shared<OclDeviceKernel>(device_, name, kernel, args);
}

string OclDeviceProgram::function(string name, KernelArgs args,
                                  KernelHints hints) {
  args_.insert(make_pair(name, args));
  stringstream ss;
  ss << "__kernel" << std::endl;

  vector<int> reqd_work_group(3, 0);
  vector<int> hint_work_group(3, 0);

  for (uint_tp i = 0; i < hints.size(); ++i) {
    switch(std::get<0>(hints[i])) {
      case KERNEL_HINT_VEC_TYPE:
        ss << "__attribute__((vec_type_hint(" << std::get<1>(hints[i])
           << ")))" << std::endl;
        break;
      case KERNEL_REQD_WORK_GROUP_X:
        reqd_work_group[0] = std::stoi(std::get<1>(hints[i]));
        break;
      case KERNEL_REQD_WORK_GROUP_Y:
        reqd_work_group[1] = std::stoi(std::get<1>(hints[i]));
        break;
      case KERNEL_REQD_WORK_GROUP_Z:
        reqd_work_group[2] = std::stoi(std::get<1>(hints[i]));
        break;
      case KERNEL_HINT_WORK_GROUP_X:
        hint_work_group[0] = std::stoi(std::get<1>(hints[i]));
        break;
      case KERNEL_HINT_WORK_GROUP_Y:
        hint_work_group[1] = std::stoi(std::get<1>(hints[i]));
        break;
      case KERNEL_HINT_WORK_GROUP_Z:
        hint_work_group[2] = std::stoi(std::get<1>(hints[i]));
        break;
    }
  }

  if (reqd_work_group[0] > 0 &&
      reqd_work_group[1] > 0 &&
      reqd_work_group[2] > 0) {
    ss << "__attribute__((reqd_work_group_size("
       << reqd_work_group[0] << ","
       << reqd_work_group[1] << ","
       << reqd_work_group[2] << ")))" << std::endl;
  }

  if (hint_work_group[0] > 0 &&
      hint_work_group[1] > 0 &&
      hint_work_group[2] > 0) {
    ss << "__attribute__((work_group_size_hint("
       << hint_work_group[0] << ","
       << hint_work_group[1] << ","
       << hint_work_group[2] << ")))" << std::endl;
  }

  ss << "void " << name << "(";
  for (uint_tp i = 0; i < args.size(); ++i) {
    uint64_t flags = std::get<2>(args[i]);
    if ((flags & KERNEL_ARG_GLOBAL_MEM) == KERNEL_ARG_GLOBAL_MEM) {
      ss << "__global ";
    }
    if ((flags & KERNEL_ARG_LOCAL_MEM) == KERNEL_ARG_LOCAL_MEM) {
      ss << "__local ";
    }
    if ((flags & KERNEL_ARG_CONST) == KERNEL_ARG_CONST) {
      ss << "const ";
    }
    string var_name = std::get<1>(args[i]);
    string off_name = std::get<1>(args[i]);
    if ((flags & KERNEL_ARG_MEM_OFFSET) == KERNEL_ARG_MEM_OFFSET) {
      var_name += "_raw_ptr";
      off_name += "_offset";
    }
    ss << std::get<0>(args[i]) << " ";
    if ((flags & KERNEL_ARG_RESTRICT) == KERNEL_ARG_RESTRICT) {
      ss << "__restrict ";
    }
    ss << var_name;
    if ((flags & KERNEL_ARG_MEM_OFFSET) == KERNEL_ARG_MEM_OFFSET) {
      ss << ", const uint_tp " << off_name;
    }
    if (i < args.size() - 1) {
      ss << ", ";
    }
  }
  ss << ") {" << std::endl;
  for (uint_tp i = 0; i < args.size(); ++i) {
    uint64_t flags = std::get<2>(args[i]);
    if ((flags & KERNEL_ARG_MEM_OFFSET) == KERNEL_ARG_MEM_OFFSET) {
      string base_name = std::get<1>(args[i]);
      string var_name = base_name + "_raw_ptr";
      string off_name = base_name + "_offset";
      if ((flags & KERNEL_ARG_GLOBAL_MEM) == KERNEL_ARG_GLOBAL_MEM) {
        ss << "__global ";
      }
      if ((flags & KERNEL_ARG_LOCAL_MEM) == KERNEL_ARG_LOCAL_MEM) {
        ss << "__local ";
      }
      if ((flags & KERNEL_ARG_CONST) == KERNEL_ARG_CONST) {
        ss << "const ";
      }
      ss << std::get<0>(args[i]) << " " << base_name;
      ss << " = " << var_name << " + " << off_name << ";";
      ss << std::endl;
    }
  }
  return ss.str();
}

string OclDeviceProgram::kernel_loop(string type,
                                            string index, string n) {
  stringstream ss;
  ss << "for (" << type << " "
     << index << " = get_global_id(0); "
     << index << " < (" << n << "); "
     << index << " += get_global_size(0)) {" << std::endl;
  return ss.str();
}

string OclDeviceProgram::setup() {
  stringstream ss;
  ss << "#define int8_t char" << std::endl;
  ss << "#define int16_t short" << std::endl;
  ss << "#define int32_t int" << std::endl;
  ss << "#define int64_t long" << std::endl;
  ss << "#define uint8_t uchar" << std::endl;
  ss << "#define uint16_t ushort" << std::endl;
  ss << "#define uint32_t uint" << std::endl;
  ss << "#define uint64_t ulong" << std::endl;

  // Test/enable KHR 64 bit (double)
  ss << "#if defined(cl_khr_fp64)" << std::endl;
  ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
  ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;

  // Test/enable AMD 64 bit (double)
  ss << "#elif defined(cl_amd_fp64)" << std::endl;
  ss << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable" << std::endl;
  ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;
  ss << "#endif" << std::endl;

  // Test/enable KHR 16 bit (half)
  ss << "#if defined(cl_khr_fp16)" << std::endl;
  ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
  ss << "#define HALF_SUPPORT_AVAILABLE" << std::endl;
  // Provide half max/min if not defined
  ss << "#ifndef HALF_MAX" << std::endl;
  ss << "#define HALF_MAX 65504.f" << std::endl;
  ss << "#endif" << std::endl;
  ss << "#ifndef HALF_MIN" << std::endl;
  ss << "#define HALF_MIN 6.10352e-5f" << std::endl;
  ss << "#endif" << std::endl;
  ss << "#endif" << std::endl;

  ss << this->define_type<int_tp>("int_tp");
  ss << this->define_type<uint_tp>("uint_tp");
  ss << this->define_type<int_tp>("int_tpc");
  ss << this->define_type<uint_tp>("uint_tpc");
  return ss.str();
}

string OclDeviceProgram::global_ptr(string type, string name) {
  return "__global " + type + "* " + name;
}

string OclDeviceProgram::local_ptr(string type, string name) {
  return "__local " + type + "* " + name;
}

string OclDeviceProgram::local_mem(string type, string name) {
  return "__local " + type + " " + name;
}

string OclDeviceProgram::local_id(uint_tp fixed_index) {
  return "get_local_id(" + std::to_string(fixed_index) + ")";
}
string OclDeviceProgram::local_id(string runtime_index) {
  return "get_local_id(" + runtime_index + ")";
}
string OclDeviceProgram::local_size(uint_tp fixed_index) {
  return "get_local_size(" + std::to_string(fixed_index) + ")";
}
string OclDeviceProgram::local_size(string runtime_index) {
  return "get_local_size(" + runtime_index + ")";
}
string OclDeviceProgram::group_id(uint_tp fixed_index) {
  return "get_group_id(" + std::to_string(fixed_index) + ")";
}
string OclDeviceProgram::group_id(string runtime_index) {
  return "get_group_id(" + runtime_index + ")";
}
string OclDeviceProgram::group_size(uint_tp fixed_index) {
  return "get_num_groups("+ std::to_string(fixed_index) + ")";
}
string OclDeviceProgram::group_size(string runtime_index) {
  return "get_num_groups(" + runtime_index + ")";
}
string OclDeviceProgram::global_id(uint_tp fixed_index) {
  return "get_global_id(" + std::to_string(fixed_index) + ")";
}
string OclDeviceProgram::global_id(string runtime_index) {
  return "get_global_id(" + runtime_index + ")";
}
string OclDeviceProgram::global_size(uint_tp fixed_index) {
  return "get_global_size(" + std::to_string(fixed_index) + ")";
}
string OclDeviceProgram::global_size(string runtime_index) {
  return "get_global_size(" + runtime_index + ")";
}

string OclDeviceProgram::local_barrier() {
  return "barrier(CLK_LOCAL_MEM_FENCE);";
}

string OclDeviceProgram::global_barrier() {
  return "barrier(CLK_GLOBAL_MEM_FENCE);";
}

string OclDeviceProgram::atomics() {
  stringstream ss;

  // 32 bit atomics
  ss << "#if defined(cl_khr_int32_base_atomics)" << std::endl;
  ss << "#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable"
     << std::endl;
  ss << "#define ATOMICS_32_AVAILABLE" << std::endl;
  ss << "#endif" << std::endl;
  ss << "#if defined(cl_khr_global_int32_base_atomics)" << std::endl;
  ss << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable"
     << std::endl;
  ss << "#define ATOMICS_32_AVAILABLE" << std::endl;
  ss << "#endif" << std::endl;

  // 64 bit atomics
  ss << "#if defined(cl_khr_int64_base_atomics)" << std::endl;
  ss << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
     << std::endl;
  ss << "#define ATOMICS_64_AVAILABLE" << std::endl;
  ss << "#endif" << std::endl;

  vector<string> atomic_datatypes({"half", "float", "double"});
  vector<string> atomic_funcs({ "add", "sub", "mul", "div" });
  vector<string> atomic_ops({ "+", "-", "*", "/" });

  // OpenCL atomics, derived from:
  // https://streamcomputing.eu/blog/2016-02-09/
  // atomic-operations-for-floats-in-opencl-improved/
  for (int j = 0; j < atomic_datatypes.size(); ++j) {
    string atomic_datatype = atomic_datatypes[j];
    if (atomic_datatype == "float") {
      ss << "#if defined(ATOMICS_32_AVAILABLE)" << std::endl;
    } else if (atomic_datatype == "half") {
      ss << "#if defined(ATOMICS_32_AVAILABLE)"
         << " && defined(HALF_SUPPORT_AVAILABLE)" << std::endl;
    } else if (atomic_datatype == "double") {
      ss << "#if defined(ATOMICS_64_AVAILABLE)"
         << " && defined(DOUBLE_SUPPORT_AVAILABLE)" << std::endl;
    }
    for (int i = 0; i < atomic_funcs.size(); ++i) {
      ss << "inline void caffe_gpu_atomic_"
         << atomic_datatype << "_" << atomic_funcs[i];
      ss << "(volatile __global " << atomic_datatype << "* source, const "
         << atomic_datatype << " operand) {"
         << std::endl;
      ss << "union {" << std::endl;
      if (atomic_datatype == "double") {
        ss << "unsigned long intVal;" << std::endl;
      } else {
        ss << "unsigned int intVal;" << std::endl;
      }
      if (atomic_datatype == "half") {
        ss << atomic_datatype << " floatVal[2];" << std::endl;
      } else {
        ss << atomic_datatype << " floatVal[1];" << std::endl;
      }
      ss << "} next, expected, current;" << std::endl;
      ss << "current.floatVal[0] = *source;" << std::endl;
      if (atomic_datatype == "half") {
        ss << "current.floatVal[1] = *(source + 1);" << std::endl;
      }
      ss << "do {" << std::endl;
      ss << "expected.intVal = current.intVal;" << std::endl;
      ss << "next.floatVal[0] = expected.floatVal[0] "
         << atomic_ops[i] << " operand;" << std::endl;
      if (atomic_datatype == "half") {
        ss << "next.floatVal[1] = expected.floatVal[1]; " << std::endl;
      }
      ss << "current.intVal = ";
      if (atomic_datatype == "double") {
        ss << "atom_cmpxchg((volatile __global unsigned long *)";
      } else {
        ss << "atomic_cmpxchg((volatile __global unsigned int *)";
      }
      ss << "source, expected.intVal, next.intVal);" << std::endl;
      ss << "} while (current.intVal != expected.intVal);" << std::endl;
      ss << "}" << std::endl;
    }
    ss << "#endif" << std::endl;
  }
  return ss.str();
}

inline string pointer_suffix(uint64_t flags) {
  if ((flags & KERNEL_ARG_GLOBAL_MEM) == KERNEL_ARG_GLOBAL_MEM ||
      (flags & KERNEL_ARG_LOCAL_MEM) == KERNEL_ARG_LOCAL_MEM) {
    return "*";
  }
  return "";
}

string OclDeviceProgram::kernel_arg_type_void(uint64_t flags) {
  return "void" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_bool(uint64_t flags) {
  if ((flags & KERNEL_ARG_GLOBAL_MEM) == KERNEL_ARG_GLOBAL_MEM ||
      (flags & KERNEL_ARG_LOCAL_MEM) == KERNEL_ARG_LOCAL_MEM) {
    return "bool" + pointer_suffix(flags);
  } else {
    // OpenCL kernel arguments are not supporting booleans, convert to int8_t
    return "int8_t";
  }
}
string OclDeviceProgram::kernel_arg_type_char(uint64_t flags) {
  return "char" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_half(uint64_t flags) {
  if ((flags & KERNEL_ARG_GLOBAL_MEM) == KERNEL_ARG_GLOBAL_MEM ||
      (flags & KERNEL_ARG_LOCAL_MEM) == KERNEL_ARG_LOCAL_MEM) {
    return "half" + pointer_suffix(flags);
  } else {
    // OpenCL kernel arguments are not supporting halfs, convert to float
    return "float";
  }
}
string OclDeviceProgram::kernel_arg_type_float(uint64_t flags) {
  return "float" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_double(uint64_t flags) {
  return "double" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_int8_t(uint64_t flags) {
  return "int8_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_int16_t(uint64_t flags) {
  return "int16_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_int32_t(uint64_t flags) {
  return "int32_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_int64_t(uint64_t flags) {
  return "int64_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_uint8_t(uint64_t flags) {
  return "uint8_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_uint16_t(uint64_t flags) {
  return "uint16_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_uint32_t(uint64_t flags) {
  return "uint32_t" + pointer_suffix(flags);
}
string OclDeviceProgram::kernel_arg_type_uint64_t(uint64_t flags) {
  return "uint64_t" + pointer_suffix(flags);
}

string OclDeviceProgram::device_type_name_void() const {
  return "void";
}
string OclDeviceProgram::device_type_name_bool() const {
  return "bool";
}
string OclDeviceProgram::device_type_name_char() const {
  return "char";
}
string OclDeviceProgram::device_type_name_half() const {
  return "half";
}
string OclDeviceProgram::device_type_name_float() const {
  return "float";
}
string OclDeviceProgram::device_type_name_double() const {
  return "double";
}
string OclDeviceProgram::device_type_name_int8() const {
  return "char";
}
string OclDeviceProgram::device_type_name_int16() const {
  return "short";
}
string OclDeviceProgram::device_type_name_int32() const {
  return "int";
}
string OclDeviceProgram::device_type_name_int64() const {
  return "long";
}
string OclDeviceProgram::device_type_name_uint8() const {
  return "uchar";
}
string OclDeviceProgram::device_type_name_uint16() const {
  return "ushort";
}
string OclDeviceProgram::device_type_name_uint32() const {
  return "uint";
}
string OclDeviceProgram::device_type_name_uint64() const {
  return "ulong";
}

string OclDeviceProgram::convert_type_char(int_tp vec_len,
                                           string src_val) const {
  return "convert_char" + (vec_len > 0 ? std::to_string(vec_len) : "")
      + "_sat(" + src_val + ")";
}
string OclDeviceProgram::convert_type_half(int_tp vec_len,
                                           string src_val) const {
  return "convert_half" + (vec_len > 0 ? std::to_string(vec_len) : "")
      + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_float(int_tp vec_len,
                                            string src_val) const {
  return "convert_float" + (vec_len > 0 ? std::to_string(vec_len) : "")
      + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_double(int_tp vec_len,
                                             string src_val) const {
  return "convert_double" + (vec_len > 0 ? std::to_string(vec_len) : "")
      + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_uint8(int_tp vec_len,
                                            string src_val) const {
  return "convert_uchar" + (vec_len > 0 ? std::to_string(vec_len) : "")
      + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_uint16(int_tp vec_len,
                                             string src_val) const {
   return "convert_ushort" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_uint32(int_tp vec_len,
                                             string src_val) const {
   return "convert_uint" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_uint64(int_tp vec_len,
                                             string src_val) const {
   return "convert_ulong" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_int8(int_tp vec_len,
                                           string src_val) const {
  return "convert_char" + (vec_len > 0 ? std::to_string(vec_len) : "")
      + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_int16(int_tp vec_len,
                                            string src_val) const {
   return "convert_short" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_int32(int_tp vec_len,
                                            string src_val) const {
   return "convert_int" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + "(" + src_val + ")";
}
string OclDeviceProgram::convert_type_int64(int_tp vec_len,
                                            string src_val) const {
   return "convert_long" + (vec_len > 0 ? std::to_string(vec_len) : "")
       + "(" + src_val + ")";
}

string OclDeviceProgram::helper_functions_half() const {
  return "";
}
string OclDeviceProgram::helper_functions_float() const {
  return "";
}
string OclDeviceProgram::helper_functions_double() const {
  return "";
}
string OclDeviceProgram::helper_functions_uint8() const {
  return "";
}
string OclDeviceProgram::helper_functions_uint16() const {
  return "";
}
string OclDeviceProgram::helper_functions_uint32() const {
  return "";
}
string OclDeviceProgram::helper_functions_uint64() const {
  return "";
}

#endif  // USE_OPENCL

}
