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
LibDNN<Dtype>::LibDNN() {
}

template<typename Dtype>
std::string LibDNN<Dtype>::generate_header() {
  std::stringstream ss;

  if (dev_ptr_->backend() == BACKEND_OpenCL) {
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

    if (std::is_same<Dtype, half_float::half>::value) {
      ss << "#if defined(cl_khr_fp16)" << std::endl;
      ss << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
      ss << "#define HALF_SUPPORT_AVAILABLE" << std::endl;
      ss << "#endif" << std::endl;
    }

    // Test/enable 32 bit atomics
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

    // 64 bit integers
    if (sizeof(int_tp) == 8 || std::is_same<Dtype, double>::value) {
      // Test/enable 64 bit atomics
      ss << "#if defined(cl_khr_int64_base_atomics)" << std::endl;
      ss << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
         << std::endl;
      ss << "#define ATOMICS_64_AVAILABLE" << std::endl;
      ss << "#endif" << std::endl;
    }
  }

  if (std::is_same<Dtype, double>::value) {
    ss << "#define Dtype double" << std::endl;
    ss << "#define Dtype1 double" << std::endl;
    // double2, double4, double8, double16
    for (int_tp i = 2; i <= 16; i *= 2) {
      ss << "#define Dtype" << i << " double" << i << std::endl;
    }
  } else if (std::is_same<Dtype, float>::value) {
    ss << "#define Dtype float" << std::endl;
    ss << "#define Dtype1 float" << std::endl;
    // float2, float4, float8, float16
    for (int_tp i = 2; i <= 16; i *= 2) {
      ss << "#define Dtype" << i << " float" << i << std::endl;
    }
  }
#ifdef HAS_HALF_SUPPORT
  else if (std::is_same<Dtype, half_float::half>::value) {
    ss << "#define Dtype half" << std::endl;
    ss << "#define Dtype1 half" << std::endl;
    // half2, half4, half8, half16
    for (int_tp i = 2; i <= 16; i *= 2) {
      ss << "#define Dtype" << i << " half" << i << std::endl;
    }
  }
#endif

  if (std::is_same<Dtype, half_float::half>::value) {
    ss << "#define KERNEL_ARG_DTYPE float" << std::endl;
  } else {
    ss << "#define KERNEL_ARG_DTYPE Dtype" << std::endl;
  }

  std::vector<std::string> elems4({
      "x", "y", "z", "w" });
  std::vector<std::string> elems16({
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8", "s9", "sA", "sB", "sC", "sD", "sE", "sF" });

  for (int_tp i = 1; i <= 16; i *= 2) {
    for (int_tp j = 0; j < i; ++j) {
      if (i == 1) {
        ss << "#define VEC_" << i << "_" << j << "(X)" << " X" << std::endl;
      } else if (i < 8) {
        ss << "#define VEC_" << i << "_" << j << "(X)" << " X." << elems4[j]
           << std::endl;
      } else {
        ss << "#define VEC_" << i << "_" << j << "(X)" << " X." << elems16[j]
           << std::endl;
      }
    }
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

  if (dev_ptr_->backend() == BACKEND_CUDA) {
    // Prepare definitions for OpenCL => CUDA cross compile
    // Mainly from: http://www.cedricnugteren.nl/tutorial.php?page=10
    ss << "#define __kernel __placeholder__" << std::endl;
    ss << "#define __global" << std::endl;
    ss << "#define __placeholder__ extern \"C\" __global__" << std::endl;
    ss << "#define __local __shared__" << std::endl;
    ss << "#define __restricted __restricted__" << std::endl;
    ss << "#define barrier(x) __syncthreads()" << std::endl;

    ss << "#define FLT_MIN 1.175494350822287507969e-38f"
       << std::endl;
    ss << "#define FLT_MAX 340282346638528859811704183484516925440.0f"
       << std::endl;

    ss << "__device__ int get_local_id(int x) {" << std::endl;
    ss << "if (x == 0) return threadIdx.x;" << std::endl;
    ss << "if (x == 1) return threadIdx.y;" << std::endl;
    ss << "if (x == 2) return threadIdx.z;" << std::endl;
    ss << "return 0;" << std::endl;
    ss << "}" << std::endl;

    ss << "__device__ int get_group_id(int x) {" << std::endl;
    ss << "if (x == 0) return blockIdx.x;" << std::endl;
    ss << "if (x == 1) return blockIdx.y;" << std::endl;
    ss << "if (x == 2) return blockIdx.z;" << std::endl;
    ss << "return 0;" << std::endl;
    ss << "}" << std::endl;

    ss << "__device__ int get_global_id(int x) {" << std::endl;
    ss << "if (x == 0) return blockIdx.x * blockDim.x" << " + threadIdx.x;"
       << std::endl;
    ss << "if (x == 1) return blockIdx.y * blockDim.y" << " + threadIdx.y;"
       << std::endl;
    ss << "if (x == 2) return blockIdx.z * blockDim.z" << " + threadIdx.z;"
       << std::endl;
    ss << "return 0;" << std::endl;
    ss << "}" << std::endl;

    ss << "__device__ int get_global_size(int x) {" << std::endl;
    ss << "if (x == 0) return blockDim.x * gridDim.x;" << std::endl;
    ss << "if (x == 1) return blockDim.y * gridDim.y;" << std::endl;
    ss << "if (x == 2) return blockDim.z * gridDim.z;" << std::endl;
    ss << "return 0;" << std::endl;
    ss << "}" << std::endl;
  }

  std::vector<std::string> atomic_funcs({ "Add", "Sub", "Mul", "Div" });
  std::vector<std::string> atomic_ops({ "+", "-", "*", "/" });

  // Atomic operations
  if (dev_ptr_->backend() == BACKEND_OpenCL) {
    // OpenCL atomics, derived from:
    // https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
    if (std::is_same<Dtype, double>::value) {
      ss << "#ifdef ATOMICS_64_AVAILABLE" << std::endl;
    } else {
      ss << "#ifdef ATOMICS_32_AVAILABLE" << std::endl;
    }
    // FIXME, half version has bug.
    for (int i = 0; i < atomic_funcs.size(); ++i) {
      ss << "inline void atomic" << atomic_funcs[i];
      ss << "(volatile __global Dtype* source, const Dtype operand) {"
         << std::endl;
      ss << "union {" << std::endl;
      if (std::is_same<Dtype, double>::value) {
        ss << "unsigned long intVal;" << std::endl;
      } else {
        ss << "unsigned int intVal;" << std::endl;
      }
      if (std::is_same<Dtype, half_float::half>::value) {
        ss << "Dtype floatVal[2];" << std::endl;
      } else {
        ss << "Dtype floatVal[1];" << std::endl;
      }
      ss << "} next, expected, current;" << std::endl;
      ss << "current.floatVal[0] = *source;" << std::endl;
      if (std::is_same<Dtype, half_float::half>::value)
        ss << "current.floatVal[1] = *(source + 1);" << std::endl;
      ss << "do {" << std::endl;
      ss << "expected.intVal = current.intVal;" << std::endl;
      ss << "next.floatVal[0] = expected.floatVal[0] "
         << atomic_ops[i] << " operand;" << std::endl;
      if (std::is_same<Dtype, half_float::half>::value) {
        ss << "next.floatVal[1] = expected.floatVal[1]; " << std::endl;
      }
      ss << "current.intVal = ";
      if (std::is_same<Dtype, double>::value) {
        ss << "atom_cmpxchg((volatile __global unsigned long *)";
      } else {
        ss << "atomic_cmpxchg((volatile __global unsigned int *)";
      }
      ss << "source, expected.intVal, next.intVal);" << std::endl;
      ss << "} while (current.intVal != expected.intVal);" << std::endl;
      ss << "}" << std::endl;
    }
    if (std::is_same<Dtype, double>::value) {
      ss << "#endif" << std::endl;
    } else {
      ss << "#endif" << std::endl;
    }
  }

  // Memory set
  ss << "__kernel void fill_memory(const int_tp n, "
     << "const KERNEL_ARG_DTYPE alpha,"
     << "__global Dtype* x, const int_tp offx) {" << std::endl;
  ss << "for (int_tp index = get_global_id(0); index < n; "
     << "index += get_global_size(0)) {" << std::endl;
  ss << "x[index + offx] = alpha;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}


template<typename Dtype>
bool LibDNN<Dtype>::CompileKernels() {
  std::string code_ext = "";

  if (dev_ptr_->backend() == BACKEND_OpenCL) {
    code_ext = ".cl";
  }
  if (dev_ptr_->backend() == BACKEND_CUDA)  {
    code_ext = ".cu";
  }

#ifdef LIBDNN_DEBUG
  FILE* fp = fopen((".libdnn_debug/" + string_identifier() + code_ext).c_str(),
                   "wb");
  fwrite(kernel_.c_str(), sizeof(char), kernel_.length(), fp);
  fclose(fp);
#endif  // LIBDNN_DEBUG

#ifdef USE_GREENTEA
  if (dev_ptr_->backend() == BACKEND_OpenCL) {
    CompileKernelsOpenCL(&(viennacl::ocl::get_context(dev_ptr_->id())));
  }
#endif  // USE_GREENTEA
#ifdef USE_CUDA
  if (dev_ptr_->backend() == BACKEND_CUDA) {
    CompileKernelsCuda();
  }
#endif  // USE_CUDA
  return true;
}

#ifdef USE_GREENTEA
template<typename Dtype>
viennacl::ocl::program LibDNN<Dtype>::CompileKernelsOpenCL(
    viennacl::ocl::context *ctx) {

  std::string build_opts = "";

  if (fast_unsafe_math_) {
    build_opts += "-cl-fast-relaxed-math -cl-mad-enable ";
  }

  if (is_same<Dtype, float>::value) {
    build_opts += "-cl-single-precision-constant ";
  }

  ctx->build_options(build_opts);

  ocl_program_ = ctx->add_program(kernel_.c_str(), "kernel_program");

#ifdef LIBDNN_DEBUG
  size_t bin_sz;
  clGetProgramInfo(ocl_program_.handle().get(),
                   CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL);
  unsigned char *bin = (unsigned char *)malloc(bin_sz);  // NOLINT
  clGetProgramInfo(ocl_program_.handle().get(),
                   CL_PROGRAM_BINARIES, sizeof(unsigned char *), &bin, NULL);
  FILE* fp = fopen((".libdnn_debug/" + string_identifier() + ".clptx").c_str(),
                   "wb");
  fwrite(bin, sizeof(char), bin_sz, fp);
  fclose(fp);
  free(bin);  // NOLINT
#endif  // LIBDNN_DEBUG

  return ocl_program_;
}
#endif  // USE_GREENTEA

#ifdef USE_CUDA
template<typename Dtype>
nvrtcProgram LibDNN<Dtype>::CompileKernelsCuda() {
  nvrtcCreateProgram(&cuda_program_, kernel_.c_str(), NULL, 0, NULL, NULL);

  std::vector<const char*> build_opts;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_ptr_->id());

  std::string arch_opt = "--gpu-architecture=compute_"
      + std::to_string(prop.major) + std::to_string(prop.minor);
  std::string stdcpp_opt = "--std=c++11";
  std::string fum_opt = "--use_fast_math";

  build_opts.push_back(arch_opt.c_str());
  build_opts.push_back(stdcpp_opt.c_str());
  if (fast_unsafe_math_) {
    build_opts.push_back(fum_opt.c_str());
  }
  nvrtcCompileProgram(cuda_program_, build_opts.size(), &build_opts[0]);

  size_t ptxSize;
  nvrtcGetPTXSize(cuda_program_, &ptxSize);
  char *ptx = new char[ptxSize];
  nvrtcGetPTX(cuda_program_, ptx);

  cuModuleLoadDataEx(&cuda_module_, ptx, 0, 0, 0);

#ifdef LIBDNN_DEBUG
  size_t log_size;
  nvrtcGetProgramLogSize(cuda_program_, &log_size);
  std::vector<char> log(log_size);
  nvrtcGetProgramLog(cuda_program_, log.data());

  std::cout << "CUDA compile log:" << std::endl;
  std::cout << log.data() << std::endl;

  FILE* fp = fopen((".libdnn_debug/" + string_identifier() + ".cuptx").c_str(),
                   "wb");
  fwrite(ptx, sizeof(char), ptxSize, fp);
  fclose(fp);
  free(ptx);
#endif  // LIBDNN_DEBUG

  return cuda_program_;
}
#endif  // USE_CUDA

template<typename Dtype>
void LibDNN<Dtype>::AllocateMemory(void** ptr, uint_tp size, int_tp flags) {
  if (dev_ptr_->backend() == BACKEND_OpenCL) {
#ifdef USE_GREENTEA
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_ptr_->id());
  *ptr = (void*)clCreateBuffer(ctx.handle().get(),    // NOLINT
                               flags,
                               size, nullptr, nullptr);
#endif  // USE_GREENTEA
  } else {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMalloc(ptr, size));
#endif  // USE_CUDA
  }
}

template<typename Dtype>
void LibDNN<Dtype>::SetMemory(Dtype* memory, int_tp count, int_tp offset,
                              Dtype value) {
  if (dev_ptr_->backend() == BACKEND_OpenCL) {
#ifdef USE_GREENTEA
    viennacl::ocl::kernel &kernel = ocl_program_.get_kernel("fill_memory");
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_ptr_->id());

    int wgs = dev_ptr_->workgroup_size(0);

    kernel.local_work_size(0, wgs);
    kernel.local_work_size(1, 1);
    kernel.local_work_size(2, 1);

    kernel.global_work_size(0, ((count - 1) / wgs + 1) * wgs);
    kernel.global_work_size(1, 1);
    kernel.global_work_size(2, 1);

    viennacl::ocl::enqueue(
        kernel(count, fixup_arg_type(value), WrapHandle((cl_mem) memory, &ctx),
        offset), ctx.get_queue());
#endif  // USE_GREENTEA
  } else {
#ifdef USE_CUDA
    CUfunction kernel;
    cuModuleGetFunction(&kernel, cuda_module_, "fill_memory");

    void *args[] = { &count, &value, &memory, &offset };
    cuLaunchKernel(kernel, (count + 512 - 1) / 512,   // Grid X
                   1,                                 // Grid Y
                   1,                                 // Grid Z
                   512, 1, 1,                         // Local
                   0, NULL, args, 0);                 // Arguments
#endif  // USE_CUDA
  }
}

INSTANTIATE_CLASS(LibDNN);

}  // namespace caffe

#endif  // USE_LIBDNN
