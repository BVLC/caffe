#ifdef USE_OCL
#include <glog/logging.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/cl_kernel.hpp"
#include "caffe/util/device_alternate.hpp"

using std::map;
using std::pair;
using std::string;
using std::vector;

extern "C" const char _cl_threefry_start;
extern "C" const char _cl_threefry_end;
extern "C" const char _cl_math_functions_start;
extern "C" const char _cl_math_functions_end;

#define OCL_CHECK_OPT(err) \
    { if (err != CL_SUCCESS) { \
        if (no_check) \
          return false; \
        else \
          OCL_CHECK(err); \
    } }

namespace caffe {

struct Impl {
  ~Impl() {
    destroy_kernels();
  }

  bool submit_program(const char* name,
      const vector< pair<const char*, const char*> >& program_srcs,
      const char * options, bool no_check) {
    if (programs_.find(name) != programs_.end()) {
      return true;
    }

    static const char* core_defines =
        "#define Dtype float\n"
        "#define Dtype2 float2\n"
        "#define Dtype4 float4\n"
        "#define Dtype8 float8\n"
        "#define Dtype16 float16\n"
        "#define OCL_KERNEL_LOOP(i, n)"
        " for (int i = get_global_id(0); i < (n); i += get_global_size(0))\n";

    vector<const char*> sources;
    vector<size_t> sizes;
    sources.push_back(core_defines);
    sizes.push_back(strlen(core_defines));

    for (vector<pair<const char*, const char*> >::const_iterator
        it = program_srcs.begin(); it != program_srcs.end(); ++it) {
      sources.push_back(it->first);
      sizes.push_back(it->second - it->first);
    }

    cl_int errcode;

    cl_program program = clCreateProgramWithSource(
        Caffe::cl_state().get_context(), sources.size(), &sources[0], &sizes[0],
        &errcode);
    OCL_CHECK_OPT(errcode);

    errcode = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (errcode != CL_SUCCESS) {
      char log[16384];
      clGetProgramBuildInfo(program, Caffe::cl_state().get_device(),
          CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
      printf("%s", log);
    }
    OCL_CHECK_OPT(errcode);

    programs_[name] = program;

    cl_uint num_kernels;
    OCL_CHECK_OPT(clCreateKernelsInProgram(program, 0, NULL, &num_kernels));

    vector<cl_kernel> kernels(num_kernels);
    OCL_CHECK_OPT(clCreateKernelsInProgram(program, num_kernels, kernels.data(),
      NULL));

    for (vector<cl_kernel>::iterator it = kernels.begin(); it != kernels.end();
         ++it) {
      ClKernel kernel(*it);

      char name[256];
      OCL_CHECK_OPT(clGetKernelInfo(*it, CL_KERNEL_FUNCTION_NAME, sizeof name,
          name, NULL));

      kernels_[name] = kernel;
    }

    return true;
  }

  void release_program(const char* name) {
    map<string, cl_program>::iterator itp = programs_.find(name);

    vector<string> kernels_to_del;
    for (map<string, ClKernel>::iterator it = kernels_.begin();
        it != kernels_.end(); ++it) {
      cl_program program;
      OCL_CHECK(clGetKernelInfo(it->second, CL_KERNEL_PROGRAM, sizeof program,
          &program, NULL));

      if (program == itp->second)
        kernels_to_del.push_back(it->first);
    }

    for (vector<string>::iterator it = kernels_to_del.begin();
      it != kernels_to_del.end(); ++it) {
      map<string, ClKernel>::iterator itk = kernels_.find(*it);


      clReleaseKernel(itk->second);
      kernels_.erase(itk);
    }

    clReleaseProgram(itp->second);
    programs_.erase(itp);
  }

  int get_num_programs() {
    return programs_.size();
  }

  void destroy_kernels() {
    for (map<string, ClKernel>::iterator it = kernels_.begin();
         it != kernels_.end(); ++it)
      clReleaseKernel(it->second);
    kernels_.clear();

    for (map<string, cl_program>::iterator it = programs_.begin();
         it != programs_.end(); ++it)
      clReleaseProgram(it->second);
    programs_.clear();
  }

  ClKernel& get_kernel(const char * name) {
    map<string, ClKernel>::iterator it = kernels_.find(name);
    if (it == kernels_.end()) {
      if (programs_.find(name) == programs_.end()) {
        vector< pair<const char*, const char*> > srcs;
        srcs.push_back(make_pair(&_cl_threefry_start, &_cl_threefry_end));
        srcs.push_back(make_pair(&_cl_math_functions_start,
            &_cl_math_functions_end));
        submit_program("math_functions", srcs, NULL, false);
        return get_kernel(name);
      } else {
        LOG(FATAL) << "Unknown cl kernel " << name;
      }
    }

    return it->second;
  }

  map<string, cl_program> programs_;
  map<string, ClKernel> kernels_;
};
static Impl impl_;

bool clkernel_submit_program(const char* name,
    const std::vector< std::pair<const char*, const char*> >& program_srcs,
    const char* options, bool no_check) {
  return impl_.submit_program(name, program_srcs, options, no_check);
}

void clkernel_release_program(const char* program_src) {
  impl_.release_program(program_src);
}

int clkernel_get_num_programs() {
  return impl_.get_num_programs();
}

void clkernel_destroy_kernels() {
  impl_.destroy_kernels();
}

ClKernel& clkernel_get_kernel(const char * name) {
  return impl_.get_kernel(name);
}

cl_mem ClKernel::get_buffer_from_ptr(const void* ptr) {
  ClMemOff<uint8_t> buf = Caffe::cl_state().get_buffer_mem(ptr);
  if (buf.memobj == NULL || buf.offset != 0)
    LOG(FATAL) << "Invalid memory or offset";
  return buf.memobj;
}

std::pair<cl_mem, int> ClKernel::get_buffer_offset_from_ptr(const void* ptr) {
  ClMemOff<uint8_t> buf = Caffe::cl_state().get_buffer_mem(ptr);
  return std::pair<cl_mem, int>(buf.memobj, static_cast<int>(buf.offset));
}

void ClKernel::enqueue(const size_t size) {
  const size_t global_work_size = CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(size);
  const size_t local_workgroup_size = OCL_LOCAL_WORKGROUP_SIZE;
  OCL_CHECK(clEnqueueNDRangeKernel(Caffe::cl_state().get_command_queue(),
      kernel_, 1, NULL, &global_work_size, &local_workgroup_size,
      0, NULL, NULL));
}

void ClKernel::enqueue_blocking(const size_t size) {
  const size_t global_work_size = CAFFE_GET_PADDED_GLOBAL_WORK_SIZE(size);
  const size_t local_workgroup_size = OCL_LOCAL_WORKGROUP_SIZE;

  OCL_CHECK(clEnqueueNDRangeKernel(Caffe::cl_state().get_command_queue(),
      kernel_, 1, NULL, &global_work_size, &local_workgroup_size,
      0, NULL, NULL));
  clFinish(Caffe::cl_state().get_command_queue());
}

void ClKernel::enqueue_params(const size_t size, const int N) {
  set_arg(0, N);
  enqueue_blocking(size);
}

}  // namespace caffe
#endif  // USE_OCL
