#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/test/test_caffe_main.hpp"

#ifndef CPU_ONLY  // CPU-GPU test
#ifdef USE_GREENTEA

#include "caffe/greentea/cl_kernels.hpp"
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"

namespace caffe {

template<typename TypeParam>
class OpenCLKernelCompileTest : public GPUDeviceTest<TypeParam> {
 protected:
  OpenCLKernelCompileTest() {
  }

  virtual void SetUp() {
  }

  virtual ~OpenCLKernelCompileTest() {
  }
};

TYPED_TEST_CASE(OpenCLKernelCompileTest, TestDtypes);

TYPED_TEST(OpenCLKernelCompileTest, TestCompile) {
  device* dev = Caffe::GetDefaultDevice();
  bool failure = false;
  if (dev->backend() == BACKEND_OpenCL) {
    int kcount = getKernelBundleCount();
    for (int i = 0; i < kcount; ++i) {
      std::string kernel = getKernelBundleSource<TypeParam>(i);
      std::string name = getKernelBundleName(i);
      std::string options = "";

      const char* kernel_program = kernel.c_str();
      size_t kernel_program_size = kernel.size();
      cl_int err;

      viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev->id());

      cl_program program = clCreateProgramWithSource(ctx.handle().get(), 1,
                                          (const char **)&kernel_program,
                                          &kernel_program_size, &err);

      clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);

      cl_build_status build_status;
      clGetProgramBuildInfo(program, ctx.devices()[0].id(),
                            CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status),
                            &build_status, NULL);
      if (build_status != CL_SUCCESS) {
        char *build_log;
        size_t ret_val_size;
        clGetProgramBuildInfo(program, ctx.devices()[0].id(),
                         CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        build_log = new char[ret_val_size+1];
        clGetProgramBuildInfo(program, ctx.devices()[0].id(),
                         CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        std::cout << "BUILD LOG: " << std::endl;
        std::cout << build_log << std::endl;
        delete[] build_log;
        failure = true;
      } else {
        std::cout << "Kernel bundle: " << name << ": OK" << std::endl;
      }
    }
    ASSERT_FALSE(failure);
  }
}

}  // namespace caffe
#endif  // USE_GREENTEA
#endif  // !CPU_ONLY
