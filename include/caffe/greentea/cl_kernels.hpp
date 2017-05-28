// AUTOMATICALLY GENERATED FILE, DO NOT EDIT
#include <string>
#include "caffe/common.hpp"
#ifdef USE_GREENTEA
#ifndef GREENTEA_CL_KERNELS_HPP_
#define GREENTEA_CL_KERNELS_HPP_
#include "caffe/greentea/greentea.hpp"
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
namespace caffe {
viennacl::ocl::program & RegisterKernels(viennacl::ocl::context *ctx);
viennacl::ocl::program & submit_conv_spatial_program(
viennacl::ocl::context *ctx, string name, string options);
std::string getKernelBundleName(int index);
int getKernelBundleCount();
template<typename Dtype>
std::string getKernelBundleSource(int index);
}  // namespace caffe
#endif
#endif
