// AUTOMATICALLY GENERATED FILE, DO NOT EDIT
#include "caffe/common.hpp"
#ifdef USE_GREENTEA
#include "caffe/greentea/cl_kernels.hpp"
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#ifdef DISABLE_DOUBLE_SUPPORT
  #define DOUBLE_SUPPORT "#define DISABLE_DOUBLE_SUPPORT\n"
#else
  #define DOUBLE_SUPPORT "#define ENABLE_DOUBLE_SUPPORT\n"
#endif  // DISABLE_DOUBLE_SUPPORT
namespace caffe {
#ifdef USE_INDEX_64
static std::string header = DOUBLE_SUPPORT "#ifndef __OPENCL_VERSION__\n#define __kernel\n#define __global\n#define __constant\n#define __local\n#define get_global_id(x) 0\n#define get_global_size(x) 0\n#define get_local_id(x) 0\n#define get_local_size(x) 0\n#define FLT_MAX 0\n#define FLT_MIN 0\n#define cl_khr_fp64\n#define cl_amd_fp64\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#define CLK_LOCAL_MEM_FENCE\n#define CLK_GLOBAL_MEM_FENCE\n#define Dtype float\n#define barrier(x)\n#define atomic_cmpxchg(x, y, z) x\n#define signbit(x) x\n#define int_tp long\n#define uint_tp unsigned long\n#define int_tpc long\n#define uint_tpc unsigned long\n#endif\n\n#define CONCAT(A,B) A##_##B\n#define TEMPLATE(name,type) CONCAT(name,type)\n\n#define TYPE_FLOAT 1\n#define TYPE_DOUBLE 2\n\n#if defined(cl_khr_fp64)\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#elif defined(cl_amd_fp64)\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#endif\n\n#if defined(cl_khr_int64_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n#define ATOMICS_64_AVAILABLE\n#endif\n\n#if defined(cl_khr_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if defined(cl_khr_global_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif";  // NOLINT
static std::string definitions_64 = DOUBLE_SUPPORT "// Types used for parameters, offset computations and so on\n#define int_tp long\n#define uint_tp unsigned long\n\n// Definitions used to cast the types above as needed\n#define int_tpc long\n#define uint_tpc unsigned long";  // NOLINT
#else
static std::string header = DOUBLE_SUPPORT "#ifndef __OPENCL_VERSION__\n#define __kernel\n#define __global\n#define __constant\n#define __local\n#define get_global_id(x) 0\n#define get_global_size(x) 0\n#define get_local_id(x) 0\n#define get_local_size(x) 0\n#define FLT_MAX 0\n#define FLT_MIN 0\n#define cl_khr_fp64\n#define cl_amd_fp64\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#define CLK_LOCAL_MEM_FENCE\n#define CLK_GLOBAL_MEM_FENCE\n#define Dtype float\n#define barrier(x)\n#define atomic_cmpxchg(x, y, z) x\n#define signbit(x) x\n#define int_tp long\n#define uint_tp unsigned long\n#define int_tpc long\n#define uint_tpc unsigned long\n#endif\n\n#define CONCAT(A,B) A##_##B\n#define TEMPLATE(name,type) CONCAT(name,type)\n\n#define TYPE_FLOAT 1\n#define TYPE_DOUBLE 2\n\n#if defined(cl_khr_fp64)\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#elif defined(cl_amd_fp64)\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#endif\n\n#if defined(cl_khr_int64_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n#define ATOMICS_64_AVAILABLE\n#endif\n\n#if defined(cl_khr_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if defined(cl_khr_global_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif";  // NOLINT
static std::string definitions_32 = DOUBLE_SUPPORT "// Types used for parameters, offset computations and so on\n#define int_tp int\n#define uint_tp unsigned int\n\n// Definitions used to cast the types above as needed\n#define int_tpc int\n#define uint_tpc unsigned int";  // NOLINT
#endif
static std::vector<std::vector<std::string>> cl_kernels{
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(relu_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out,",    // NOLINT
"Dtype negative_slope) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(relu_backward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff,",    // NOLINT
"Dtype negative_slope) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out_diff[index] = in_diff[index]",    // NOLINT
"* ((in_data[index] > 0?1.0:0.0) + (in_data[index] <= 0?1.0:0.0) * negative_slope);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(tanh_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = tanh(in[index]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(tanh_backward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* out_data,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"Dtype tanhx = out_data[index];",    // NOLINT
"out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sigmoid_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = 1.0 / (1.0 + exp(-in[index]));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sigmoid_backward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* out_data,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"const Dtype sigmoid_x = out_data[index];",    // NOLINT
"out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(threshold,Dtype)(const int_tp n, const Dtype threshold,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] > threshold ? 1.0 : 0.0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(prelu_forward,Dtype)(const int_tp n, const int_tp channels,",    // NOLINT
"const int_tp dim,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out,",    // NOLINT
"__global const Dtype* slope_data,",    // NOLINT
"const int_tp div_factor) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"int_tp c = (index / dim) % channels / div_factor;",    // NOLINT
"out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(prelu_backward,Dtype)(const int_tp n, const int_tp channels,",    // NOLINT
"const int_tp dim,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff,",    // NOLINT
"__global const Dtype* slope_data,",    // NOLINT
"const int_tp div_factor) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"int_tp c = (index / dim) % channels / div_factor;",    // NOLINT
"out_diff[index] = in_diff[index]",    // NOLINT
"* ((Dtype)(in_data[index] > 0?1.0:0.0) + (Dtype)(in_data[index] <= 0?1.0:0.0) * slope_data[c]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(prelu_param_backward,Dtype)(const int_tp n, const int_tp rows,",    // NOLINT
"const int_tp rowPitch,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0?1.0:0.0);",    // NOLINT
"for (int k = 1; k < rows; k++) {",    // NOLINT
"out_diff[index] += in_diff[index + k * rowPitch]",    // NOLINT
"* in_data[index + k * rowPitch]",    // NOLINT
"* (in_data[index + k * rowPitch] <= 0?1.0:0.0);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sce_loss_forward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* input_data,",    // NOLINT
"__global const Dtype* target,",    // NOLINT
"__global Dtype* loss,",    // NOLINT
"const int_tp has_ignore_label_,",    // NOLINT
"const int_tp ignore_label_,",    // NOLINT
"__global Dtype* counts) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < nthreads; i += get_global_size(0)) {",    // NOLINT
"const int_tp target_value = (int_tp)(target[i]);",    // NOLINT
"if (has_ignore_label_ == 1 && target_value == ignore_label_) {",    // NOLINT
"loss[i] = 0.0;",    // NOLINT
"counts[i] = 0.0;",    // NOLINT
"} else {",    // NOLINT
"loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0.0)) -",    // NOLINT
"log((Dtype)1.0 + exp(input_data[i] - (Dtype)2.0 * input_data[i] *",    // NOLINT
"(input_data[i] >= 0.0)));",    // NOLINT
"counts[i] = 1.0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sce_loss_ignore_diff,Dtype)(const int_tp count,",    // NOLINT
"const int_tp ignore_label,",    // NOLINT
"__global const Dtype* target,",    // NOLINT
"__global Dtype* diff) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < count; i += get_global_size(0)) {",    // NOLINT
"const int_tp target_value = (int_tp)(target[i]);",    // NOLINT
"if (target_value == ignore_label) {",    // NOLINT
"diff[i] = 0.0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(gpu_set,Dtype)(const int_tp n, const Dtype alpha, __global Dtype* y) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[index] = alpha;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(batch_norm_use_global_stats_in_place,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const Dtype scale, const Dtype eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global Dtype* top) {",    // NOLINT
"const int_tp idx_num = get_global_id(0);",    // NOLINT
"const int_tp idx_chans = get_global_id(1);",    // NOLINT
"const int_tp idx_spatial_dim = get_global_id(2);",    // NOLINT
"",    // NOLINT
"Dtype m = mean[idx_chans];",    // NOLINT
"Dtype v = variance[idx_chans];",    // NOLINT
"",    // NOLINT
"m = -scale * m;",    // NOLINT
"v = (Dtype)native_powr((float)mad(scale, v, eps), (float)-0.5);",    // NOLINT
"",    // NOLINT
"const int_tp out_off = (idx_num * channels + idx_chans) * spatial_dim + idx_spatial_dim;",    // NOLINT
"top[out_off] = v * (top[out_off] + m);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(batch_norm_use_global_stats,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const Dtype scale, const Dtype eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global const Dtype* bottom,",    // NOLINT
"__global Dtype* top) {",    // NOLINT
"const int_tp idx_num = get_global_id(0);",    // NOLINT
"const int_tp idx_chans = get_global_id(1);",    // NOLINT
"const int_tp idx_spatial_dim = get_global_id(2);",    // NOLINT
"",    // NOLINT
"Dtype m = mean[idx_chans];",    // NOLINT
"Dtype v = variance[idx_chans];",    // NOLINT
"",    // NOLINT
"m = -scale * m;",    // NOLINT
"v = (Dtype)native_powr((float)mad(scale, v, eps), (float)-0.5);",    // NOLINT
"",    // NOLINT
"const int_tp out_off = (idx_num * channels + idx_chans) * spatial_dim + idx_spatial_dim;",    // NOLINT
"top[out_off] = v * (bottom[out_off] + m);",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(br_forward,Dtype)(const int_tp count, const int_tp inner_dim,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const Dtype* permut,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < count;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp n = index / (inner_dim);",    // NOLINT
"int_tp in_n = (int_tp) (permut[n]);",    // NOLINT
"out[index] = in[in_n * (inner_dim) + index % (inner_dim)];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(br_backward,Dtype)(const int_tp count, const int_tp inner_dim,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const Dtype* top_indexes,",    // NOLINT
"__global const Dtype* begins,",    // NOLINT
"__global const Dtype* counts,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < count;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp n = index / (inner_dim);",    // NOLINT
"out[index] = 0;",    // NOLINT
"int_tp lower = (int_tp) (begins[n]);",    // NOLINT
"int_tp upper = lower + (int_tp) (counts[n]);",    // NOLINT
"for (int_tp i = lower; i < upper; ++i) {",    // NOLINT
"int_tp in_n = (int_tp) (top_indexes[i]);",    // NOLINT
"out[index] += in[in_n * (inner_dim) + index % (inner_dim)];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(null_kernel,Dtype)(Dtype arg) {",    // NOLINT
"Dtype out = arg;",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bias_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const Dtype* bias,",    // NOLINT
"const int_tp bias_dim,",    // NOLINT
"const int_tp inner_dim,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp bias_index = (index / inner_dim) % bias_dim;",    // NOLINT
"out[index] = in[index] + bias[bias_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(scale_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const Dtype* scale,",    // NOLINT
"const int_tp scale_dim,",    // NOLINT
"const int_tp inner_dim,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp scale_index = (index / inner_dim) % scale_dim;",    // NOLINT
"out[index] = in[index] * scale[scale_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(scale_bias_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const Dtype* scale,",    // NOLINT
"__global const Dtype* bias,",    // NOLINT
"const int_tp scale_dim,",    // NOLINT
"const int_tp inner_dim,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp scale_index = (index / inner_dim) % scale_dim;",    // NOLINT
"out[index] = in[index] * scale[scale_index] + bias[scale_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bnll_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"if (in[index] > 0.0f) {",    // NOLINT
"out[index] = in[index] + log((Dtype) (1.0 + exp(-in[index])));",    // NOLINT
"} else {",    // NOLINT
"out[index] = log((Dtype) (1.0 + exp(in[index])));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bnll_backward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"Dtype kBNLL_THRESHOLD = 50.;",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"Dtype expval = exp(min(in_data[index], kBNLL_THRESHOLD));",    // NOLINT
"out_diff[index] = in_diff[index] * expval / (expval + 1.);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(kernel_channel_max,Dtype)(const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"__global const Dtype* data,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < num * spatial_dim; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp n = index / spatial_dim;",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"float maxval = -FLT_MAX;",    // NOLINT
"for (int_tp c = 0; c < channels; ++c) {",    // NOLINT
"maxval = max((Dtype)(data[(n * channels + c) * spatial_dim + s]), (Dtype)maxval);",    // NOLINT
"}",    // NOLINT
"out[index] = maxval;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(kernel_channel_subtract,Dtype)(const int_tp count, const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"__global const Dtype* channel_max,",    // NOLINT
"__global Dtype* data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < count;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp n = index / channels / spatial_dim;",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"data[index] -= channel_max[n * spatial_dim + s];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(kernel_exp,Dtype)(const int_tp count, __global const Dtype* data,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < count;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"out[index] = exp(data[index]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(kernel_channel_sum,Dtype)(const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"__global const Dtype* data,",    // NOLINT
"__global Dtype* channel_sum) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < num * spatial_dim; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp n = index / spatial_dim;",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"Dtype sum = 0;",    // NOLINT
"for (int_tp c = 0; c < channels; ++c) {",    // NOLINT
"sum += data[(n * channels + c) * spatial_dim + s];",    // NOLINT
"}",    // NOLINT
"channel_sum[index] = sum;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(kernel_channel_div,Dtype)(const int_tp count, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"__global const Dtype* channel_sum,",    // NOLINT
"__global Dtype* data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < count;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp n = index / channels / spatial_dim;",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"data[index] /= channel_sum[n * spatial_dim + s];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(kernel_channel_dot,Dtype)(const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"__global const Dtype* data_1,",    // NOLINT
"__global const Dtype* data_2,",    // NOLINT
"__global Dtype* channel_dot) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < num * spatial_dim; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp n = index / spatial_dim;",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"Dtype dot = 0;",    // NOLINT
"for (int_tp c = 0; c < channels; ++c) {",    // NOLINT
"dot += (data_1[(n * channels + c) * spatial_dim + s]",    // NOLINT
"* data_2[(n * channels + c) * spatial_dim + s]);",    // NOLINT
"}",    // NOLINT
"channel_dot[index] = dot;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(concat,Dtype)(const int_tp nthreads, __global const Dtype* in_data,",    // NOLINT
"const int forward, const int_tp num_concats,",    // NOLINT
"const int_tp concat_size,",    // NOLINT
"const int_tp top_concat_axis,",    // NOLINT
"const int_tp bottom_concat_axis,",    // NOLINT
"const int_tp offset_concat_axis,",    // NOLINT
"__global Dtype* out_data) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp total_concat_size = concat_size * bottom_concat_axis;",    // NOLINT
"const int_tp concat_num = index / total_concat_size;",    // NOLINT
"const int_tp concat_index = index % total_concat_size;",    // NOLINT
"const int_tp top_index = concat_index",    // NOLINT
"+ (concat_num * top_concat_axis + offset_concat_axis) * concat_size;",    // NOLINT
"if (forward == 1) {",    // NOLINT
"out_data[top_index] = in_data[index];",    // NOLINT
"} else {",    // NOLINT
"out_data[index] = in_data[top_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(cll_backward,Dtype)(const int_tp count, const int_tp channels,",    // NOLINT
"const Dtype margin, const Dtype alpha, __global const Dtype* y,",    // NOLINT
"__global const Dtype* diff, __global const Dtype* dist_sq,",    // NOLINT
"__global Dtype *bottom_diff) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < count;",    // NOLINT
"i += get_global_size(0)) {",    // NOLINT
"int_tp n = i / channels;  // the num index, to access y and dist_sq",    // NOLINT
"if (trunc(y[n]) != 0.) {  // similar pairs",    // NOLINT
"bottom_diff[i] = alpha * diff[i];",    // NOLINT
"} else {  // dissimilar pairs",    // NOLINT
"Dtype mdist = 0.;",    // NOLINT
"Dtype beta = 0.;",    // NOLINT
"Dtype dist = sqrt(dist_sq[n]);",    // NOLINT
"mdist = (margin - dist);",    // NOLINT
"beta = -alpha * mdist / (dist + 1e-4) * diff[i];",    // NOLINT
"if (mdist > 0.) {",    // NOLINT
"bottom_diff[i] = beta;",    // NOLINT
"} else {",    // NOLINT
"bottom_diff[i] = 0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(cll_backward_legacy,Dtype)(const int count, const int channels,",    // NOLINT
"const Dtype margin, const Dtype alpha, __global Dtype* y,",    // NOLINT
"__global Dtype* diff, __global Dtype* dist_sq,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < count;",    // NOLINT
"i += get_global_size(0)) {",    // NOLINT
"int n = i / channels;  // the num index, to access y and dist_sq",    // NOLINT
"if (trunc(y[n]) != 0.) {  // similar pairs",    // NOLINT
"bottom_diff[i] = alpha * diff[i];",    // NOLINT
"} else {  // dissimilar pairs",    // NOLINT
"Dtype mdist = 0.;",    // NOLINT
"Dtype beta = 0.;",    // NOLINT
"mdist = (margin - dist_sq[n]);",    // NOLINT
"beta = -alpha;",    // NOLINT
"if (mdist > 0.) {",    // NOLINT
"bottom_diff[i] = beta;",    // NOLINT
"} else {",    // NOLINT
"bottom_diff[i] = 0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(conv_layer_spatial_phony,Dtype)(Dtype arg) {",    // NOLINT
"Dtype out = arg;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) do { (_dst_)[(_offset_)] = (_data_);} while(0)",    // NOLINT
"",    // NOLINT
"#define __CAT(x, y) x##y",    // NOLINT
"#define CAT(x, y) __CAT(x, y)",    // NOLINT
"#define LOOP0(VAR, STMT)",    // NOLINT
"#define LOOP1(VAR, STMT) (STMT); (VAR)++;",    // NOLINT
"#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;",    // NOLINT
"#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))",    // NOLINT
"",    // NOLINT
"#ifdef MULTI",    // NOLINT
"__kernel void CFMultiNoPadding(",    // NOLINT
"__global Dtype* image_data,",    // NOLINT
"int_tp image_offset,",    // NOLINT
"__global Dtype* kernel_data, int_tp kernel_offset,",    // NOLINT
"__global Dtype* bias,const int_tp bias_offset,",    // NOLINT
"__global Dtype* convolved_image,const int_tp convolved_image_offset,",    // NOLINT
"const ushort input_width,",    // NOLINT
"const ushort input_height,",    // NOLINT
"const ushort output_width,",    // NOLINT
"const ushort output_height,",    // NOLINT
"const ushort pad_w,",    // NOLINT
"const ushort pad_h) {",    // NOLINT
"",    // NOLINT
"const int_tp outputX = get_global_id(0);",    // NOLINT
"const int_tp outputY = get_global_id(1);",    // NOLINT
"const int_tp kernelNum = get_global_id(2)*ZPAR;",    // NOLINT
"if(outputX < output_width && outputY < output_height)",    // NOLINT
"{",    // NOLINT
"Dtype sum[ZPAR];",    // NOLINT
"for(int_tp kern =0; kern < ZPAR; kern++)",    // NOLINT
"{",    // NOLINT
"sum[kern] = 0.0f;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"const int_tp org_y = outputY * STRIDE_H - pad_h;",    // NOLINT
"const int_tp org_x = outputX * STRIDE_W - pad_w;",    // NOLINT
"const int_tp currentKernelOffset = kernel_offset + kernelNum*KERNEL_H*KERNEL_W*CHANNELS;",    // NOLINT
"const int_tp biasIndex=bias_offset + kernelNum;",    // NOLINT
"const int_tp local_image_offset = org_y*input_width + org_x;",    // NOLINT
"const int_tp imageSize = input_width*input_height;",    // NOLINT
"",    // NOLINT
"__global Dtype* image_dataPtrFloat = (image_data + (image_offset + local_image_offset));",    // NOLINT
"__global Dtype* kernel_dataPtrFloat = (kernel_data + (currentKernelOffset));",    // NOLINT
"",    // NOLINT
"for(int_tp c = 0; c < CHANNELS; c++)",    // NOLINT
"{",    // NOLINT
"for(int_tp y = 0; y < KERNEL_H; y++)",    // NOLINT
"{",    // NOLINT
"for(int_tp x = 0; x < KERNEL_W; x++)",    // NOLINT
"{",    // NOLINT
"if(!(org_y + y * DILATION_Y >= 0 && org_y + y * DILATION_Y < input_height && org_x + x * DILATION_X >= 0 && org_x + x * DILATION_X < input_width))",    // NOLINT
"{",    // NOLINT
"continue;",    // NOLINT
"}",    // NOLINT
"for(int_tp kern =0; kern < ZPAR; kern++)",    // NOLINT
"{",    // NOLINT
"sum[kern] += image_dataPtrFloat[x * DILATION_X] * kernel_dataPtrFloat[kern*KERNEL_H*KERNEL_W*CHANNELS + x];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"image_dataPtrFloat += input_width * DILATION_Y;",    // NOLINT
"kernel_dataPtrFloat += KERNEL_W;",    // NOLINT
"}",    // NOLINT
"image_dataPtrFloat += imageSize - input_width*KERNEL_H*DILATION_Y;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(APPLY_BIAS == 1)",    // NOLINT
"{",    // NOLINT
"for(int_tp kern = 0; kern < ZPAR; kern++)",    // NOLINT
"{",    // NOLINT
"if(kernelNum+kern < OUTPUT_Z)",    // NOLINT
"{",    // NOLINT
"int_tp offset = convolved_image_offset + (kernelNum+kern)*output_height*output_width + outputY*output_width + outputX;",    // NOLINT
"ACTIVATION_FUNCTION(convolved_image, offset, sum[kern] + bias[biasIndex +kern]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"for(int_tp kern = 0; kern < ZPAR; kern++)",    // NOLINT
"{",    // NOLINT
"if(kernelNum+kern < OUTPUT_Z)",    // NOLINT
"{",    // NOLINT
"int_tp offset = convolved_image_offset + (kernelNum+kern)*output_height*output_width + outputY*output_width + outputX;",    // NOLINT
"ACTIVATION_FUNCTION(convolved_image, offset, sum[kern]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"//Begin IDLF kernels below here",    // NOLINT
"#ifdef IDLF",    // NOLINT
"",    // NOLINT
"#define activation_function(x) (x)",    // NOLINT
"#define OUT_BLOCK_SIZE (OUT_BLOCK_WIDTH*OUT_BLOCK_HEIGHT)",    // NOLINT
"",    // NOLINT
"// Each work-item computes a OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT region of one output map.",    // NOLINT
"// Each work-group (which will be mapped to 1 SIMD16/SIMD8 EU thread) will compute 16/8 different feature maps, but each feature map is for the same region of the imput image.",    // NOLINT
"// NDRange:  (output_width+pad)/ OUT_BLOCK_WIDTH, (output_height+pad)/OUT_BLOCK_HEIGHT, NUM_FILTERS/OUT_BLOCK_DEPTH",    // NOLINT
"",    // NOLINT
"// NOTE: for beignet this reqd_work_group_size does not guarantee that SIMD16/8 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.",    // NOLINT
"__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))",    // NOLINT
"kernel void",    // NOLINT
"convolve_simd(  // __global float *inputs, __global float* weights, __global float* outputs",    // NOLINT
"__global float* inputs_base,",    // NOLINT
"filter_qualifier float* weights_base,",    // NOLINT
"__global float* biases_base,",    // NOLINT
"__global float* outputs_base,",    // NOLINT
"const ushort input_width,",    // NOLINT
"const ushort input_height,",    // NOLINT
"const ushort output_width,",    // NOLINT
"const ushort output_height)",    // NOLINT
"{",    // NOLINT
"__global float* outputs = outputs_base;",    // NOLINT
"__global float* inputs = inputs_base;",    // NOLINT
"filter_qualifier float* weights = weights_base;",    // NOLINT
"__global float* biases = biases_base;",    // NOLINT
"",    // NOLINT
"uint_tp oc = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column",    // NOLINT
"uint_tp or = get_global_id(1) * OUT_BLOCK_HEIGHT;// or = Output Row",    // NOLINT
"uint_tp fm = get_global_id(2);// fm = Feature Map = od = Output Depth",    // NOLINT
"uint_tp fmg = get_group_id(2);",    // NOLINT
"uint_tp lid = get_local_id(2);",    // NOLINT
"",    // NOLINT
"float out[OUT_BLOCK_SIZE];",    // NOLINT
"",    // NOLINT
"int_tp in_addr;",    // NOLINT
"",    // NOLINT
"// find weights adress of given neuron (lid is index)",    // NOLINT
"uint_tp weight_addr = (fmg % (ALIGNED_NUM_FILTERS/SIMD_SIZE)) * INPUT_DEPTH * KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE + lid;",    // NOLINT
"",    // NOLINT
"for(int_tp i=0;i<OUT_BLOCK_SIZE;i++) {",    // NOLINT
"out[i]=0.0f;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"uint_tp num_in_batch = ( fm ) / ALIGNED_NUM_FILTERS;",    // NOLINT
"",    // NOLINT
"uint_tp input_batch_offset = num_in_batch * input_height * input_width * TOTAL_INPUT_DEPTH_SIZE;",    // NOLINT
"",    // NOLINT
"int curr_y = or * STRIDEY + INPUT_START_Y + ( lid / ( TILE_X / 4 ) );",    // NOLINT
"int curr_x = oc * STRIDEX + INPUT_START_X + ( lid % ( TILE_X / 4 ) ) * 4;",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0",    // NOLINT
"int saved_y = curr_y;",    // NOLINT
"#endif",    // NOLINT
"in_addr = input_batch_offset + INPUT_START_Z * input_height * input_width",    // NOLINT
"+  (curr_y - INPUT_PAD_H) * input_width             // y tile offset",    // NOLINT
"+   curr_x - INPUT_PAD_W;                        // x tile offset",    // NOLINT
"union {",    // NOLINT
"float4 in_vec[INVEC_SIZE];",    // NOLINT
"float in_array[INVEC_SIZE * 4];",    // NOLINT
"} in_buf;",    // NOLINT
"",    // NOLINT
"for(int_tp kd = 0; kd < INPUT_DEPTH; kd++)",    // NOLINT
"{",    // NOLINT
"int_tp in_offset = in_addr;",    // NOLINT
"int_tp reg = 0;",    // NOLINT
"LOOP(INVEC_SIZE, reg,",    // NOLINT
"{",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0",    // NOLINT
"if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + 3 >= INPUT_PAD_W && curr_x < input_width + INPUT_PAD_W) {",    // NOLINT
"if (curr_x < INPUT_PAD_W) {",    // NOLINT
"in_buf.in_vec[reg].s0 = 0;",    // NOLINT
"if (curr_x + 1 >= INPUT_PAD_W)",    // NOLINT
"in_buf.in_vec[reg].s1 = *(inputs + in_offset + 1);",    // NOLINT
"else",    // NOLINT
"in_buf.in_vec[reg].s1 = 0;",    // NOLINT
"if (curr_x + 2 >= INPUT_PAD_W)",    // NOLINT
"in_buf.in_vec[reg].s2 = *(inputs + in_offset + 2);",    // NOLINT
"else",    // NOLINT
"in_buf.in_vec[reg].s2 = 0;",    // NOLINT
"in_buf.in_vec[reg].s3 = *(inputs + in_offset + 3);",    // NOLINT
"} else {",    // NOLINT
"in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);    // read SIMD_SIZE elements",    // NOLINT
"if (curr_x + 1 >= input_width + INPUT_PAD_W)",    // NOLINT
"in_buf.in_vec[reg].s1 = 0;",    // NOLINT
"if (curr_x + 2 >= input_width + INPUT_PAD_W)",    // NOLINT
"in_buf.in_vec[reg].s2 = 0;",    // NOLINT
"if (curr_x + 3 >= input_width + INPUT_PAD_W)",    // NOLINT
"in_buf.in_vec[reg].s3 = 0;",    // NOLINT
"}",    // NOLINT
"} else {",    // NOLINT
"in_buf.in_vec[reg] = 0;",    // NOLINT
"}",    // NOLINT
"curr_y += TILE_Y_STRIDE;",    // NOLINT
"#else",    // NOLINT
"in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);    // read SIMD_SIZE elements",    // NOLINT
"#endif",    // NOLINT
"in_offset += input_width * TILE_Y_STRIDE;",    // NOLINT
"});",    // NOLINT
"in_addr += input_height * input_width;",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0",    // NOLINT
"curr_y = saved_y;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT != 1",    // NOLINT
"#define WEIGHT_PREF 8",    // NOLINT
"#else",    // NOLINT
"#define WEIGHT_PREF 1",    // NOLINT
"#endif",    // NOLINT
"union {",    // NOLINT
"float w[WEIGHT_PREF];",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT != 1",    // NOLINT
"uint8 ui8;",    // NOLINT
"#endif",    // NOLINT
"} weight_buf;",    // NOLINT
"int_tp w_idx=0;",    // NOLINT
"",    // NOLINT
"uint_tp orig_weight_addr = weight_addr;",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT != 1",    // NOLINT
"weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);",    // NOLINT
"weight_addr += SIMD_SIZE * WEIGHT_PREF;",    // NOLINT
"#else",    // NOLINT
"weight_buf.w[0] = as_float(intel_sub_group_block_read((__global uint *)&weights[weight_addr]));",    // NOLINT
"weight_addr += SIMD_SIZE * 1;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define BLOCK_IN(n) sub_group_broadcast( in_buf.in_array[((n)%4) + ((n) / (TILE_Y_STRIDE * TILE_X)) * 4], (((n) % (TILE_Y_STRIDE * TILE_X))/4))",    // NOLINT
"",    // NOLINT
"int_tp kr = 0;  // kr = Kernel Row",    // NOLINT
"LOOP(KERNEL_HEIGHT, kr,// LOOP is a macro that unrolls the loop.",    // NOLINT
"{",    // NOLINT
"int_tp kc = 0;  // kc = Kernel Column",    // NOLINT
"LOOP(KERNEL_WIDTH, kc,",    // NOLINT
"{",    // NOLINT
"for(int_tp br=0; br < OUT_BLOCK_HEIGHT; br++) {",    // NOLINT
"for(int_tp bc=0; bc < OUT_BLOCK_WIDTH; bc++) {",    // NOLINT
"float input = BLOCK_IN((br * STRIDEY + kr * DILATION_Y) * TILE_X + bc * STRIDEX + kc * DILATION_X);",    // NOLINT
"out[br * OUT_BLOCK_WIDTH + bc] = mad(weight_buf.w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT > WEIGHT_PREF",    // NOLINT
"// We assume KERNEL_W is equal to KERNEL_H here.",    // NOLINT
"if ((w_idx + 1) % WEIGHT_PREF == 0",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 != 0",    // NOLINT
"&& ((w_idx + 1) <= (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF))",    // NOLINT
"#endif",    // NOLINT
") {",    // NOLINT
"weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);",    // NOLINT
"weight_addr += SIMD_SIZE * WEIGHT_PREF;  // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.",    // NOLINT
"}",    // NOLINT
"#if KERNEL_WIDTH*KERNEL_HEIGHT % 8 == 0",    // NOLINT
"// need to do nothing",    // NOLINT
"#else",    // NOLINT
"else if ((w_idx + 1) %  WEIGHT_PREF == 0 && ((w_idx + 1) > (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF)))",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 1",    // NOLINT
"weight_buf.w[0] = weights[weight_addr];",    // NOLINT
"#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 2",    // NOLINT
"weight_buf.ui8.s01 = intel_sub_group_block_read2((__global uint *)&weights[weight_addr]);",    // NOLINT
"#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 <= 4",    // NOLINT
"weight_buf.ui8.s0123 = intel_sub_group_block_read4((__global uint *)&weights[weight_addr]);",    // NOLINT
"#else",    // NOLINT
"weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"++w_idx;",    // NOLINT
"});",    // NOLINT
"});",    // NOLINT
"weight_addr = orig_weight_addr + KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE;",    // NOLINT
"",    // NOLINT
"}",    // NOLINT
"// dead code to work around possible compiler bug.",    // NOLINT
"if (ALIGNED_NUM_FILTERS != NUM_FILTERS && fm > 0xfffffffeul) {",    // NOLINT
"outputs[0] = BLOCK_IN(fm % SIMD_SIZE);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"fm = fm % ALIGNED_NUM_FILTERS;",    // NOLINT
"",    // NOLINT
"if ((ALIGNED_NUM_FILTERS == NUM_FILTERS || fm < NUM_FILTERS)) {",    // NOLINT
"",    // NOLINT
"uint_tp out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH + fm ) * output_width * output_height;",    // NOLINT
"out_addr += or * output_width + oc;",    // NOLINT
"float bias = biases[fm];",    // NOLINT
"",    // NOLINT
"for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {",    // NOLINT
"if (r + or >= output_height) break;",    // NOLINT
"for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {",    // NOLINT
"if (c + oc >= output_width) break;",    // NOLINT
"// this does a scattered write to SIMD_SIZE different feature maps, so that data within one map is contiguous, thus ready for input to next layer.",    // NOLINT
"outputs[out_addr + r * output_width + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"/*******************************************************************************",    // NOLINT
"Copyright © 2016, Intel Corporation",    // NOLINT
"",    // NOLINT
"Permission is hereby granted, free of charge, to any person obtaining a",    // NOLINT
"copy of this software and associated documentation files (the \"Software\"),",    // NOLINT
"to deal in the Software without restriction, including without limitation",    // NOLINT
"the rights to use, copy, modify, merge, publish, distribute, sublicense,",    // NOLINT
"and/or sell copies of the Software, and to permit persons to whom the",    // NOLINT
"Software is furnished to do so, subject to the following conditions:",    // NOLINT
"",    // NOLINT
"The above copyright notice and this permission notice shall be included in",    // NOLINT
"all copies or substantial portions of the Software.",    // NOLINT
"",    // NOLINT
"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR",    // NOLINT
"IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,",    // NOLINT
"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL",    // NOLINT
"THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER",    // NOLINT
"LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING",    // NOLINT
"FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER",    // NOLINT
"DEALINGS IN THE SOFTWARE.",    // NOLINT
"******************************************************************************/",    // NOLINT
"#ifdef Conv_Interleaved",    // NOLINT
"typedef struct float1 { float s0; } float1;",    // NOLINT
"typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;",    // NOLINT
"typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;",    // NOLINT
"typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;",    // NOLINT
"typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;",    // NOLINT
"typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;",    // NOLINT
"float s6; float s7; float s8; float s9;} float10;",    // NOLINT
"typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;",    // NOLINT
"float s6; float s7; float s8; float s9; float sa;} float11;",    // NOLINT
"typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;",    // NOLINT
"float s6; float s7; float s8; float s9; float sa; float sb; } float12;",    // NOLINT
"typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;",    // NOLINT
"float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;",    // NOLINT
"typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;",    // NOLINT
"float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;",    // NOLINT
"typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;",    // NOLINT
"float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;",    // NOLINT
"typedef struct float0 { float s0; } float0; //never used but makes compiler happy.",    // NOLINT
"",    // NOLINT
"#define OUT_PITCH_X output_width",    // NOLINT
"#define ROW_PITCH input_width",    // NOLINT
"",    // NOLINT
"#ifdef FUSED_CONV_ELTWISE",    // NOLINT
"#define GEMM_LIKE_KERNEL_ARGS         __global Dtype* eltwise_data,     const __global Dtype *src0,       const __global Dtype *src1,       const __global Dtype *biases,     __global Dtype *dst,              const ushort input_width,         const ushort input_height,        const ushort output_width,        const ushort output_height,       const int_tp out_pitch_y,         const int_tp out_pitch_z,         const int_tp aligned_input_size,     const int_tp slice_pitch",    // NOLINT
"#else",    // NOLINT
"#define GEMM_LIKE_KERNEL_ARGS         const __global Dtype *src0,       const __global Dtype *src1,       const __global Dtype *biases,     __global Dtype *dst,              const ushort input_width,         const ushort input_height,        const ushort output_width,        const ushort output_height,       const int_tp out_pitch_y,         const int_tp out_pitch_z,         const int_tp aligned_input_size,     const int_tp slice_pitch",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"#ifdef GEMM_LIKE_CONV_32_1",    // NOLINT
"//////////////////////////////////////////////////////////////////////////////",    // NOLINT
"// Conv_Interleaved_32_1_flex",    // NOLINT
"//",    // NOLINT
"// Convolution: each workitem computes 1 patch x 32 filters worth of output",    // NOLINT
"// data.  Kernel's inner loop works on a single tile consisting of one",    // NOLINT
"// row from each patch and the filter data corresponding to that row.  Filter",    // NOLINT
"// matrix is interleaved to reduce GRF bank conflicts.  Patches are walked",    // NOLINT
"// by rows and then by slices.  Relies on sub_group extension for block",    // NOLINT
"// reads and SIMD broadcast.  Allows flexible sizing of TILE width (TILE_N)",    // NOLINT
"// by dynamically selecting one of two code paths: one uses TILE_N = 32 and",    // NOLINT
"// the other uses TILE_N = 8, 16, or 24.",    // NOLINT
"#define TILE_M          1",    // NOLINT
"#define TILE_K          KERNEL_WIDTH",    // NOLINT
"#define TILE_N          32",    // NOLINT
"",    // NOLINT
"#ifdef __BEIGNET__",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(8)))",    // NOLINT
"#endif",    // NOLINT
"__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)",    // NOLINT
"{",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"const int global_z = get_global_id(2);",    // NOLINT
"int interleaved_y;",    // NOLINT
"int kernel_y;",    // NOLINT
"int kernel_idx;",    // NOLINT
"",    // NOLINT
"#define DOT_PRODUCT_8( _result, _rowA, colB )        {           _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );          _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );          _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );          _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );          _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );          _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );          _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );          _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );      }",    // NOLINT
"typedef CAT( float, KERNEL_WIDTH ) float_t;",    // NOLINT
"",    // NOLINT
"// True for all threads if filter_width is multiple of TILE_N",    // NOLINT
"// else, true for all but right-most column of threads.",    // NOLINT
"if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N )",    // NOLINT
"{",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"float8  blockC00 = 0.f;",    // NOLINT
"float8  blockC10 = 0.f;",    // NOLINT
"float8  blockC20 = 0.f;",    // NOLINT
"float8  blockC30 = 0.f;",    // NOLINT
"",    // NOLINT
"// Src0 (patch input) is directly used as atile.",    // NOLINT
"// Each work item points to the start of a different patch.",    // NOLINT
"// atile is M rows x K columns.",    // NOLINT
"int curr_x = ( global_y % output_width ) * STRIDE_X;",    // NOLINT
"int curr_y = ( global_y / output_width ) * STRIDE_Y;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"int saved_y = curr_y;",    // NOLINT
"#endif",    // NOLINT
"const __global float *src0_read = src0",    // NOLINT
"+ aligned_input_size * global_z                            // batch offset",    // NOLINT
"+ (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset",    // NOLINT
"+ (curr_x - INPUT_PAD_W);                 // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global float *src1_read = src1 + ( global_x * TILE_N  * 2);",    // NOLINT
"",    // NOLINT
"// Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.",    // NOLINT
"// Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch",    // NOLINT
"// and KERNEL_WIDTH/2 rows of interleaved filter.",    // NOLINT
"int patch_depth = 0;",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"int patch_row = 0;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"curr_y = saved_y;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"// Load atile and btile.",    // NOLINT
"// Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.",    // NOLINT
"// The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non",    // NOLINT
"// interleaved row is padded with zero to ensure same size as interleaved rows. This",    // NOLINT
"// interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the",    // NOLINT
"// kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.",    // NOLINT
"// (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..",    // NOLINT
"// (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...",    // NOLINT
"// (0, 2) (8, 2) (16, 2) (24, 2) ...       ...",    // NOLINT
"// ...",    // NOLINT
"const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;",    // NOLINT
"",    // NOLINT
"#if INPUT_PAD_W == 0 && INPUT_PAD_H == 0 && DILATION_X == 1 && DILATION_Y == 1",    // NOLINT
"float_t blockA00 = ( (const __global float_t*)src0_read )[  0  ];",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"#else",    // NOLINT
"float_t blockA00;",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + pos * DILATION_X >= INPUT_PAD_W && curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y += DILATION_Y;",    // NOLINT
"#endif",    // NOLINT
"src0_read += (ROW_PITCH * DILATION_Y);",    // NOLINT
"",    // NOLINT
"float blockB00[KERNEL_WIDTH*4];",    // NOLINT
"float8* p8BlockB00 = (float8*)blockB00;",    // NOLINT
"float4* p4BlockB00 = (float4*)blockB00;",    // NOLINT
"float*  pBlockB00 =  (float* )blockB00;",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p8BlockB00[interleaved_y] = as_float8( intel_sub_group_block_read8( (const __global uint*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p4BlockB00[KERNEL_WIDTH - 1] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"kernel_idx = 0;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"} )",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"//while( ++patch_row < 1 ); //debug",    // NOLINT
"while( ++patch_row < KERNEL_HEIGHT );",    // NOLINT
"",    // NOLINT
"src0_read += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y); // reset to start of next slice of patch",    // NOLINT
"}",    // NOLINT
"//while ( ++patch_depth < 1 ); //debug",    // NOLINT
"while ( ++patch_depth < INPUT_DEPTH );",    // NOLINT
"",    // NOLINT
"// Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:",    // NOLINT
"// (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.",    // NOLINT
"__global float *out = dst",    // NOLINT
"+ global_z * out_pitch_z                                                   // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                       // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset",    // NOLINT
"+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"float bias[4];",    // NOLINT
"float4 *bias_vec;",    // NOLINT
"bias_vec = (float4*)bias;",    // NOLINT
"*bias_vec = as_float4(intel_sub_group_block_read4((__global uint *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"if (global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < 8; i++)",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"out[( 8+i) * out_pitch_y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);",    // NOLINT
"out[(16+i) * out_pitch_y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);",    // NOLINT
"out[(24+i) * out_pitch_y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#if TILE_N_LAST > 0",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"int i = 0;",    // NOLINT
"float8  blockC[TILE_N_LAST_DIV8];",    // NOLINT
"LOOP(TILE_N_LAST_DIV8, i,",    // NOLINT
"{",    // NOLINT
"blockC[i] = 0.f;",    // NOLINT
"} )",    // NOLINT
"",    // NOLINT
"// Src0 (patch input) is directly used as atile.",    // NOLINT
"// Each work item points to the start of a different patch.",    // NOLINT
"// atile is M rows x K columns.",    // NOLINT
"int curr_x = ( global_y % output_width ) * STRIDE_X;",    // NOLINT
"int curr_y = ( global_y / output_width ) * STRIDE_Y;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"int saved_y = curr_y;",    // NOLINT
"#endif",    // NOLINT
"const __global float *src0_read = src0",    // NOLINT
"+ aligned_input_size * global_z                            // batch offset",    // NOLINT
"+ (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset",    // NOLINT
"+ (curr_x - INPUT_PAD_W);                 // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global float *src1_read = src1 + ( global_x * TILE_N  * 2);",    // NOLINT
"",    // NOLINT
"// Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.",    // NOLINT
"// Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch",    // NOLINT
"// and KERNEL_WIDTH/2 rows of interleaved filter.",    // NOLINT
"int patch_depth = 0;",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"int patch_row = 0;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"curr_y = saved_y;",    // NOLINT
"#endif",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"// Load atile and interleaved btile.",    // NOLINT
"const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;",    // NOLINT
"#if INPUT_PAD_W == 0 && INPUT_PAD_H == 0 && DILATION_X == 1 && DILATION_Y == 1",    // NOLINT
"float_t blockA00 = ( (const __global float_t*)src0_read )[  0  ];",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"#else",    // NOLINT
"float_t blockA00;",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + pos * DILATION_X >= INPUT_PAD_W && curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y += DILATION_Y;",    // NOLINT
"#endif",    // NOLINT
"src0_read += (ROW_PITCH * DILATION_Y);",    // NOLINT
"float blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"float2* p2BlockB = (float2* )blockB;",    // NOLINT
"p2BlockB[interleaved_y] = as_float2( intel_sub_group_block_read2( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"float4* p4BlockB = (float4* )blockB;",    // NOLINT
"p4BlockB[interleaved_y] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"//TODO: broken.  No block_read6",    // NOLINT
"float6* p6BlockB = (float6* )blockB;",    // NOLINT
"(*((float8*)(&p6BlockB[interleaved_y]))).s0123 = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );",    // NOLINT
"(*((float8*)(&p6BlockB[interleaved_y]))).s45 = as_float2( intel_sub_group_block_read2( (const __global uint*)(src1_read + 4 * 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"float* pBlockB = (float* )blockB;",    // NOLINT
"pBlockB[KERNEL_WIDTH - 1] = as_float( intel_sub_group_block_read( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"float2* p2BlockB = (float2* )blockB;",    // NOLINT
"p2BlockB[KERNEL_WIDTH - 1] = as_float2( intel_sub_group_block_read2( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"float3* p3BlockB = (float3* )blockB;",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s01 = as_float2( intel_sub_group_block_read2( (const __global uint*)src1_read ) );",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s2 = as_float( intel_sub_group_block_read( (const __global uint*) (src1_read + 2 * 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"float* pBlockB = (float*)blockB;",    // NOLINT
"kernel_idx = 0;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 2",    // NOLINT
"DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 3",    // NOLINT
"DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"} )",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 2",    // NOLINT
"DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 3",    // NOLINT
"DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"//while( ++patch_row < 1 ); //debug",    // NOLINT
"while( ++patch_row < KERNEL_HEIGHT );",    // NOLINT
"",    // NOLINT
"src0_read += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch",    // NOLINT
"}",    // NOLINT
"//while ( ++patch_depth < 1 );  //debug",    // NOLINT
"while ( ++patch_depth < INPUT_DEPTH );",    // NOLINT
"",    // NOLINT
"// Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:",    // NOLINT
"// (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.",    // NOLINT
"__global float *out = dst",    // NOLINT
"+ global_z * out_pitch_z                                                   // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                       // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset",    // NOLINT
"+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"float bias[4];",    // NOLINT
"float4 *bias_vec;",    // NOLINT
"bias_vec = (float4*)bias;",    // NOLINT
"*bias_vec = as_float4(intel_sub_group_block_read4((__global uint *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"if (global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < 8; i++)",    // NOLINT
"{",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 0 ) out[( 0+i) * out_pitch_y] = blockC[0][i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 1 ) out[( 8+i) * out_pitch_y] = blockC[1][i] + intel_sub_group_shuffle(bias[1], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 2 ) out[(16+i) * out_pitch_y] = blockC[2][i] + intel_sub_group_shuffle(bias[2], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 3 ) out[(24+i) * out_pitch_y] = blockC[3][i] + intel_sub_group_shuffle(bias[3], i);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#ifdef GEMM_LIKE_CONV_32_1_SIMD16",    // NOLINT
"#define TILE_M          1",    // NOLINT
"#define TILE_K          KERNEL_WIDTH",    // NOLINT
"#define TILE_N          32",    // NOLINT
"",    // NOLINT
"#ifndef __BEIGNET__",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(16)))",    // NOLINT
"#endif",    // NOLINT
"__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)",    // NOLINT
"{",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"const int global_z = get_global_id(2);",    // NOLINT
"int interleaved_y;",    // NOLINT
"int kernel_y;",    // NOLINT
"int kernel_idx;",    // NOLINT
"",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x16.  Thus each thread calculates 16*M rows x N cols of ctile.",    // NOLINT
"Dtype16  blockC00 = 0.f;",    // NOLINT
"Dtype16  blockC10 = 0.f;",    // NOLINT
"",    // NOLINT
"// Src0 (patch input) is directly used as atile.",    // NOLINT
"// Each work item points to the start of a different patch.",    // NOLINT
"// atile is M rows x K columns.",    // NOLINT
"int curr_x = ( global_y % output_width ) * STRIDE_X;",    // NOLINT
"int curr_y = ( global_y / output_width ) * STRIDE_Y;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"int saved_y = curr_y;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"const __global Dtype *src0_read = src0",    // NOLINT
"+ aligned_input_size * global_z                            // batch offset",    // NOLINT
"+ (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset",    // NOLINT
"+ curr_x - INPUT_PAD_W;                 // x offset",    // NOLINT
"const __global Dtype *src0_read_orig = src0_read;",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global Dtype *src1_read = src1 + ( global_x * TILE_N * 2 );",    // NOLINT
"",    // NOLINT
"#define DOT_PRODUCT_16( _result, _rowA, colB )        {           _result.s0 = mad( _rowA, sub_group_broadcast( colB,  0 ), _result.s0 );          _result.s1 = mad( _rowA, sub_group_broadcast( colB,  1 ), _result.s1 );          _result.s2 = mad( _rowA, sub_group_broadcast( colB,  2 ), _result.s2 );          _result.s3 = mad( _rowA, sub_group_broadcast( colB,  3 ), _result.s3 );          _result.s4 = mad( _rowA, sub_group_broadcast( colB,  4 ), _result.s4 );          _result.s5 = mad( _rowA, sub_group_broadcast( colB,  5 ), _result.s5 );          _result.s6 = mad( _rowA, sub_group_broadcast( colB,  6 ), _result.s6 );          _result.s7 = mad( _rowA, sub_group_broadcast( colB,  7 ), _result.s7 );          _result.s8 = mad( _rowA, sub_group_broadcast( colB,  8 ), _result.s8 );          _result.s9 = mad( _rowA, sub_group_broadcast( colB,  9 ), _result.s9 );          _result.sa = mad( _rowA, sub_group_broadcast( colB, 10 ), _result.sa );          _result.sb = mad( _rowA, sub_group_broadcast( colB, 11 ), _result.sb );          _result.sc = mad( _rowA, sub_group_broadcast( colB, 12 ), _result.sc );          _result.sd = mad( _rowA, sub_group_broadcast( colB, 13 ), _result.sd );          _result.se = mad( _rowA, sub_group_broadcast( colB, 14 ), _result.se );          _result.sf = mad( _rowA, sub_group_broadcast( colB, 15 ), _result.sf );      }",    // NOLINT
"typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;",    // NOLINT
"// Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.",    // NOLINT
"// Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch",    // NOLINT
"// and KERNEL_WIDTH/2 rows of interleaved filter.",    // NOLINT
"int patch_depth = 0;",    // NOLINT
"#ifndef __BEIGNET__",    // NOLINT
"__attribute__((opencl_unroll_hint(1)))",    // NOLINT
"#endif",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"int patch_row = 0;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0",    // NOLINT
"curr_y = saved_y;",    // NOLINT
"#endif",    // NOLINT
"#ifndef __BEIGNET__",    // NOLINT
"__attribute__((opencl_unroll_hint(1)))",    // NOLINT
"#endif",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"// Load atile and btile.",    // NOLINT
"// Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype16 granularity.",    // NOLINT
"// The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non",    // NOLINT
"// interleaved row is padded with zero to ensure same size as interleaved rows. This",    // NOLINT
"// interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the",    // NOLINT
"// kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.",    // NOLINT
"// (0, 0) (16, 0) (32, 0) (48, 0) ...     (0, 0) ( 0, 1) (16, 0) ( 0, 1) (32, 0) (0, 1) (48, 0) ...",    // NOLINT
"// (0, 1) (16, 1) (32, 1) (48, 1) ... =>  (0, 2) (16, 2) (32, 2) (48, 2) ...",    // NOLINT
"// (0, 2) (16, 2) (32, 2) (48, 2) ...     ...",    // NOLINT
"// ...",    // NOLINT
"const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;",    // NOLINT
"",    // NOLINT
"#if INPUT_PAD_W == 0 && INPUT_PAD_H == 0 && DILATION_X == 1 && DILATION_Y == 1",    // NOLINT
"Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read )[  0  ];",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"#else",    // NOLINT
"Dtype_t blockA00;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y >= INPUT_PAD_H && curr_y < input_height + INPUT_PAD_H && curr_x + pos * DILATION_X >= INPUT_PAD_W && curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y += DILATION_Y;",    // NOLINT
"#endif",    // NOLINT
"src0_read += ROW_PITCH * DILATION_X;",    // NOLINT
"uint blockB00[KERNEL_WIDTH * 2];",    // NOLINT
"uint4* p4BlockB00 = (uint4*)blockB00;",    // NOLINT
"uint2* p2BlockB00 = (uint2*)blockB00;",    // NOLINT
"Dtype* pBlockB00  = (Dtype*)blockB00;",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p4BlockB00[interleaved_y] = intel_sub_group_block_read4( (const __global uint*)src1_read );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p2BlockB00[KERNEL_WIDTH - 1] = intel_sub_group_block_read2( (const __global uint*)src1_read );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"kernel_idx = 0;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_16( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"//while( ++patch_row < 1 ); //debug",    // NOLINT
"while( ++patch_row < KERNEL_HEIGHT );",    // NOLINT
"",    // NOLINT
"src0_read += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch",    // NOLINT
"}",    // NOLINT
"//while ( ++patch_depth < 1 );  //debug",    // NOLINT
"while ( ++patch_depth < INPUT_DEPTH );",    // NOLINT
"",    // NOLINT
"// Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:",    // NOLINT
"// (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.",    // NOLINT
"__global Dtype *out = dst",    // NOLINT
"+ global_z * out_pitch_z                                                   // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                       // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset",    // NOLINT
"+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"Dtype bias[2];",    // NOLINT
"Dtype2 *bias_vec;",    // NOLINT
"bias_vec = (Dtype2*)bias;",    // NOLINT
"*bias_vec = as_float2(intel_sub_group_block_read2((__global uint *)biases + group_x * TILE_N));",    // NOLINT
"// Work around a potential compiler bug.",    // NOLINT
"if (group_x > 0xFFFFFFFEul)",    // NOLINT
"out[0] = bias[0] + bias[1];",    // NOLINT
"",    // NOLINT
"if (global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"#if ( ( OUT_DEPTH % TILE_N ) == 0 )",    // NOLINT
"for (int i = 0; i < 16; i++)",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"out[(16+i) * out_pitch_y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);;",    // NOLINT
"}",    // NOLINT
"#elif ( ( OUT_DEPTH % 16 ) == 0 )",    // NOLINT
"if ( ( global_x + 1 ) < get_global_size(0) )",    // NOLINT
"{",    // NOLINT
"for ( int i = 0; i < 16; i++ )",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);;",    // NOLINT
"out[(16+i) * out_pitch_y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < 16; i++)",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#else",    // NOLINT
"if ( ( global_x + 1 ) < get_global_size(0) )",    // NOLINT
"{",    // NOLINT
"for ( int i = 0; i < 16; i++ )",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);;",    // NOLINT
"out[(16+i) * out_pitch_y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"#if ( (OUT_DEPTH % TILE_N) > 16 )",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < 16 ; i++)",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);;",    // NOLINT
"}",    // NOLINT
"for (int i = 0; i < OUT_DEPTH % 16 ; i++)",    // NOLINT
"{",    // NOLINT
"out[(16+i) * out_pitch_y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#else",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < OUT_DEPTH % 16 ; i++)",    // NOLINT
"{",    // NOLINT
"out[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#ifdef GEMM_LIKE_CONV_32_2",    // NOLINT
"",    // NOLINT
"//////////////////////////////////////////////////////////////////////////////",    // NOLINT
"// Conv_Interleaved_32_2_flex",    // NOLINT
"//",    // NOLINT
"// Convolution: each workitem computes 1 patch x 32 filters worth of output",    // NOLINT
"// data.  Kernel's inner loop works on a single tile consisting of one",    // NOLINT
"// row from each patch and the filter data corresponding to that row.  Filter",    // NOLINT
"// matrix is interleaved to reduce GRF bank conflicts.  Patches are walked",    // NOLINT
"// by rows and then by slices.  Relies on sub_group extension for block",    // NOLINT
"// reads and SIMD broadcast.  Allows flexible sizing of TILE width (TILE_N)",    // NOLINT
"// by dynamically selecting one of two code paths: one uses TILE_N = 32 and",    // NOLINT
"// the other uses TILE_N = 8, 16, or 24.",    // NOLINT
"#define TILE_M          2",    // NOLINT
"#define TILE_K          KERNEL_WIDTH",    // NOLINT
"#define TILE_N          32",    // NOLINT
"",    // NOLINT
"#ifdef __BEIGNET__",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(8)))",    // NOLINT
"#endif",    // NOLINT
"__kernel void Conv_Interleaved(GEMM_LIKE_KERNEL_ARGS)",    // NOLINT
"{",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"const int global_z = get_global_id(2);",    // NOLINT
"int interleaved_y;",    // NOLINT
"int kernel_y;",    // NOLINT
"int kernel_idx;",    // NOLINT
"",    // NOLINT
"#define DOT_PRODUCT_8( _result, _rowA, colB )        {           _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );          _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );          _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );          _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );          _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );          _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );          _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );          _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );      }",    // NOLINT
"typedef CAT( float, KERNEL_WIDTH ) float_t;",    // NOLINT
"",    // NOLINT
"// True for all threads if filter_width is multiple of TILE_N",    // NOLINT
"// else, true for all but right-most column of threads.",    // NOLINT
"if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N )",    // NOLINT
"{",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"float8  blockC00 = 0.f;",    // NOLINT
"float8  blockC10 = 0.f;",    // NOLINT
"float8  blockC20 = 0.f;",    // NOLINT
"float8  blockC30 = 0.f;",    // NOLINT
"float8  blockC01 = 0.f;",    // NOLINT
"float8  blockC11 = 0.f;",    // NOLINT
"float8  blockC21 = 0.f;",    // NOLINT
"float8  blockC31 = 0.f;",    // NOLINT
"",    // NOLINT
"// Src0 (patch input) is directly used as atile.",    // NOLINT
"// Each work item points to the start of a different patch.",    // NOLINT
"// atile is M rows x K columns.",    // NOLINT
"int curr_x0 = ( ( global_y * TILE_M + 0 ) % output_width ) * STRIDE_X;",    // NOLINT
"int curr_x1 = ( ( global_y * TILE_M + 1 ) % output_width ) * STRIDE_X;",    // NOLINT
"int curr_y0 = ( ( global_y * TILE_M + 0 ) / output_width ) * STRIDE_Y;",    // NOLINT
"int curr_y1 = ( ( global_y * TILE_M + 1 ) / output_width ) * STRIDE_Y;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"int saved_y0 = curr_y0;",    // NOLINT
"int saved_y1 = curr_y1;",    // NOLINT
"#endif",    // NOLINT
"const __global float *src0_read0 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x0 - INPUT_PAD_W;                // x offset",    // NOLINT
"const __global float *src0_read1 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x1 - INPUT_PAD_W;                // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global float *src1_read = src1 + ( global_x * TILE_N * 2);",    // NOLINT
"",    // NOLINT
"// Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.",    // NOLINT
"// Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch",    // NOLINT
"// and KERNEL_WIDTH/2 rows of interleaved filter.",    // NOLINT
"int patch_depth = 0;",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"int patch_row = 0;",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"// Load atile and btile.",    // NOLINT
"// Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.",    // NOLINT
"// The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non",    // NOLINT
"// interleaved row is padded with zero to ensure same size as interleaved rows. This",    // NOLINT
"// interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the",    // NOLINT
"// kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.",    // NOLINT
"// (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..",    // NOLINT
"// (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...",    // NOLINT
"// (0, 2) (8, 2) (16, 2) (24, 2) ...       ...",    // NOLINT
"// ...",    // NOLINT
"const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;",    // NOLINT
"#if INPUT_PAD_H == 0 && INPUT_PAD_W == 0 && DILATION_X == 1 && DILATION_Y == 1",    // NOLINT
"float_t blockA00 = ( (const __global float_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;",    // NOLINT
"float_t blockA01 = ( (const __global float_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"float*  pblockA01 = (float*)(&blockA01);",    // NOLINT
"#else",    // NOLINT
"float_t blockA00;",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read0[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y0 += DILATION_Y;",    // NOLINT
"float_t blockA01;",    // NOLINT
"float*  pblockA01 = (float*)(&blockA01);",    // NOLINT
"pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y1 >= INPUT_PAD_H && curr_y1 < input_height + INPUT_PAD_H && curr_x1 + pos * DILATION_X >= INPUT_PAD_W && curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA01[pos] = src0_read1[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA01[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y1 += DILATION_Y;",    // NOLINT
"src0_read0 += ROW_PITCH * DILATION_Y;",    // NOLINT
"src0_read1 += ROW_PITCH * DILATION_Y;",    // NOLINT
"#endif",    // NOLINT
"float blockB00[KERNEL_WIDTH*4];",    // NOLINT
"float8* p8BlockB00 = (float8*)blockB00;",    // NOLINT
"float4* p4BlockB00 = (float4*)blockB00;",    // NOLINT
"float*  pBlockB00 =  (float* )blockB00;",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p8BlockB00[interleaved_y] = as_float8( intel_sub_group_block_read8( (const __global uint*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p4BlockB00[KERNEL_WIDTH - 1] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"kernel_idx = 0;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"//while( ++patch_row < 1 ); //debug",    // NOLINT
"while( ++patch_row < KERNEL_HEIGHT );",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"curr_y0 = saved_y0;",    // NOLINT
"curr_y1 = saved_y1;",    // NOLINT
"#endif",    // NOLINT
"src0_read0 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch",    // NOLINT
"src0_read1 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y );",    // NOLINT
"}",    // NOLINT
"//while ( ++patch_depth < 1 );  //debug",    // NOLINT
"while ( ++patch_depth < INPUT_DEPTH );",    // NOLINT
"",    // NOLINT
"// Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:",    // NOLINT
"// (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.",    // NOLINT
"__global float *out0 = dst",    // NOLINT
"+ global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"__global float *out1 = dst",    // NOLINT
"+ global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"float bias[4];",    // NOLINT
"float4 *bias_vec;",    // NOLINT
"bias_vec = (float4*)bias;",    // NOLINT
"*bias_vec = as_float4(intel_sub_group_block_read4((__global uint *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"if( global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"out0[( 0+i) * out_pitch_y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"out0[( 8+i) * out_pitch_y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);",    // NOLINT
"out0[(16+i) * out_pitch_y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);",    // NOLINT
"out0[(24+i) * out_pitch_y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if( global_y * TILE_M + 1 < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"out1[( 0+i) * out_pitch_y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"out1[( 8+i) * out_pitch_y] = blockC11[i] + intel_sub_group_shuffle(bias[1], i);",    // NOLINT
"out1[(16+i) * out_pitch_y] = blockC21[i] + intel_sub_group_shuffle(bias[2], i);",    // NOLINT
"out1[(24+i) * out_pitch_y] = blockC31[i] + intel_sub_group_shuffle(bias[3], i);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#if TILE_N_LAST > 0",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"int i = 0;",    // NOLINT
"float8  blockC0[TILE_N_LAST_DIV8];",    // NOLINT
"float8  blockC1[TILE_N_LAST_DIV8];",    // NOLINT
"LOOP(TILE_N_LAST_DIV8, i,",    // NOLINT
"{",    // NOLINT
"blockC0[i] = 0.f;",    // NOLINT
"blockC1[i] = 0.f;",    // NOLINT
"} )",    // NOLINT
"",    // NOLINT
"// Src0 (patch input) is directly used as atile.",    // NOLINT
"// Each work item points to the start of a different patch.",    // NOLINT
"// atile is M rows x K columns.",    // NOLINT
"int curr_x0 = ( ( global_y * TILE_M + 0 ) % output_width ) * STRIDE_X;",    // NOLINT
"int curr_x1 = ( ( global_y * TILE_M + 1 ) % output_width ) * STRIDE_X;",    // NOLINT
"int curr_y0 = ( ( global_y * TILE_M + 0 ) / output_width ) * STRIDE_Y;",    // NOLINT
"int curr_y1 = ( ( global_y * TILE_M + 1 ) / output_width ) * STRIDE_Y;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"int saved_y0 = curr_y0;",    // NOLINT
"int saved_y1 = curr_y1;",    // NOLINT
"#endif",    // NOLINT
"const __global float *src0_read0 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x0 - INPUT_PAD_W;                // x offset",    // NOLINT
"const __global float *src0_read1 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x1 - INPUT_PAD_W;                // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global float *src1_read = src1 + ( global_x * TILE_N  * 2);",    // NOLINT
"",    // NOLINT
"// Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.",    // NOLINT
"// Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch",    // NOLINT
"// and KERNEL_WIDTH/2 rows of interleaved filter.",    // NOLINT
"int patch_depth = 0;",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"int patch_row = 0;",    // NOLINT
"do",    // NOLINT
"{",    // NOLINT
"// Load atile and interleaved btile.",    // NOLINT
"const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;",    // NOLINT
"#if INPUT_PAD_H == 0 && INPUT_PAD_W == 0 && DILATION_X == 1 && DILATION_Y == 1",    // NOLINT
"float_t blockA00 = ( (const __global float_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;",    // NOLINT
"float_t blockA01 = ( (const __global float_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"float*  pblockA01 = (float*)(&blockA01);",    // NOLINT
"#else",    // NOLINT
"float_t blockA00;",    // NOLINT
"float*  pblockA00 = (float*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read0[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y0 += DILATION_Y;",    // NOLINT
"float_t blockA01;",    // NOLINT
"float*  pblockA01 = (float*)(&blockA01);",    // NOLINT
"pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y1 >= INPUT_PAD_H && curr_y1 < input_height + INPUT_PAD_H && curr_x1 + pos * DILATION_X >= INPUT_PAD_W && curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA01[pos] = src0_read1[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA01[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y1 += DILATION_Y;",    // NOLINT
"src0_read0 += (ROW_PITCH * DILATION_Y);",    // NOLINT
"src0_read1 += (ROW_PITCH * DILATION_Y);",    // NOLINT
"#endif",    // NOLINT
"float blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"float2* p2BlockB = (float2* )blockB;",    // NOLINT
"p2BlockB[interleaved_y] = as_float2( intel_sub_group_block_read2( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"float4* p4BlockB = (float4* )blockB;",    // NOLINT
"p4BlockB[interleaved_y] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"//TODO: broken.  No block_read6",    // NOLINT
"float6* p6BlockB = (float6* )blockB;",    // NOLINT
"(*((float8*)(&p6BlockB[interleaved_y]))).s0123 = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );",    // NOLINT
"(*((float8*)(&p6BlockB[interleaved_y]))).s45 = as_float2( intel_sub_group_block_read2( (const __global uint*)(src1_read + 4 * 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"float* pBlockB = (float* )blockB;",    // NOLINT
"pBlockB[KERNEL_WIDTH - 1] = as_float( intel_sub_group_block_read( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"float2* p2BlockB = (float2* )blockB;",    // NOLINT
"p2BlockB[KERNEL_WIDTH - 1] = as_float2( intel_sub_group_block_read2( (const __global uint*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"float3* p3BlockB = (float3* )blockB;",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s01 = as_float2( intel_sub_group_block_read2( (const __global uint*)src1_read ) );",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s2 = as_float( intel_sub_group_block_read( (const __global uint*) (src1_read + 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"float* pBlockB = (float*)blockB;",    // NOLINT
"kernel_idx = 0;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_8( blockC0[0], pblockA00[kernel_y    ], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC0[0], pblockA00[kernel_y + 1], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 2",    // NOLINT
"DOT_PRODUCT_8( blockC0[1], pblockA00[kernel_y    ], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC0[1], pblockA00[kernel_y + 1], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 3",    // NOLINT
"DOT_PRODUCT_8( blockC0[2], pblockA00[kernel_y    ], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y    ], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_8( blockC0[2], pblockA00[kernel_y + 1], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y + 1], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"} )",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"DOT_PRODUCT_8( blockC0[0], pblockA00[kernel_y], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 2",    // NOLINT
"DOT_PRODUCT_8( blockC0[1], pblockA00[kernel_y], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#if TILE_N_LAST_DIV8 >= 3",    // NOLINT
"DOT_PRODUCT_8( blockC0[2], pblockA00[kernel_y], pBlockB[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y], pBlockB[kernel_idx] ); kernel_idx++;",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"//while( ++patch_row < 1 ); //debug",    // NOLINT
"while( ++patch_row < KERNEL_HEIGHT );",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"curr_y0 = saved_y0;",    // NOLINT
"curr_y1 = saved_y1;",    // NOLINT
"#endif",    // NOLINT
"src0_read0 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y ); // reset to start of next slice of patch",    // NOLINT
"src0_read1 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y );",    // NOLINT
"}",    // NOLINT
"//while ( ++patch_depth < 1 );  //debug",    // NOLINT
"while ( ++patch_depth < INPUT_DEPTH );",    // NOLINT
"",    // NOLINT
"// Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:",    // NOLINT
"// (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.",    // NOLINT
"__global float *out0 = dst",    // NOLINT
"+ global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"__global float *out1 = dst",    // NOLINT
"+ global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"float bias[4];",    // NOLINT
"float4 *bias_vec;",    // NOLINT
"bias_vec = (float4*)bias;",    // NOLINT
"*bias_vec = as_float4(intel_sub_group_block_read4((__global uint *)biases + group_x * TILE_N));",    // NOLINT
"if( global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 0 ) out0[( 0+i) * out_pitch_y] = blockC0[0][i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 1 ) out0[( 8+i) * out_pitch_y] = blockC0[1][i] + intel_sub_group_shuffle(bias[1], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 2 ) out0[(16+i) * out_pitch_y] = blockC0[2][i] + intel_sub_group_shuffle(bias[2], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 3 ) out0[(24+i) * out_pitch_y] = blockC0[3][i] + intel_sub_group_shuffle(bias[3], i);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if( global_y * TILE_M + 1 < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 0 ) out1[( 0+i) * out_pitch_y] = blockC1[0][i] + intel_sub_group_shuffle(bias[0], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 1 ) out1[( 8+i) * out_pitch_y] = blockC1[1][i] + intel_sub_group_shuffle(bias[1], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 2 ) out1[(16+i) * out_pitch_y] = blockC1[2][i] + intel_sub_group_shuffle(bias[2], i);",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 3 ) out1[(24+i) * out_pitch_y] = blockC1[3][i] + intel_sub_group_shuffle(bias[3], i);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(copyImage, Dtype)",    // NOLINT
"(__global Dtype* image_data,",    // NOLINT
"int_tp image_offset,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp adjustedHeight, const int_tp adjustedWidth,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w,",    // NOLINT
"__global Dtype* output_image,",    // NOLINT
"const int_tp output_offset,",    // NOLINT
"const int_tp batch_size) {",    // NOLINT
"",    // NOLINT
"uint_tp sX = get_global_id(0);",    // NOLINT
"uint_tp sY = get_global_id(1);",    // NOLINT
"uint_tp sZ = get_global_id(2);",    // NOLINT
"",    // NOLINT
"int_tp in_y = sY - pad_h;",    // NOLINT
"int_tp in_x = sX - pad_w;",    // NOLINT
"",    // NOLINT
"int_tp batch_offset = 0;",    // NOLINT
"int_tp adjusted_batch_offset = 0;",    // NOLINT
"for(uint_tp batch_idx = 0; batch_idx < batch_size; batch_idx++) {",    // NOLINT
"int_tp dst_offset = adjusted_batch_offset + output_offset + sZ*adjustedHeight*adjustedWidth + sY*adjustedWidth +sX;",    // NOLINT
"int_tp src_offset = batch_offset + image_offset + sZ*height*width + in_y*width + in_x;",    // NOLINT
"if((in_y >= 0 && in_y < height && in_x >= 0 && in_x < width))",    // NOLINT
"output_image[dst_offset] = image_data[src_offset];",    // NOLINT
"else",    // NOLINT
"output_image[dst_offset] = 0;",    // NOLINT
"batch_offset += height * width * channels;",    // NOLINT
"adjusted_batch_offset += adjustedHeight * adjustedWidth * channels;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(copyWeightsSwizzled, Dtype)",    // NOLINT
"(__global Dtype* weightIn,",    // NOLINT
"__global Dtype* weightOut,",    // NOLINT
"const int_tp kernel_w,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp outputs,",    // NOLINT
"const int_tp swizzleFactor) {",    // NOLINT
"",    // NOLINT
"uint_tp sX = get_global_id(0);",    // NOLINT
"",    // NOLINT
"//Original location",    // NOLINT
"",    // NOLINT
"//Output location",    // NOLINT
"int_tp outputSublayer = channels / swizzleFactor;",    // NOLINT
"int_tp outputSublayerIndex = channels % swizzleFactor;",    // NOLINT
"",    // NOLINT
"int_tp filter = sX / (kernel_w*kernel_h*channels);",    // NOLINT
"int_tp kernel_X = sX % kernel_w;",    // NOLINT
"int_tp kernel_Y = (sX / kernel_w) % kernel_h;",    // NOLINT
"int_tp kernel_C = (sX / (kernel_w * kernel_h)) % channels;",    // NOLINT
"",    // NOLINT
"int_tp FP = filter / swizzleFactor;",    // NOLINT
"int_tp F1 = filter % swizzleFactor;",    // NOLINT
"",    // NOLINT
"weightOut[FP*(kernel_w*kernel_h*channels*swizzleFactor) + kernel_C*(kernel_w*kernel_h*swizzleFactor) + kernel_Y*(kernel_w*swizzleFactor) + kernel_X*swizzleFactor + F1]",    // NOLINT
"= weightIn[filter*(kernel_w*kernel_h*channels) + kernel_C*(kernel_w*kernel_h) + kernel_Y*kernel_w + kernel_X];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(crop_copy, Dtype)(const int_tp n,",    // NOLINT
"const int_tp height,",    // NOLINT
"const int_tp width,",    // NOLINT
"const int_tp src_inner_stride,",    // NOLINT
"const int_tp dest_inner_stride,",    // NOLINT
"__global const Dtype* src,",    // NOLINT
"const int_tp src_off,",    // NOLINT
"__global Dtype* dest,",    // NOLINT
"const int_tp dest_off) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp src_start = index * src_inner_stride + src_off;",    // NOLINT
"int_tp dest_start = index * dest_inner_stride + dest_off;",    // NOLINT
"for (int_tp i = 0; i < width; ++i) {",    // NOLINT
"dest[dest_start + i] = src[src_start + i];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(dropout_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const uint_tp* mask,",    // NOLINT
"const uint_tp threshold,",    // NOLINT
"const Dtype scale,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] * ((mask[index] > threshold)?1.0:0.0) * scale;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(dropout_backward,Dtype)(",    // NOLINT
"const int_tp n, __global const Dtype* in_diff,",    // NOLINT
"__global const uint_tp* mask, const uint_tp threshold,",    // NOLINT
"const Dtype scale,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out_diff[index] = in_diff[index] * ((mask[index] > threshold)?1.0:0.0) * scale;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(eltwise_max_forward,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data_a,",    // NOLINT
"__global const Dtype* bottom_data_b, const int_tp blob_idx,",    // NOLINT
"__global Dtype* top_data,",    // NOLINT
"__global int_tp* mask) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"Dtype maxval = -FLT_MAX;",    // NOLINT
"int_tp maxidx = -1;",    // NOLINT
"if (bottom_data_a[index] > bottom_data_b[index]) {",    // NOLINT
"// only update for very first bottom_data blob (blob_idx == 0)",    // NOLINT
"if (blob_idx == 0) {",    // NOLINT
"maxval = bottom_data_a[index];",    // NOLINT
"top_data[index] = maxval;",    // NOLINT
"maxidx = blob_idx;",    // NOLINT
"mask[index] = maxidx;",    // NOLINT
"}",    // NOLINT
"} else {",    // NOLINT
"maxval = bottom_data_b[index];",    // NOLINT
"top_data[index] = maxval;",    // NOLINT
"maxidx = blob_idx + 1;",    // NOLINT
"mask[index] = maxidx;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(eltwise_max_backward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* top_diff,",    // NOLINT
"const int_tp blob_idx,",    // NOLINT
"__global const int_tp* mask,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"Dtype gradient = 0;",    // NOLINT
"if (mask[index] == blob_idx) {",    // NOLINT
"gradient += top_diff[index];",    // NOLINT
"}",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(elu_forward,Dtype)(const int n, __global const Dtype* in,",    // NOLINT
"__global Dtype* out,",    // NOLINT
"Dtype alpha) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] > 0 ? in[index] : alpha * (exp(in[index]) - 1.0);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(elu_backward,Dtype)(const int n, __global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* out_data,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff,",    // NOLINT
"Dtype alpha) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out_diff[index] =",    // NOLINT
"in_data[index] > 0 ?",    // NOLINT
"in_diff[index] : in_diff[index] * (out_data[index] + alpha);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(embed_forward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* bottom_data,",    // NOLINT
"__global const Dtype* weight,",    // NOLINT
"const int_tp M, const int_tp N,",    // NOLINT
"const int_tp K,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"for (int_tp top_index = get_global_id(0); top_index < nthreads;",    // NOLINT
"top_index += get_global_size(0)) {",    // NOLINT
"const int_tp n = top_index / N;",    // NOLINT
"const int_tp d = top_index % N;",    // NOLINT
"const int_tp index = (int_tp)(bottom_data[n]);",    // NOLINT
"const int_tp weight_index = index * N + d;",    // NOLINT
"top_data[top_index] = weight[weight_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// atomic_add from: http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html",    // NOLINT
"#if (TYPE == TYPE_FLOAT)",    // NOLINT
"#ifdef ATOMICS_32_AVAILABLE",    // NOLINT
"inline void TEMPLATE(atomic_add,Dtype)(volatile __global Dtype *source, const Dtype operand) {",    // NOLINT
"union {",    // NOLINT
"uint_tp intVal;",    // NOLINT
"Dtype floatVal;",    // NOLINT
"} newVal;",    // NOLINT
"union {",    // NOLINT
"uint_tp intVal;",    // NOLINT
"Dtype floatVal;",    // NOLINT
"} prevVal;",    // NOLINT
"do {",    // NOLINT
"prevVal.floatVal = *source;",    // NOLINT
"newVal.floatVal = prevVal.floatVal + operand;",    // NOLINT
"} while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(embed_backward,Dtype)(const int_tp nthreads, __global const Dtype* bottom_data,",    // NOLINT
"__global const Dtype* top_diff, const int_tp M, const int_tp N, const int_tp K,",    // NOLINT
"__global Dtype* weight_diff) {",    // NOLINT
"for (int_tp top_index = get_global_id(0); top_index < nthreads;",    // NOLINT
"top_index += get_global_size(0)) {",    // NOLINT
"const int_tp n = top_index / N;",    // NOLINT
"const int_tp d = top_index % N;",    // NOLINT
"const int_tp index = (int_tp)(bottom_data[n]);",    // NOLINT
"const int_tp weight_index = index * N + d;",    // NOLINT
"",    // NOLINT
"TEMPLATE(atomic_add,Dtype)((weight_diff + weight_index), *(top_diff + top_index));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if (TYPE == TYPE_DOUBLE)",    // NOLINT
"#ifdef ATOMICS_64_AVAILABLE",    // NOLINT
"inline void TEMPLATE(atomic_add,Dtype)(volatile __global Dtype *source, const Dtype operand) {",    // NOLINT
"union {",    // NOLINT
"unsigned long intVal;",    // NOLINT
"Dtype floatVal;",    // NOLINT
"} newVal;",    // NOLINT
"union {",    // NOLINT
"unsigned long intVal;",    // NOLINT
"Dtype floatVal;",    // NOLINT
"} prevVal;",    // NOLINT
"do {",    // NOLINT
"prevVal.floatVal = *source;",    // NOLINT
"newVal.floatVal = prevVal.floatVal + operand;",    // NOLINT
"} while (atom_cmpxchg((volatile __global unsigned long *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(embed_backward,Dtype)(const int_tp nthreads, __global const Dtype* bottom_data,",    // NOLINT
"__global const Dtype* top_diff, const int_tp M, const int_tp N, const int_tp K,",    // NOLINT
"__global Dtype* weight_diff) {",    // NOLINT
"for (int_tp top_index = get_global_id(0); top_index < nthreads;",    // NOLINT
"top_index += get_global_size(0)) {",    // NOLINT
"const int_tp n = top_index / N;",    // NOLINT
"const int_tp d = top_index % N;",    // NOLINT
"const int_tp index = (int_tp)(bottom_data[n]);",    // NOLINT
"const int_tp weight_index = index * N + d;",    // NOLINT
"",    // NOLINT
"TEMPLATE(atomic_add,Dtype)((weight_diff + weight_index), *(top_diff + top_index));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(fft_phony,Dtype)(Dtype arg) {",    // NOLINT
"Dtype out = arg;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#ifdef FFT",    // NOLINT
"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"#define DtypeComplex Dtype2",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_cyclic_shift_in,Dtype)(",    // NOLINT
"__global Dtype* fft_gpu_weights_real, const int_tp offset_fft_gpu_weights_real,",    // NOLINT
"__global Dtype* weight, const int_tp offset_weight,",    // NOLINT
"const int_tp ker_size, const int_tp ch_gr, const int_tp ker_size_ch_gr,",    // NOLINT
"const int_tp ker_w, const int_tp ker_c_h, const int_tp ker_c_w,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width, const int_tp complex_w_len) {",    // NOLINT
"fft_gpu_weights_real += offset_fft_gpu_weights_real;",    // NOLINT
"weight += offset_weight;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = gId / ker_size_ch_gr;",    // NOLINT
"int_tp c = (gId - out * ker_size_ch_gr) / ker_size;",    // NOLINT
"int_tp map_offset = out * ch_gr + c;",    // NOLINT
"int_tp map_offset_ker_size = map_offset * ker_size;",    // NOLINT
"int_tp pos_in_map = gId - map_offset_ker_size;",    // NOLINT
"int_tp h = pos_in_map / ker_w;",    // NOLINT
"int_tp h_ker_w = h * ker_w;",    // NOLINT
"int_tp w = pos_in_map - h_ker_w;",    // NOLINT
"int_tp src_idx = map_offset_ker_size + h_ker_w + w;",    // NOLINT
"int_tp ky = h - ker_c_h;",    // NOLINT
"if (ky < 0) ky += fft_height;",    // NOLINT
"int_tp kx = w - ker_c_w;",    // NOLINT
"if (kx < 0) kx += fft_width;",    // NOLINT
"int_tp dst_idx = (map_offset * fft_height + ky) * complex_w_len + kx;",    // NOLINT
"fft_gpu_weights_real[dst_idx] = weight[src_idx];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width < 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_in_naive,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp size,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp h = gId / width;",    // NOLINT
"int_tp w = gId - (h * width);",    // NOLINT
"int_tp dst_idx = (h*stride_h + pad_h)*width_out + (w*stride_w + pad_w);",    // NOLINT
"map_out[dst_idx] = map_in[gId];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width < 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_in_naive_2d,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp map_out_size, const int_tp size, const int_tp count,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId_x = get_global_id(0);",    // NOLINT
"int_tp gId_y = get_global_id(1);",    // NOLINT
"int_tp h = gId_x / width;",    // NOLINT
"int_tp w = gId_x - (h * width);",    // NOLINT
"int_tp src_idx = gId_y * size + gId_x;",    // NOLINT
"int_tp dst_idx = gId_y * map_out_size +",    // NOLINT
"(h * stride_h + pad_h) * width_out + (w * stride_w + pad_w);",    // NOLINT
"map_out[dst_idx] = map_in[src_idx];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width >= 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_in,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp size,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp count = size >> 2;",    // NOLINT
"int_tp gId4 = gId << 2;",    // NOLINT
"int_tp h = gId4 / width;",    // NOLINT
"int_tp w = gId4 - (h * width);",    // NOLINT
"int_tp dst_h = h*stride_h + pad_h;",    // NOLINT
"int_tp dst_w = w*stride_w + pad_w;",    // NOLINT
"int_tp dst_idx = dst_h*width_out + dst_w;",    // NOLINT
"if (gId < count) {",    // NOLINT
"Dtype4 map_in_cache4 = vload4(gId, map_in);",    // NOLINT
"int_tp has_pad = width - dst_w;",    // NOLINT
"if (has_pad >= 4) {",    // NOLINT
"vstore4(map_in_cache4, dst_idx >> 2, map_out);",    // NOLINT
"} else {",    // NOLINT
"if (0 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx] = map_in_cache4.x;",    // NOLINT
"if (1 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 1;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx+1] = map_in_cache4.y;",    // NOLINT
"if (2 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 2;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx+2] = map_in_cache4.z;",    // NOLINT
"if (3 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 3;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx+3] = map_in_cache4.w;",    // NOLINT
"dst_h += 1;",    // NOLINT
"dst_w = pad_w;",    // NOLINT
"}",    // NOLINT
"} else if (gId == count) {",    // NOLINT
"int_tp res = size - (count << 2); /* size % 4 */",    // NOLINT
"if (res > 0) {",    // NOLINT
"Dtype4 map_in_cache4 = 0.f;",    // NOLINT
"if (res >= 1)",    // NOLINT
"map_in_cache4.x = map_in[gId4];",    // NOLINT
"if (res >= 2)",    // NOLINT
"map_in_cache4.y = map_in[gId4+1];",    // NOLINT
"if (res == 3)",    // NOLINT
"map_in_cache4.z = map_in[gId4+2];",    // NOLINT
"int_tp has_pad = width - dst_w;",    // NOLINT
"if (has_pad >= 4) {",    // NOLINT
"vstore4(map_in_cache4, dst_idx >> 2, map_out);",    // NOLINT
"} else {",    // NOLINT
"if (0 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx] = map_in_cache4.x;",    // NOLINT
"if (1 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 1;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx+1] = map_in_cache4.y;",    // NOLINT
"if (2 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 2;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx+2] = map_in_cache4.z;",    // NOLINT
"if (3 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 3;",    // NOLINT
"}",    // NOLINT
"map_out[dst_idx+3] = map_in_cache4.w;",    // NOLINT
"dst_h += 1;",    // NOLINT
"dst_w = pad_w;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width >= 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_in_2d,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp map_out_size, const int_tp size, const int_tp count,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp gId_y = get_global_id(1);",    // NOLINT
"int_tp gId4 = gId << 2;",    // NOLINT
"int_tp h = gId4 / width;",    // NOLINT
"int_tp w = gId4 - (h * width);",    // NOLINT
"int_tp dst_h = h*stride_h + pad_h;",    // NOLINT
"int_tp dst_w = w*stride_w + pad_w;",    // NOLINT
"int_tp dst_idx = dst_h*width_out + dst_w;",    // NOLINT
"const __global Dtype* map_in_2d = map_in + gId_y * size;",    // NOLINT
"__global Dtype* map_out_2d = map_out + gId_y * map_out_size;",    // NOLINT
"if (gId < count) {",    // NOLINT
"Dtype4 map_in_cache4 = vload4(gId, map_in_2d);",    // NOLINT
"int_tp has_pad = width - dst_w;",    // NOLINT
"if (has_pad >= 4) {",    // NOLINT
"vstore4(map_in_cache4, dst_idx >> 2, map_out_2d);",    // NOLINT
"} else {",    // NOLINT
"if (0 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx] = map_in_cache4.x;",    // NOLINT
"if (1 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 1;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx+1] = map_in_cache4.y;",    // NOLINT
"if (2 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 2;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx+2] = map_in_cache4.z;",    // NOLINT
"if (3 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 3;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx+3] = map_in_cache4.w;",    // NOLINT
"dst_h += 1;",    // NOLINT
"dst_w = pad_w;",    // NOLINT
"}",    // NOLINT
"} else if (gId == count) {",    // NOLINT
"int_tp res = size - (count << 2); /* size % 4 */",    // NOLINT
"if (res > 0) {",    // NOLINT
"Dtype4 map_in_cache4 = 0.f;",    // NOLINT
"if (res >= 1)",    // NOLINT
"map_in_cache4.x = map_in_2d[gId4];",    // NOLINT
"if (res >= 2)",    // NOLINT
"map_in_cache4.y = map_in_2d[gId4+1];",    // NOLINT
"if (res == 3)",    // NOLINT
"map_in_cache4.z = map_in_2d[gId4+2];",    // NOLINT
"int_tp has_pad = width - dst_w;",    // NOLINT
"if (has_pad >= 4) {",    // NOLINT
"vstore4(map_in_cache4, dst_idx >> 2, map_out_2d);",    // NOLINT
"} else {",    // NOLINT
"if (0 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx] = map_in_cache4.x;",    // NOLINT
"if (1 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 1;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx+1] = map_in_cache4.y;",    // NOLINT
"if (2 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 2;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx+2] = map_in_cache4.z;",    // NOLINT
"if (3 == has_pad) {",    // NOLINT
"dst_idx += width_out + pad_w - dst_w - 3;",    // NOLINT
"}",    // NOLINT
"map_out_2d[dst_idx+3] = map_in_cache4.w;",    // NOLINT
"dst_h += 1;",    // NOLINT
"dst_w = pad_w;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width_out < 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_out_naive,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp size,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width,",    // NOLINT
"const int_tp ker_center_h, const int_tp ker_center_w,",    // NOLINT
"const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp h_out = gId / width_out;",    // NOLINT
"int_tp w_out = gId - (h_out * width_out);",    // NOLINT
"int_tp h = h_out * stride_h + ker_center_h;",    // NOLINT
"int_tp w = w_out * stride_w + ker_center_w;",    // NOLINT
"int_tp src_idx = h*fft_width + w;",    // NOLINT
"map_out[gId] = map_in[src_idx];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width_out < 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_out_naive_2d,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp size, const int_tp count, const int_tp map_in_size,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width,",    // NOLINT
"const int_tp ker_center_h, const int_tp ker_center_w,",    // NOLINT
"const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp h_out = gId / width_out;",    // NOLINT
"int_tp w_out = gId - (h_out * width_out);",    // NOLINT
"int_tp h = h_out * stride_h + ker_center_h;",    // NOLINT
"int_tp w = w_out * stride_w + ker_center_w;",    // NOLINT
"int_tp src_idx = out * map_in_size + h*fft_width + w;",    // NOLINT
"int_tp dst_idx = out * size + gId;",    // NOLINT
"map_out[dst_idx] = map_in[src_idx];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width_out >= 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_out,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp size,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width,",    // NOLINT
"const int_tp ker_c_h, const int_tp ker_c_w,",    // NOLINT
"const int_tp stride_h, const int_tp stride_w, const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp count = size >> 2;",    // NOLINT
"int_tp gId4 = gId << 2;",    // NOLINT
"int_tp h_out = gId4 / width_out;",    // NOLINT
"int_tp w_out = gId4 - (h_out * width_out);",    // NOLINT
"int_tp h = h_out * stride_h + ker_c_h;",    // NOLINT
"int_tp w = w_out * stride_w + ker_c_w;",    // NOLINT
"int_tp src_idx = h*fft_width + w;",    // NOLINT
"if (gId < count) {",    // NOLINT
"Dtype4 map_in_cache4;",    // NOLINT
"int_tp has_pad = width_out - (w - pad_w);",    // NOLINT
"if (has_pad >= 4) {",    // NOLINT
"map_in_cache4 = vload4(src_idx >> 2, map_in);",    // NOLINT
"} else {",    // NOLINT
"int_tp right_elements = fft_width - width_out;",    // NOLINT
"if (0 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.x = map_in[src_idx];",    // NOLINT
"if (1 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.y = map_in[src_idx+1];",    // NOLINT
"if (2 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.z = map_in[src_idx+2];",    // NOLINT
"if (3 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.w = map_in[src_idx+3];",    // NOLINT
"}",    // NOLINT
"vstore4(map_in_cache4, gId, map_out);",    // NOLINT
"} else if (gId == count) {",    // NOLINT
"int_tp res = size - (count << 2); /* size % 4 */",    // NOLINT
"if (res > 0) {",    // NOLINT
"for (int_tp i = gId4; i < size; ++i) {",    // NOLINT
"map_out[i] = map_in[src_idx];",    // NOLINT
"src_idx++;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Use when width_out >= 4 */",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_left_top_out_2d,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp size, const int_tp count, const int_tp map_in_size,",    // NOLINT
"const int_tp height_out, const int_tp width_out,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width,",    // NOLINT
"const int_tp ker_c_h, const int_tp ker_c_w,",    // NOLINT
"const int_tp stride_h, const int_tp stride_w, const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp gId4 = gId << 2;",    // NOLINT
"int_tp h_out = gId4 / width_out;",    // NOLINT
"int_tp w_out = gId4 - (h_out * width_out);",    // NOLINT
"int_tp h = h_out * stride_h + ker_c_h;",    // NOLINT
"int_tp w = w_out * stride_w + ker_c_w;",    // NOLINT
"int_tp src_idx = h*fft_width + w;",    // NOLINT
"const __global Dtype* map_in_2d = map_in + out * map_in_size;",    // NOLINT
"__global Dtype* map_out_2d = map_out + out * size;",    // NOLINT
"if (gId < count) {",    // NOLINT
"Dtype4 map_in_cache4;",    // NOLINT
"int_tp has_pad = width_out - (w - pad_w);",    // NOLINT
"if (has_pad >= 4) {",    // NOLINT
"map_in_cache4 = vload4(src_idx >> 2, map_in_2d);",    // NOLINT
"} else {",    // NOLINT
"int_tp right_elements = fft_width - width_out;",    // NOLINT
"if (0 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.x = map_in_2d[src_idx];",    // NOLINT
"if (1 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.y = map_in_2d[src_idx+1];",    // NOLINT
"if (2 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.z = map_in_2d[src_idx+2];",    // NOLINT
"if (3 == has_pad) {",    // NOLINT
"src_idx += right_elements;",    // NOLINT
"}",    // NOLINT
"map_in_cache4.w = map_in_2d[src_idx+3];",    // NOLINT
"}",    // NOLINT
"vstore4(map_in_cache4, gId, map_out_2d);",    // NOLINT
"} else if (gId == count) {",    // NOLINT
"int_tp res = size - (count << 2); /* size % 4 */",    // NOLINT
"if (res > 0) {",    // NOLINT
"const __global Dtype4* map_in_2d_4 =",    // NOLINT
"(const __global Dtype4*)(map_in_2d + src_idx);",    // NOLINT
"__global Dtype4* map_out_2d_4 = (__global Dtype4*)(map_out_2d + gId4);",    // NOLINT
"if (res == 3) {",    // NOLINT
"map_out_2d_4[0].xyz = map_in_2d_4[0].xyz;",    // NOLINT
"} else if (res == 2) {",    // NOLINT
"map_out_2d_4[0].xy = map_in_2d_4[0].xy;",    // NOLINT
"} else if (res == 1) {",    // NOLINT
"map_out_2d_4[0].x = map_in_2d_4[0].x;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_cyclic_shift_out,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp width_out,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width,",    // NOLINT
"const int_tp ker_center_h, const int_tp ker_center_w,",    // NOLINT
"const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp h_out = gId / width_out;",    // NOLINT
"int_tp w_out = gId - (h_out * width_out);",    // NOLINT
"int_tp h = h_out * stride_h + pad_h;",    // NOLINT
"int_tp w = w_out * stride_w + pad_w;",    // NOLINT
"int_tp ky = h - ker_center_h;",    // NOLINT
"if (ky < 0) ky += fft_height;",    // NOLINT
"int_tp kx = w - ker_center_w;",    // NOLINT
"if (kx < 0) kx += fft_width;",    // NOLINT
"int_tp src_idx = ky*fft_width + kx;",    // NOLINT
"map_out[gId] = map_in[src_idx];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(copy2buffer_cyclic_shift_out_2d,Dtype)(__global Dtype* map_out,",    // NOLINT
"const int_tp offset_map_out,",    // NOLINT
"const __global Dtype* map_in, const int_tp offset_map_in,",    // NOLINT
"const int_tp map_out_size, const int_tp map_in_size,",    // NOLINT
"const int_tp width_out,",    // NOLINT
"const int_tp fft_height, const int_tp fft_width,",    // NOLINT
"const int_tp ker_center_h, const int_tp ker_center_w,",    // NOLINT
"const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp pad_h, const int_tp pad_w) {",    // NOLINT
"map_out += offset_map_out;",    // NOLINT
"map_in  += offset_map_in;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp gId_y = get_global_id(1);",    // NOLINT
"int_tp h_out = gId / width_out;",    // NOLINT
"int_tp w_out = gId - (h_out * width_out);",    // NOLINT
"int_tp h = h_out * stride_h + pad_h;",    // NOLINT
"int_tp w = w_out * stride_w + pad_w;",    // NOLINT
"int_tp ky = h - ker_center_h;",    // NOLINT
"if (ky < 0) ky += fft_height;",    // NOLINT
"int_tp kx = w - ker_center_w;",    // NOLINT
"if (kx < 0) kx += fft_width;",    // NOLINT
"int_tp src_idx = gId_y * map_in_size + ky*fft_width + kx;",    // NOLINT
"int_tp dst_idx = gId_y * map_out_size + gId;",    // NOLINT
"map_out[dst_idx] = map_in[src_idx];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_conjugate_multiplication_1d,Dtype)(__global Dtype* dst,",    // NOLINT
"const int_tp offset_dst,",    // NOLINT
"const __global Dtype* src1, const int_tp offset_src1,",    // NOLINT
"const __global Dtype* src2, const int_tp offset_src2,",    // NOLINT
"const int_tp ch_gr) {",    // NOLINT
"dst += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp size = get_global_size(0);",    // NOLINT
"Dtype4 dst_cache = 0.f;",    // NOLINT
"int_tp src_idx;",    // NOLINT
"Dtype4 s1_cache;",    // NOLINT
"Dtype4 s2_cache;",    // NOLINT
"for (int_tp c = 0; c < ch_gr; ++c) {",    // NOLINT
"src_idx = size * c + gId;",    // NOLINT
"s1_cache = vload4(src_idx, src1);",    // NOLINT
"s2_cache = vload4(src_idx, src2);",    // NOLINT
"dst_cache.x +=  s1_cache.x * s2_cache.x + s1_cache.y * s2_cache.y;",    // NOLINT
"dst_cache.y += -s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;",    // NOLINT
"dst_cache.z +=  s1_cache.z * s2_cache.z + s1_cache.w * s2_cache.w;",    // NOLINT
"dst_cache.w += -s1_cache.z * s2_cache.w + s1_cache.w * s2_cache.z;",    // NOLINT
"}",    // NOLINT
"((__global Dtype4*)(&dst[gId<<2]))[0] += dst_cache;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_conjugate_multiplication_2d,Dtype)(__global Dtype* dst,",    // NOLINT
"const int_tp offset_dst,",    // NOLINT
"const __global Dtype* src1, const int_tp offset_src1,",    // NOLINT
"const __global Dtype* src2, const int_tp offset_src2,",    // NOLINT
"const int_tp out_gr, const int_tp map_size, const int_tp ch_gr) {",    // NOLINT
"dst += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp src1_idx, src2_idx;",    // NOLINT
"int_tp dst_map_offset = map_size * out;",    // NOLINT
"int_tp dst_idx = dst_map_offset + gId;",    // NOLINT
"Dtype4 s1_cache, s2_cache;",    // NOLINT
"Dtype4 dst_cache = 0.f;",    // NOLINT
"int_tp map_offset = dst_map_offset * ch_gr;",    // NOLINT
"for (int_tp i = 0; i < ch_gr; ++i) {",    // NOLINT
"src1_idx = map_size * i + gId;",    // NOLINT
"src2_idx = map_offset + src1_idx;",    // NOLINT
"s1_cache = vload4(src1_idx, src1);",    // NOLINT
"s2_cache = vload4(src2_idx, src2);",    // NOLINT
"dst_cache.xz += mad( s1_cache.xz, s2_cache.xz, s1_cache.yw * s2_cache.yw);",    // NOLINT
"dst_cache.yw += mad(-s1_cache.xz, s2_cache.yw, s1_cache.yw * s2_cache.xz);",    // NOLINT
"}",    // NOLINT
"vstore4(dst_cache, dst_idx, dst);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_conjugate_multiplication_2d_SLM,Dtype)(",    // NOLINT
"__global Dtype* restrict dst, const int_tp offset_dst,",    // NOLINT
"const __global Dtype* restrict src1, const int_tp offset_src1,",    // NOLINT
"__local Dtype* local_src1,",    // NOLINT
"const __global Dtype* restrict src2, const int_tp offset_src2,",    // NOLINT
"const int_tp out_gr, const int_tp map_size, const int_tp ch_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"if (gId >= map_size) return; /* Do not remove this */",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"if (out >= out_gr) return;   /* Do not remove this */",    // NOLINT
"dst += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp tId = get_local_id(0);",    // NOLINT
"int_tp local_out = get_local_id(1);",    // NOLINT
"int_tp tile_size = get_local_size(0);",    // NOLINT
"Dtype4 s1_cache;",    // NOLINT
"if (local_out == 0) {",    // NOLINT
"for (int_tp c = 0; c < ch_gr; ++c) {",    // NOLINT
"s1_cache = vload4(map_size * c + gId, src1);",    // NOLINT
"vstore4(s1_cache, tile_size * c + tId, local_src1);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"int_tp dst_map_offset = map_size * out;",    // NOLINT
"int_tp dst_idx = (dst_map_offset + gId) << 2;",    // NOLINT
"Dtype4 dst_cache = 0.f;",    // NOLINT
"Dtype4 s2_cache;",    // NOLINT
"int_tp ch_offset = 0;",    // NOLINT
"int_tp map_offset = dst_map_offset * ch_gr;",    // NOLINT
"for (int_tp c = 0; c < ch_gr; ++c) {",    // NOLINT
"ch_offset = map_size * c;",    // NOLINT
"s1_cache = vload4(tile_size * c + tId, local_src1);",    // NOLINT
"s2_cache = vload4(map_offset + ch_offset + gId, src2);",    // NOLINT
"dst_cache.xz += mad( s1_cache.xz, s2_cache.xz, s1_cache.yw * s2_cache.yw);",    // NOLINT
"dst_cache.yw += mad(-s1_cache.xz, s2_cache.yw, s1_cache.yw * s2_cache.xz);",    // NOLINT
"}",    // NOLINT
"((__global Dtype4*)(&dst[dst_idx]))[0] += dst_cache;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_conjugate_multiplication_3d,Dtype)(__global Dtype* dst,",    // NOLINT
"const int_tp offset_dst,",    // NOLINT
"const __global Dtype* src1, const int_tp offset_src1,",    // NOLINT
"const __global Dtype* src2, const int_tp offset_src2,",    // NOLINT
"const int_tp out_gr, const int_tp size, const int_tp ch_gr) {",    // NOLINT
"dst  += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp ch  = get_global_id(2);",    // NOLINT
"Dtype4 dst_cache = 0.f;",    // NOLINT
"Dtype4 s1_cache  = ((__global Dtype4*)(&(src1[(size*ch+gId)<<2])))[0];",    // NOLINT
"Dtype4 s2_cache  = ((__global Dtype4*)(&(src2[(size*(out*ch_gr+ch)+gId)<<2])))[0];",    // NOLINT
"dst_cache.x =  s1_cache.x * s2_cache.x + s1_cache.y * s2_cache.y;",    // NOLINT
"dst_cache.y = -s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;",    // NOLINT
"dst_cache.z =  s1_cache.z * s2_cache.z + s1_cache.w * s2_cache.w;",    // NOLINT
"dst_cache.w = -s1_cache.z * s2_cache.w + s1_cache.w * s2_cache.z;",    // NOLINT
"((__global Dtype4*)(&dst[(size*out+gId)<<2]))[0] += dst_cache;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_conjugate_multiplication_3d_SLM,Dtype)(__global Dtype* dst,",    // NOLINT
"const int_tp offset_dst, __local Dtype* local_dst,",    // NOLINT
"const __global Dtype* src1, const int_tp offset_src1,",    // NOLINT
"__local Dtype* local_src1, const __global Dtype* src2,",    // NOLINT
"const int_tp offset_src2, const int_tp out_gr, const int_tp map_size,",    // NOLINT
"const int_tp ch_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"if (gId >= map_size) return; /* Do not remove this */",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"if (out >= out_gr) return;   /* Do not remove this */",    // NOLINT
"int_tp ch = get_global_id(2);",    // NOLINT
"if (ch >= ch_gr) return;     /* Do not remove this */",    // NOLINT
"dst += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp tId = get_local_id(0);",    // NOLINT
"int_tp local_out = get_local_id(1);",    // NOLINT
"int_tp tile_size = get_local_size(0);",    // NOLINT
"Dtype4 s1_cache;",    // NOLINT
"if (local_out == 0) {",    // NOLINT
"s1_cache = vload4(map_size * ch + gId, src1);",    // NOLINT
"vstore4(s1_cache, tile_size * ch + tId, local_src1);",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"int_tp dst_map_offset = map_size * out;",    // NOLINT
"int_tp dst_idx = (dst_map_offset + gId) << 2;",    // NOLINT
"Dtype4 dst_cache = 0.f;",    // NOLINT
"Dtype4 s2_cache;",    // NOLINT
"s1_cache = vload4(tile_size * ch + tId, local_src1);",    // NOLINT
"s2_cache = vload4((dst_map_offset * ch_gr) + (map_size * ch) + gId, src2);",    // NOLINT
"dst_cache.x +=  s1_cache.x * s2_cache.x + s1_cache.y * s2_cache.y;",    // NOLINT
"dst_cache.y += -s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;",    // NOLINT
"dst_cache.z +=  s1_cache.z * s2_cache.z + s1_cache.w * s2_cache.w;",    // NOLINT
"dst_cache.w += -s1_cache.z * s2_cache.w + s1_cache.w * s2_cache.z;",    // NOLINT
"((__global Dtype4*)(&dst[dst_idx]))[0] += dst_cache;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_multiplication_1d,Dtype)(__global Dtype* dst,",    // NOLINT
"const int_tp offset_dst,",    // NOLINT
"const __global Dtype* src1, const int_tp offset_src1,",    // NOLINT
"const __global Dtype* src2, const int_tp offset_src2,",    // NOLINT
"const int_tp size, const int_tp ch_gr) {",    // NOLINT
"dst += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"Dtype4 s2_cache;",    // NOLINT
"Dtype4 dst_cache = 0.f;",    // NOLINT
"int_tp idx_with_ch;",    // NOLINT
"Dtype4 s1_cache = vload4(gId, src1);",    // NOLINT
"for (int_tp ch = 0; ch < ch_gr; ++ch) {",    // NOLINT
"idx_with_ch = size * ch + gId;",    // NOLINT
"s2_cache = vload4(idx_with_ch, src2);",    // NOLINT
"dst_cache.xz = s1_cache.xz * s2_cache.xz - s1_cache.yw * s2_cache.yw;",    // NOLINT
"dst_cache.yw = s1_cache.xz * s2_cache.yw + s1_cache.yw * s2_cache.xz;",    // NOLINT
"((__global Dtype4*)(&dst[idx_with_ch<<2]))[0] += dst_cache;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_multiplication_2d_SLM,Dtype)(__global Dtype* restrict dst,",    // NOLINT
"const int_tp offset_dst, __local Dtype* local_dst,",    // NOLINT
"const __global Dtype* restrict src1, const int_tp offset_src1,",    // NOLINT
"const __global Dtype* restrict src2, const int_tp offset_src2,",    // NOLINT
"const int_tp num_output, const int_tp size, const int_tp ch_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"if (gId >= size) return;",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"if (out >= num_output) return;",    // NOLINT
"dst += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp tId = get_local_id(0);",    // NOLINT
"int_tp tOut = get_local_id(1);",    // NOLINT
"int_tp tile_size = get_local_size(0);",    // NOLINT
"int_tp local_out_size = get_local_size(1);",    // NOLINT
"int_tp out_offset = out * size;",    // NOLINT
"int_tp out_ch_offset = out_offset * ch_gr;",    // NOLINT
"int_tp tile_size_in_all_ch = tile_size * ch_gr;",    // NOLINT
"int_tp local_out_ch_offset = tOut * tile_size_in_all_ch;",    // NOLINT
"int_tp src2_idx, local_dst_idx;",    // NOLINT
"Dtype4 s2_cache, dst_cache;",    // NOLINT
"int_tp src1_idx = out_offset + gId;",    // NOLINT
"Dtype4 s1_cache = vload4(src1_idx, src1);",    // NOLINT
"for (int_tp ch = 0; ch < ch_gr; ++ch) {",    // NOLINT
"src2_idx = out_ch_offset + ch * size + gId;",    // NOLINT
"s2_cache = vload4(src2_idx, src2);",    // NOLINT
"dst_cache.xz = s1_cache.xz * s2_cache.xz - s1_cache.yw * s2_cache.yw;",    // NOLINT
"dst_cache.yw = s1_cache.xz * s2_cache.yw + s1_cache.yw * s2_cache.xz;",    // NOLINT
"local_dst_idx = local_out_ch_offset + ch * tile_size + tId;",    // NOLINT
"vstore4(dst_cache, local_dst_idx, local_dst);",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"int_tp start_idx, half_start_idx;",    // NOLINT
"int_tp ch_offset;",    // NOLINT
"int_tp this_idx, that_idx;",    // NOLINT
"for (int_tp offset = local_out_size >>= 1; offset > 0; offset >>=1) {",    // NOLINT
"if (tOut < offset) {",    // NOLINT
"start_idx = tOut * tile_size_in_all_ch + tId;",    // NOLINT
"half_start_idx = (tOut + offset) * tile_size_in_all_ch + tId;",    // NOLINT
"for (int_tp ch = 0; ch < ch_gr; ++ch) {",    // NOLINT
"ch_offset = ch * tile_size;",    // NOLINT
"this_idx = (start_idx + ch_offset) << 2;",    // NOLINT
"that_idx = (half_start_idx + ch_offset) << 2;",    // NOLINT
"((__local Dtype4*)(&local_dst[this_idx]))[0] +=",    // NOLINT
"((__local Dtype4*)(&local_dst[that_idx]))[0];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"}",    // NOLINT
"if (tOut == 0) {",    // NOLINT
"for (int_tp ch = 0; ch < ch_gr; ++ch) {",    // NOLINT
"dst_cache = vload4(tile_size * ch + tId, local_dst);",    // NOLINT
"((__global Dtype4*)(&dst[(size * ch + gId)<<2]))[0] += dst_cache;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(complex_multiplication_3d,Dtype)(__global Dtype* dst,",    // NOLINT
"const int_tp offset_dst,",    // NOLINT
"const __global Dtype* src1, const int_tp offset_src1,",    // NOLINT
"const __global Dtype* src2, const int_tp offset_src2,",    // NOLINT
"const int_tp size, const int_tp ch_gr, const int_tp out_gr, const int_tp num_output) {",    // NOLINT
"dst  += offset_dst;",    // NOLINT
"src1 += offset_src1;",    // NOLINT
"src2 += offset_src2;",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp ch  = get_global_id(1);",    // NOLINT
"int_tp out = get_global_id(2);",    // NOLINT
"int_tp g = out / out_gr;",    // NOLINT
"ch += (g * ch_gr);",    // NOLINT
"int_tp c_offset = ch - ((ch / ch_gr) * ch_gr);",    // NOLINT
"__global Dtype2* dst_ch = ((__global Dtype2*)(dst)) + (size * ch);",    // NOLINT
"__global Dtype2* src1_out = ((__global Dtype2*)(src1)) + (size * out);",    // NOLINT
"__global Dtype2* src2_out_ch = ((__global Dtype2*)(src2)) + (size * (out * ch_gr + c_offset));",    // NOLINT
"Dtype2 s1_cache  = src1_out[gId];",    // NOLINT
"Dtype2 s2_cache  = src2_out_ch[gId];",    // NOLINT
"Dtype2 dst_cache = 0.f;",    // NOLINT
"dst_cache.x = s1_cache.x * s2_cache.x - s1_cache.y * s2_cache.y;",    // NOLINT
"dst_cache.y = s1_cache.x * s2_cache.y + s1_cache.y * s2_cache.x;",    // NOLINT
"dst_ch[gId] += dst_cache;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"/* Convert [RRRR...GGGG...BBBB...] to [RGBRGBRGBRGB...] */",    // NOLINT
"/* Reshape 2 */",    // NOLINT
"__kernel void TEMPLATE(convert_data_to_channel_major,Dtype)(__global Dtype2* dst,",    // NOLINT
"const __global Dtype2* src, const int_tp size, const int_tp ch_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"__global Dtype* dst_ptr = (__global Dtype*)(dst + (gId * ch_gr));",    // NOLINT
"const __global Dtype* src_ptr = (const __global Dtype*)(src + gId);",    // NOLINT
"Dtype2 s;",    // NOLINT
"int_tp src_idx = 0;",    // NOLINT
"for (int_tp i = 0; i < ch_gr; ++i) {",    // NOLINT
"s = vload2(src_idx, src_ptr);",    // NOLINT
"vstore2(s, i, dst_ptr);",    // NOLINT
"src_idx += size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"/* Reshape 1 */",    // NOLINT
"/*__kernel void TEMPLATE(convert_data_to_channel_major(__global Dtype4* dst,",    // NOLINT
"const __global Dtype4* src, const int_tp size, const int_tp ch_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"const __global Dtype4* src_ptr4 = src + gId;",    // NOLINT
"__global Dtype4* dst_ptr4 = dst + (gId * ch_gr);",    // NOLINT
"for (int_tp i = 0; i < ch_gr; ++i) {",    // NOLINT
"dst_ptr4[i] = src_ptr4[i*size];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"*/",    // NOLINT
"",    // NOLINT
"/* Convert multiple [RRRR...GGGG...BBBB...] to multiple [RGBRGBRGBRGB...] */",    // NOLINT
"/* Reshape 2 */",    // NOLINT
"__kernel void TEMPLATE(convert_weight_to_channel_major,Dtype)(__global Dtype2* dst,",    // NOLINT
"const __global Dtype2* src, const int_tp size, const int_tp ch_gr,",    // NOLINT
"const int_tp num_output) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp out_offset = out * (size * ch_gr);",    // NOLINT
"__global Dtype* dst_ptr = (__global Dtype*)(dst + out_offset + (gId * ch_gr));",    // NOLINT
"const __global Dtype* src_ptr =",    // NOLINT
"(const __global Dtype*)(src + out_offset + gId);",    // NOLINT
"Dtype2 s;",    // NOLINT
"int_tp src_idx = 0;",    // NOLINT
"for (int_tp i = 0; i < ch_gr; ++i) {",    // NOLINT
"s = vload2(src_idx, src_ptr);",    // NOLINT
"vstore2(s, i, dst_ptr);",    // NOLINT
"src_idx += size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"/* Reshape 1 */",    // NOLINT
"/*",    // NOLINT
"__kernel void TEMPLATE(convert_weight_to_channel_major(__global Dtype4* dst,",    // NOLINT
"const __global Dtype4* src, const int_tp size, const int_tp ch_gr,",    // NOLINT
"const int_tp out_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp out_offset = out * (size * ch_gr);",    // NOLINT
"__global Dtype4* dst_ptr4 = dst + out_offset + (gId * ch_gr);",    // NOLINT
"const __global Dtype4* src_ptr4 = src + out_offset + gId;",    // NOLINT
"for (int_tp i = 0; i < ch_gr; ++i) {",    // NOLINT
"dst_ptr4[i] = src_ptr4[size * i];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"*/",    // NOLINT
"",    // NOLINT
"/* Cdotc per element */",    // NOLINT
"/* Reshape 1 */",    // NOLINT
"/*",    // NOLINT
"__kernel void TEMPLATE(batchedCdotc(__global Dtype4* dst,",    // NOLINT
"const __global Dtype4* src1, const __global Dtype4* src2,",    // NOLINT
"const int_tp size, const int_tp ch_gr, const int_tp out_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp ch_offset = gId * ch_gr;",    // NOLINT
"int_tp out_offset = out * size;",    // NOLINT
"const __global Dtype* src1_ptr = (const __global Dtype*)(src1 + ch_offset);",    // NOLINT
"const __global Dtype* src2_ptr = (const __global Dtype*)(src2 + (out_offset * ch_gr) + ch_offset);",    // NOLINT
"Dtype4 cdotc = 0.f;",    // NOLINT
"Dtype4 s1, s2;",    // NOLINT
"for (int_tp c = 0; c < ch_gr; ++c) {",    // NOLINT
"s1 = vload4(c, src1_ptr);",    // NOLINT
"s2 = vload4(c, src2_ptr);",    // NOLINT
"cdotc.xz += mad( s1.xz, s2.xz, s1.yw * s2.yw);",    // NOLINT
"cdotc.yw += mad(-s1.xz, s2.yw, s1.yw * s2.xz);",    // NOLINT
"}",    // NOLINT
"__global Dtype4* dst_ptr4 = dst + out_offset + gId;",    // NOLINT
"dst_ptr4[0] += cdotc;",    // NOLINT
"}",    // NOLINT
"*/",    // NOLINT
"",    // NOLINT
"/* Cdotc per two elements */",    // NOLINT
"/* Reshape 2 */",    // NOLINT
"__kernel void TEMPLATE(batchedCdotc,Dtype)(__global Dtype2* dst,",    // NOLINT
"const __global Dtype2* src1, const __global Dtype2* src2,",    // NOLINT
"const int_tp size, const int_tp ch_gr, const int_tp out_gr) {",    // NOLINT
"int_tp gId = get_global_id(0);",    // NOLINT
"int_tp out = get_global_id(1);",    // NOLINT
"int_tp ch_offset = gId * ch_gr;",    // NOLINT
"const __global Dtype* src1_ptr = (const __global Dtype*)(src1 + ch_offset);",    // NOLINT
"const __global Dtype* src2_ptr =",    // NOLINT
"(const __global Dtype*)(src2 + (out * size * ch_gr) + ch_offset);",    // NOLINT
"Dtype4 cdotc4 = 0.f;",    // NOLINT
"Dtype2 cdotc = 0.f;",    // NOLINT
"Dtype4 s1, s2;",    // NOLINT
"int_tp n = ch_gr >> 1;",    // NOLINT
"int_tp r = ch_gr - (n << 1);",    // NOLINT
"for (int_tp i = 0; i < n; ++i) {",    // NOLINT
"s1 = vload4(i, src1_ptr);",    // NOLINT
"s2 = vload4(i, src2_ptr);",    // NOLINT
"cdotc4.xz += mad( s1.xz, s2.xz, s1.yw * s2.yw);",    // NOLINT
"cdotc4.yw += mad(-s1.xz, s2.yw, s1.yw * s2.xz);",    // NOLINT
"}",    // NOLINT
"cdotc.x += dot(cdotc4.xz, (float2)(1));",    // NOLINT
"cdotc.y += dot(cdotc4.yw, (float2)(1));",    // NOLINT
"if (r == 1) {",    // NOLINT
"const __global Dtype* src1_ptr2 =",    // NOLINT
"(const __global Dtype*)(((const __global Dtype4*)(src1_ptr)) + n);",    // NOLINT
"const __global Dtype* src2_ptr2 =",    // NOLINT
"(const __global Dtype*)(((const __global Dtype4*)(src2_ptr)) + n);",    // NOLINT
"Dtype2 t1 = vload2(0, src1_ptr2);",    // NOLINT
"Dtype2 t2 = vload2(0, src2_ptr2);",    // NOLINT
"cdotc.x += mad( t1.x, t2.x, t1.y * t2.y);",    // NOLINT
"cdotc.y += mad(-t1.x, t2.y, t1.y * t2.x);",    // NOLINT
"}",    // NOLINT
"__global Dtype* dst_ptr = (__global Dtype*)(dst + (out * size) + gId);",    // NOLINT
"vstore2(cdotc, 0, dst_ptr);",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(fillbuffer,Dtype)(const int_tp n, const char alpha, __global char* x,",    // NOLINT
"const int_tp offx) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"x[index + offx] = alpha;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(fill,Dtype)(const int_tp n, const Dtype alpha, __global Dtype* x,",    // NOLINT
"const int_tp offx) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"x[index + offx] = alpha;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(im2col,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* data_im,",    // NOLINT
"const int_tp data_im_off,",    // NOLINT
"const int_tp height, const int_tp width,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w, const int_tp stride_h,",    // NOLINT
"const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h,",    // NOLINT
"const int_tp dilation_w,",    // NOLINT
"const int_tp height_col,",    // NOLINT
"const int_tp width_col,",    // NOLINT
"__global Dtype* data_col,",    // NOLINT
"const int_tp data_col_off) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp h_index = index / width_col;",    // NOLINT
"const int_tp h_col = h_index % height_col;",    // NOLINT
"const int_tp w_col = index % width_col;",    // NOLINT
"const int_tp c_im = h_index / height_col;",    // NOLINT
"const int_tp c_col = c_im * kernel_h * kernel_w;",    // NOLINT
"const int_tp h_offset = h_col * stride_h - pad_h;",    // NOLINT
"const int_tp w_offset = w_col * stride_w - pad_w;",    // NOLINT
"__global Dtype* data_col_ptr = data_col + data_col_off;",    // NOLINT
"data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;",    // NOLINT
"__global const Dtype* data_im_ptr = data_im + data_im_off;",    // NOLINT
"data_im_ptr += (c_im * height + h_offset) * width + w_offset;",    // NOLINT
"for (int_tp i = 0; i < kernel_h; ++i) {",    // NOLINT
"for (int_tp j = 0; j < kernel_w; ++j) {",    // NOLINT
"int_tp h_im = h_offset + i * dilation_h;",    // NOLINT
"int_tp w_im = w_offset + j * dilation_w;",    // NOLINT
"*data_col_ptr =",    // NOLINT
"(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?",    // NOLINT
"data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;",    // NOLINT
"data_col_ptr += height_col * width_col;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(col2im,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* data_col,",    // NOLINT
"const int_tp data_col_off,",    // NOLINT
"const int_tp height, const int_tp width,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w, const int_tp stride_h,",    // NOLINT
"const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h,",    // NOLINT
"const int_tp dilation_w,",    // NOLINT
"const int_tp height_col,",    // NOLINT
"const int_tp width_col,",    // NOLINT
"__global Dtype* data_im,",    // NOLINT
"const int_tp data_im_off) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"Dtype val = 0;",    // NOLINT
"const int_tp w_im = index % width + pad_w;",    // NOLINT
"const int_tp h_im = (index / width) % height + pad_h;",    // NOLINT
"const int_tp c_im = index / (width * height);",    // NOLINT
"int_tp kernel_extent_w = (kernel_w - 1) * dilation_w + 1;",    // NOLINT
"int_tp kernel_extent_h = (kernel_h - 1) * dilation_h + 1;",    // NOLINT
"// compute the start and end of the output",    // NOLINT
"const int_tp w_col_start =",    // NOLINT
"(w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;",    // NOLINT
"const int_tp w_col_end = min(w_im / stride_w + 1, width_col);",    // NOLINT
"const int_tp h_col_start =",    // NOLINT
"(h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;",    // NOLINT
"const int_tp h_col_end = min(h_im / stride_h + 1, height_col);",    // NOLINT
"// TODO: use LCM of stride and dilation to avoid unnecessary loops",    // NOLINT
"for (int_tp h_col = h_col_start; h_col < h_col_end; h_col += 1) {",    // NOLINT
"for (int_tp w_col = w_col_start; w_col < w_col_end; w_col += 1) {",    // NOLINT
"int_tp h_k = (h_im - h_col * stride_h);",    // NOLINT
"int_tp w_k = (w_im - w_col * stride_w);",    // NOLINT
"if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {",    // NOLINT
"h_k /= dilation_h;",    // NOLINT
"w_k /= dilation_w;",    // NOLINT
"int_tp data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *",    // NOLINT
"height_col + h_col) * width_col + w_col;",    // NOLINT
"val += data_col[data_col_off + data_col_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"data_im[data_im_off + index] = val;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(im2col_nd, Dtype)(const int_tp n, const int_tp num_axes,",    // NOLINT
"const int_tp channel_axis,",    // NOLINT
"__global const Dtype* data_im,",    // NOLINT
"const int_tp data_im_off,",    // NOLINT
"__global const int_tp* im_shape,",    // NOLINT
"__global const int_tp* col_shape,",    // NOLINT
"__global const int_tp* kernel_shape,",    // NOLINT
"__global const int_tp* pad,",    // NOLINT
"__global const int_tp* stride,",    // NOLINT
"__global const int_tp* dilation,",    // NOLINT
"__global Dtype* data_col,",    // NOLINT
"const int_tp data_col_off) {",    // NOLINT
"int_tp d_temp[6];",    // NOLINT
"int_tp d_iter[6];",    // NOLINT
"int_tp i;",    // NOLINT
"",    // NOLINT
"__global const int_tp* im_shape_ptr = im_shape + channel_axis;",    // NOLINT
"__global const int_tp* col_shape_ptr = col_shape + channel_axis;",    // NOLINT
"",    // NOLINT
"__local int_tp shared_dilation[6];",    // NOLINT
"__local int_tp shared_kernel_shape[6];",    // NOLINT
"__local int_tp shared_pad[6];",    // NOLINT
"__local int_tp shared_stride[6];",    // NOLINT
"__local int_tp shared_col_shape[6 + 1];",    // NOLINT
"__local int_tp shared_im_shape[6 + 1];",    // NOLINT
"",    // NOLINT
"for (int li = get_local_id(0); li < num_axes; li += get_local_size(0)) {",    // NOLINT
"shared_dilation[li] = dilation[li];",    // NOLINT
"shared_kernel_shape[li] = kernel_shape[li];",    // NOLINT
"shared_pad[li] = pad[li];",    // NOLINT
"shared_stride[li] = stride[li];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for (int li = get_local_id(0); li < num_axes + 1; li += get_local_size(0)) {",    // NOLINT
"shared_col_shape[li] = col_shape_ptr[li];",    // NOLINT
"shared_im_shape[li] = im_shape_ptr[li];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// Initialize channel_in, computed in the loop below, with intermediate",    // NOLINT
"// computations used to compute the spatial indices.",    // NOLINT
"int_tp channel_in = index;",    // NOLINT
"int_tp channel_out = 1;",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"d_temp[i] = channel_in % shared_col_shape[i + 1];",    // NOLINT
"channel_in /= shared_col_shape[i + 1];",    // NOLINT
"channel_out *= shared_kernel_shape[i];",    // NOLINT
"}",    // NOLINT
"channel_out *= channel_in;",    // NOLINT
"int_tp data_col_inc = 1;",    // NOLINT
"for (i = 0; i < num_axes; ++i) {",    // NOLINT
"channel_out *= shared_col_shape[i + 1];",    // NOLINT
"channel_out += d_temp[i];",    // NOLINT
"d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];",    // NOLINT
"channel_in *= shared_im_shape[i + 1];",    // NOLINT
"channel_in += d_temp[i];",    // NOLINT
"data_col_inc *= shared_col_shape[i + 1];",    // NOLINT
"d_iter[i] = 0;",    // NOLINT
"}",    // NOLINT
"__global Dtype* data_col_ptr = data_col + data_col_off + channel_out;",    // NOLINT
"__global const Dtype* data_im_ptr = data_im + data_im_off + channel_in;",    // NOLINT
"bool incremented;",    // NOLINT
"do {",    // NOLINT
"bool in_range = true;",    // NOLINT
"for (i = 0; i < num_axes; ++i) {",    // NOLINT
"const int_tp d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];",    // NOLINT
"in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];",    // NOLINT
"if (!in_range) {",    // NOLINT
"break;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if (in_range) {",    // NOLINT
"int_tp data_im_offset = d_iter[0] * shared_dilation[0];",    // NOLINT
"for (i = 1; i < num_axes; ++i) {",    // NOLINT
"data_im_offset *= shared_im_shape[i + 1];",    // NOLINT
"data_im_offset += d_iter[i] * shared_dilation[i];",    // NOLINT
"}",    // NOLINT
"*data_col_ptr = data_im_ptr[data_im_offset];",    // NOLINT
"} else {",    // NOLINT
"*data_col_ptr = 0;",    // NOLINT
"}",    // NOLINT
"data_col_ptr += data_col_inc;",    // NOLINT
"incremented = false;",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"const int_tp d_max = shared_kernel_shape[i];",    // NOLINT
"if (d_iter[i] == d_max - 1) {",    // NOLINT
"d_iter[i] = 0;",    // NOLINT
"} else {  // d_iter[i] < d_max - 1",    // NOLINT
"++d_iter[i];",    // NOLINT
"incremented = true;",    // NOLINT
"break;",    // NOLINT
"}",    // NOLINT
"}  // for (int_tp i = num_axes - 1; i >= 0; --i)",    // NOLINT
"} while (incremented);  // do",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(col2im_nd, Dtype)(const int_tp n, const int_tp num_axes,",    // NOLINT
"const int_tp channel_axis,",    // NOLINT
"__global const Dtype* data_col,",    // NOLINT
"const int_tp data_col_off,",    // NOLINT
"__global const int_tp* im_shape,",    // NOLINT
"__global const int_tp* col_shape,",    // NOLINT
"__global const int_tp* kernel_shape,",    // NOLINT
"__global const int_tp* pad,",    // NOLINT
"__global const int_tp* stride,",    // NOLINT
"__global const int_tp* dilation,",    // NOLINT
"__global Dtype* data_im,",    // NOLINT
"const int_tp data_im_off) {",    // NOLINT
"int_tp d_im[6];",    // NOLINT
"int_tp d_col_iter[6];",    // NOLINT
"int_tp d_col_start[6];",    // NOLINT
"int_tp d_col_end[6];",    // NOLINT
"",    // NOLINT
"__global const int_tp* im_shape_ptr = im_shape + channel_axis;",    // NOLINT
"__global const int_tp* col_shape_ptr = col_shape + channel_axis;",    // NOLINT
"",    // NOLINT
"__local int_tp shared_dilation[6];",    // NOLINT
"__local int_tp shared_kernel_shape[6];",    // NOLINT
"__local int_tp shared_pad[6];",    // NOLINT
"__local int_tp shared_stride[6];",    // NOLINT
"__local int_tp shared_col_shape[6 + 1];",    // NOLINT
"__local int_tp shared_im_shape[6 + 1];",    // NOLINT
"",    // NOLINT
"for (int li = get_local_id(0); li < num_axes; li += get_local_size(0)) {",    // NOLINT
"shared_dilation[li] = dilation[li];",    // NOLINT
"shared_kernel_shape[li] = kernel_shape[li];",    // NOLINT
"shared_pad[li] = pad[li];",    // NOLINT
"shared_stride[li] = stride[li];",    // NOLINT
"}",    // NOLINT
"for (int li = get_local_id(0); li < num_axes + 1; li += get_local_size(0)) {",    // NOLINT
"shared_col_shape[li] = col_shape_ptr[li];",    // NOLINT
"shared_im_shape[li] = im_shape_ptr[li];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"// Initialize channel_in, computed in the loop below, with intermediate",    // NOLINT
"// computations used to compute the spatial indices.",    // NOLINT
"int_tp c_im = index;",    // NOLINT
"// Calculate d_im (image dimensions).",    // NOLINT
"for (int_tp i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];",    // NOLINT
"c_im /= shared_im_shape[i + 1];",    // NOLINT
"}",    // NOLINT
"// Calculate col start/end indices.",    // NOLINT
"bool done = false;",    // NOLINT
"for (int_tp i = 0; i < num_axes; ++i) {",    // NOLINT
"const int_tp kernel_extent = shared_dilation[i]",    // NOLINT
"* (shared_kernel_shape[i] - 1) + 1;",    // NOLINT
"d_col_start[i] = d_col_iter[i] =",    // NOLINT
"(d_im[i] < kernel_extent) ?",    // NOLINT
"0 : (d_im[i] - kernel_extent) / shared_stride[i] + 1;",    // NOLINT
"d_col_end[i] = min(d_im[i] / shared_stride[i] + 1,",    // NOLINT
"shared_col_shape[i + 1]);",    // NOLINT
"if (d_col_start[i] >= d_col_end[i]) {",    // NOLINT
"// Skip computation if the dimension is 0 at any spatial axis --",    // NOLINT
"// final val will be 0.",    // NOLINT
"data_im[index] = (Dtype)0.0;",    // NOLINT
"done = true;",    // NOLINT
"break;  // for (int_tp i = 0; i < num_axes; ++i)",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if (!done) {",    // NOLINT
"// Loop over the col to compute the output val.",    // NOLINT
"Dtype val = (Dtype)0.0;",    // NOLINT
"bool incremented = true;",    // NOLINT
"bool skip = false;",    // NOLINT
"do {",    // NOLINT
"// Compute the final offset.",    // NOLINT
"int_tp final_offset = 0;",    // NOLINT
"int_tp kernel_shape_prod = 1;",    // NOLINT
"int_tp kernel_index;",    // NOLINT
"for (int_tp i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];",    // NOLINT
"if (kernel_index % shared_dilation[i]) {",    // NOLINT
"skip = true;",    // NOLINT
"break;",    // NOLINT
"} else {",    // NOLINT
"kernel_index /= shared_dilation[i];",    // NOLINT
"final_offset += kernel_index * kernel_shape_prod;",    // NOLINT
"kernel_shape_prod *= shared_kernel_shape[i];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if (!skip) {",    // NOLINT
"final_offset += kernel_shape_prod * c_im;",    // NOLINT
"for (int_tp i = 0; i < num_axes; ++i) {",    // NOLINT
"final_offset *= shared_col_shape[i + 1];",    // NOLINT
"final_offset += d_col_iter[i];",    // NOLINT
"}",    // NOLINT
"val += data_col[data_col_off + final_offset];",    // NOLINT
"}",    // NOLINT
"skip = false;",    // NOLINT
"incremented = false;",    // NOLINT
"for (int_tp i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"const int_tp d_max = d_col_end[i];",    // NOLINT
"if (d_col_iter[i] == d_max - 1) {",    // NOLINT
"d_col_iter[i] = d_col_start[i];",    // NOLINT
"} else {  // d_col_iter[i] < d_max - 1",    // NOLINT
"++d_col_iter[i];",    // NOLINT
"incremented = true;",    // NOLINT
"break;  // for (int_tp i = num_axes - 1; i >= 0; --i)",    // NOLINT
"}",    // NOLINT
"}  // for (int_tp i = num_axes - 1; i >= 0; --i)",    // NOLINT
"} while (incremented);",    // NOLINT
"data_im[data_im_off + index] = val;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_compute_output,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global const Dtype* scale,",    // NOLINT
"const Dtype negative_beta,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] * pow(scale[index], negative_beta);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,",    // NOLINT
"const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp size,",    // NOLINT
"const Dtype alpha_over_size, const Dtype k,",    // NOLINT
"__global Dtype* const scale) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp n = index / width / height;",    // NOLINT
"const int_tp offset = (n * channels * height + h) * width + w;",    // NOLINT
"const int_tp step = height * width;",    // NOLINT
"__global const Dtype* in_off = in + offset;",    // NOLINT
"__global Dtype* scale_off = scale + offset;",    // NOLINT
"int_tp head = 0;",    // NOLINT
"const int_tp pre_pad = (size - 1) / 2;",    // NOLINT
"const int_tp post_pad = size - pre_pad - 1;",    // NOLINT
"Dtype accum_scale = 0;",    // NOLINT
"// fill the scale at [n, :, h, w]",    // NOLINT
"// accumulate values",    // NOLINT
"while (head < post_pad && head < channels) {",    // NOLINT
"accum_scale += in_off[head * step] * in_off[head * step];",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// both add and subtract",    // NOLINT
"while (head < channels) {",    // NOLINT
"accum_scale += in_off[head * step] * in_off[head * step];",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// subtract only",    // NOLINT
"while (head < channels + post_pad) {",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_compute_diff,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* bottom_data,",    // NOLINT
"__global const Dtype* top_data,",    // NOLINT
"__global const Dtype* scale,",    // NOLINT
"__global const Dtype* top_diff, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height,",    // NOLINT
"const int_tp width, const int_tp size,",    // NOLINT
"const Dtype negative_beta,",    // NOLINT
"const Dtype cache_ratio,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp n = index / width / height;",    // NOLINT
"const int_tp offset = (n * channels * height + h) * width + w;",    // NOLINT
"const int_tp step = height * width;",    // NOLINT
"__global const Dtype* bottom_off = bottom_data + offset;",    // NOLINT
"__global const Dtype* top_off = top_data + offset;",    // NOLINT
"__global const Dtype* scale_off = scale + offset;",    // NOLINT
"__global const Dtype* top_diff_off = top_diff + offset;",    // NOLINT
"__global Dtype* bottom_diff_off = bottom_diff + offset;",    // NOLINT
"int_tp head = 0;",    // NOLINT
"const int_tp pre_pad = size - (size + 1) / 2;",    // NOLINT
"const int_tp post_pad = size - pre_pad - 1;",    // NOLINT
"Dtype accum_ratio = 0;",    // NOLINT
"// accumulate values",    // NOLINT
"while (head < post_pad && head < channels) {",    // NOLINT
"accum_ratio += top_diff_off[head * step] * top_off[head * step]",    // NOLINT
"/ scale_off[head * step];",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// both add and subtract",    // NOLINT
"while (head < channels) {",    // NOLINT
"accum_ratio += top_diff_off[head * step] * top_off[head * step]",    // NOLINT
"/ scale_off[head * step];",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_ratio -= top_diff_off[(head - size) * step]",    // NOLINT
"* top_off[(head - size) * step] / scale_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)",    // NOLINT
"* step] * pow(scale_off[(head - post_pad) * step], negative_beta)",    // NOLINT
"- cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// subtract only",    // NOLINT
"while (head < channels + post_pad) {",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_ratio -= top_diff_off[(head - size) * step]",    // NOLINT
"* top_off[(head - size) * step] / scale_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)",    // NOLINT
"* step] * pow(scale_off[(head - post_pad) * step], negative_beta)",    // NOLINT
"- cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_full_no_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,",    // NOLINT
"const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp size,",    // NOLINT
"const Dtype alpha_over_size, const Dtype k,",    // NOLINT
"__global Dtype* const out,",    // NOLINT
"const Dtype negative_beta) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp n = index / width / height;",    // NOLINT
"const int_tp offset = (n * channels * height + h) * width + w;",    // NOLINT
"const int_tp step = height * width;",    // NOLINT
"__global const Dtype* in_off = in + offset;",    // NOLINT
"__global Dtype* out_off = out + offset;",    // NOLINT
"Dtype scale_val;",    // NOLINT
"int_tp head = 0;",    // NOLINT
"const int_tp pre_pad = (size - 1) / 2;",    // NOLINT
"const int_tp post_pad = size - pre_pad - 1;",    // NOLINT
"Dtype accum_scale = 0;",    // NOLINT
"// fill the scale at [n, :, h, w]",    // NOLINT
"// accumulate values",    // NOLINT
"while (head < post_pad && head < channels) {",    // NOLINT
"accum_scale += in_off[head * step] * in_off[head * step];",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// both add and subtract",    // NOLINT
"while (head < channels) {",    // NOLINT
"accum_scale += in_off[head * step] * in_off[head * step];",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_val = k + accum_scale * alpha_over_size;",    // NOLINT
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// subtract only",    // NOLINT
"while (head < channels + post_pad) {",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_val = k + accum_scale * alpha_over_size;",    // NOLINT
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_full,Dtype)(const int_tp nthreads, __global const Dtype* in,",    // NOLINT
"const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp size,",    // NOLINT
"const Dtype alpha_over_size, const Dtype k,",    // NOLINT
"__global Dtype* const scale,",    // NOLINT
"__global Dtype* const out,",    // NOLINT
"const Dtype negative_beta) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp n = index / width / height;",    // NOLINT
"const int_tp offset = (n * channels * height + h) * width + w;",    // NOLINT
"const int_tp step = height * width;",    // NOLINT
"__global const Dtype* in_off = in + offset;",    // NOLINT
"__global Dtype* out_off = out + offset;",    // NOLINT
"__global Dtype* scale_off = scale + offset;",    // NOLINT
"Dtype scale_val;",    // NOLINT
"int_tp head = 0;",    // NOLINT
"const int_tp pre_pad = (size - 1) / 2;",    // NOLINT
"const int_tp post_pad = size - pre_pad - 1;",    // NOLINT
"Dtype accum_scale = 0;",    // NOLINT
"// fill the scale at [n, :, h, w]",    // NOLINT
"// accumulate values",    // NOLINT
"while (head < post_pad && head < channels) {",    // NOLINT
"accum_scale += in_off[head * step] * in_off[head * step];",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// both add and subtract",    // NOLINT
"while (head < channels) {",    // NOLINT
"accum_scale += in_off[head * step] * in_off[head * step];",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_val = k + accum_scale * alpha_over_size;",    // NOLINT
"scale_off[(head - post_pad) * step] = scale_val;",    // NOLINT
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// subtract only",    // NOLINT
"while (head < channels + post_pad) {",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_val = k + accum_scale * alpha_over_size;",    // NOLINT
"scale_off[(head - post_pad) * step] = scale_val;",    // NOLINT
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((float)scale_val, (float)negative_beta);",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"inline Dtype TEMPLATE(lstm_sigmoid,Dtype)(const Dtype x) {",    // NOLINT
"return (Dtype)1 / ((Dtype)1 + exp(-x));",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"inline Dtype TEMPLATE(lstm_tanh,Dtype)(const Dtype x) {",    // NOLINT
"return (Dtype)2 * TEMPLATE(lstm_sigmoid,Dtype)((Dtype)2 * x) - (Dtype)1;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lstm_acts_forward,Dtype)(const int_tp nthreads, const int_tp dim,",    // NOLINT
"__global const Dtype* X, __global Dtype* X_acts) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp x_dim = 4 * dim;",    // NOLINT
"const int_tp d = index % x_dim;",    // NOLINT
"if (d < 3 * dim) {",    // NOLINT
"X_acts[index] = TEMPLATE(lstm_sigmoid,Dtype)(X[index]);",    // NOLINT
"} else {",    // NOLINT
"X_acts[index] = TEMPLATE(lstm_tanh,Dtype)(X[index]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lstm_unit_forward,Dtype)(const int_tp nthreads, const int_tp dim,",    // NOLINT
"__global const Dtype* C_prev, __global const Dtype* X, __global const Dtype* cont,",    // NOLINT
"__global Dtype* C, __global Dtype* H) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp n = index / dim;",    // NOLINT
"const int_tp d = index % dim;",    // NOLINT
"__global const Dtype* X_offset = X + 4 * dim * n;",    // NOLINT
"const Dtype i = X_offset[d];",    // NOLINT
"const Dtype f = X_offset[1 * dim + d];",    // NOLINT
"const Dtype o = X_offset[2 * dim + d];",    // NOLINT
"const Dtype g = X_offset[3 * dim + d];",    // NOLINT
"const Dtype c_prev = C_prev[index];",    // NOLINT
"const Dtype c = cont[n] * f * c_prev + i * g;",    // NOLINT
"C[index] = c;",    // NOLINT
"const Dtype tanh_c = TEMPLATE(lstm_tanh,Dtype)(c);",    // NOLINT
"H[index] = o * tanh_c;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lstm_unit_backward,Dtype)(const int_tp nthreads, const int_tp dim,",    // NOLINT
"__global const Dtype* C_prev, __global const Dtype* X, __global const Dtype* C, __global const Dtype* H,",    // NOLINT
"__global const Dtype* cont, __global const Dtype* C_diff, __global const Dtype* H_diff,",    // NOLINT
"__global Dtype* C_prev_diff, __global Dtype* X_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp n = index / dim;",    // NOLINT
"const int_tp d = index % dim;",    // NOLINT
"__global const Dtype* X_offset = X + 4 * dim * n;",    // NOLINT
"const Dtype i = X_offset[d];",    // NOLINT
"const Dtype f = X_offset[1 * dim + d];",    // NOLINT
"const Dtype o = X_offset[2 * dim + d];",    // NOLINT
"const Dtype g = X_offset[3 * dim + d];",    // NOLINT
"const Dtype c_prev = C_prev[index];",    // NOLINT
"const Dtype c = C[index];",    // NOLINT
"const Dtype tanh_c = TEMPLATE(lstm_tanh,Dtype)(c);",    // NOLINT
"__global Dtype* c_prev_diff = C_prev_diff + index;",    // NOLINT
"__global Dtype* X_diff_offset = X_diff + 4 * dim * n;",    // NOLINT
"__global Dtype* i_diff = X_diff_offset + d;",    // NOLINT
"__global Dtype* f_diff = X_diff_offset + 1 * dim + d;",    // NOLINT
"__global Dtype* o_diff = X_diff_offset + 2 * dim + d;",    // NOLINT
"__global Dtype* g_diff = X_diff_offset + 3 * dim + d;",    // NOLINT
"const Dtype c_term_diff =",    // NOLINT
"C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);",    // NOLINT
"const Dtype cont_n = cont[n];",    // NOLINT
"*c_prev_diff = cont_n * c_term_diff * f;",    // NOLINT
"*i_diff = c_term_diff * g;",    // NOLINT
"*f_diff = cont_n * c_term_diff * c_prev;",    // NOLINT
"*o_diff = H_diff[index] * tanh_c;",    // NOLINT
"*g_diff = c_term_diff * i;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lstm_acts_backward,Dtype)(const int_tp nthreads, const int_tp dim,",    // NOLINT
"__global const Dtype* X_acts, __global const Dtype* X_acts_diff, __global Dtype* X_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp x_dim = 4 * dim;",    // NOLINT
"const int_tp d = index % x_dim;",    // NOLINT
"const Dtype X_act = X_acts[index];",    // NOLINT
"if (d < 3 * dim) {",    // NOLINT
"X_diff[index] = X_acts_diff[index] * X_act * ((Dtype)1 - X_act);",    // NOLINT
"} else {",    // NOLINT
"X_diff[index] = X_acts_diff[index] * ((Dtype)1 - X_act * X_act);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(mul,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa,",    // NOLINT
"__global Dtype* b,",    // NOLINT
"const int_tp offb, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[index + offy] = a[index + offa] * b[index + offb];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(div,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa,",    // NOLINT
"__global Dtype* b,",    // NOLINT
"const int_tp offb, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[index + offy] = a[index + offa] / b[index + offb];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(add_scalar,Dtype)(const int_tp N, const Dtype alpha,",    // NOLINT
"__global Dtype* Y,",    // NOLINT
"const int_tp offY) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < N; index += get_global_size(0)) {",    // NOLINT
"Y[offY + index] += alpha;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(add,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, __global const Dtype* b,",    // NOLINT
"const int_tp offb, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[offy + index] = a[offa + index] + b[offb + index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sub,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, __global const Dtype* b,",    // NOLINT
"const int_tp offb, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[offy + index] = a[offa + index] - b[offb + index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(abs,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[offy + index] = fabs((Dtype)(a[offa + index]));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(exp,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[offy + index] = exp(a[offa + index]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(log,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[offy + index] = log((Dtype)(a[offa + index]));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sqrt,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa,",    // NOLINT
"__global Dtype* y, const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[offy + index] = sqrt((Dtype)a[offa + index]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(powx,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, Dtype alpha,",    // NOLINT
"__global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"if(alpha == 2.0) {",    // NOLINT
"y[offy + index] = pow((Dtype)fabs(a[offa + index]), (Dtype)alpha);",    // NOLINT
"} else {",    // NOLINT
"y[offy + index] = pow((Dtype)a[offa + index], (Dtype)alpha);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sign,Dtype)(const int_tp n, __global const Dtype* x,",    // NOLINT
"const int_tp offx, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[index + offy] = (0.0 < x[index + offx])",    // NOLINT
"- (x[index + offx] < 0.0);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sgnbit,Dtype)(const int_tp n, __global const Dtype* x,",    // NOLINT
"const int_tp offx, __global Dtype* y,",    // NOLINT
"const int_tp offy) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"y[index + offy] = signbit(x[index + offx]);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(merge_copy_forward_stack, Dtype)(const int_tp nthreads,",    // NOLINT
"const int_tp dims,",    // NOLINT
"__global const Dtype* bottom_a,",    // NOLINT
"const int_tp forward_a,",    // NOLINT
"__global const Dtype* bottom_b,",    // NOLINT
"const int_tp forward_b,",    // NOLINT
"__global Dtype* top,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels_a,",    // NOLINT
"const int_tp channels_b,",    // NOLINT
"__global const int_tp* shape_a,",    // NOLINT
"__global const int_tp* shape_b) {",    // NOLINT
"int_tp pad[6];",    // NOLINT
"int_tp tmp_idx[6];",    // NOLINT
"int_tp size_a = 1;",    // NOLINT
"int_tp size_b = 1;",    // NOLINT
"",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"pad[i] = (shape_b[i] - shape_a[i]) / 2;",    // NOLINT
"size_a *= shape_a[i];",    // NOLINT
"size_b *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp batch_id = index / ((channels_a + channels_b) * size_a);",    // NOLINT
"int_tp bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)",    // NOLINT
"/ (channels_a * size_a)) % 2;",    // NOLINT
"int_tp counter = index;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"tmp_idx[i] = counter % shape_a[i];",    // NOLINT
"counter /= shape_a[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if (bottom_id == 0) {",    // NOLINT
"int_tp channel_id = (index / size_a) % channels_a;",    // NOLINT
"int_tp aidx = batch_id * channels_a + channel_id;",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"aidx *= shape_a[i];",    // NOLINT
"aidx += tmp_idx[i];",    // NOLINT
"}",    // NOLINT
"top[index] = (forward_a == 1) ? bottom_a[aidx] : 0;",    // NOLINT
"} else {",    // NOLINT
"int_tp channel_id = (index / size_a) % channels_b;",    // NOLINT
"int_tp bidx = (batch_id * channels_b + channel_id) * size_b;",    // NOLINT
"int_tp btemp = 1;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"bidx += btemp * (tmp_idx[i] + pad[i]);",    // NOLINT
"btemp *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"top[index] = (forward_b == 1) ? bottom_b[bidx] : 0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(merge_copy_backward_stack,Dtype)(const int_tp nthreads,",    // NOLINT
"const int_tp dims,",    // NOLINT
"__global Dtype* bottom_a,",    // NOLINT
"const int_tp backward_a,",    // NOLINT
"__global Dtype* bottom_b,",    // NOLINT
"const int_tp backward_b,",    // NOLINT
"__global const Dtype* top,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels_a,",    // NOLINT
"const int_tp channels_b,",    // NOLINT
"__global const int_tp* shape_a,",    // NOLINT
"__global const int_tp* shape_b) {",    // NOLINT
"int_tp pad[6];",    // NOLINT
"int_tp tmp_idx[6];",    // NOLINT
"int_tp size_a = 1;",    // NOLINT
"int_tp size_b = 1;",    // NOLINT
"",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"pad[i] = (shape_b[i] - shape_a[i]) / 2;",    // NOLINT
"size_a *= shape_a[i];",    // NOLINT
"size_b *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp batch_id = index / ((channels_a + channels_b) * size_a);",    // NOLINT
"int_tp bottom_id = ((index - batch_id * (channels_a + channels_b) * size_a)",    // NOLINT
"/ (channels_a * size_a)) % 2;",    // NOLINT
"int_tp counter = index;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"tmp_idx[i] = counter % shape_a[i];",    // NOLINT
"counter /= shape_a[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if (bottom_id == 0) {",    // NOLINT
"int_tp channel_id = (index / size_a) % channels_a;",    // NOLINT
"int_tp aidx = batch_id * channels_a + channel_id;",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"aidx *= shape_a[i];",    // NOLINT
"aidx += tmp_idx[i];",    // NOLINT
"}",    // NOLINT
"bottom_a[aidx] = (backward_a == 1) ? top[index] : 0;",    // NOLINT
"} else {",    // NOLINT
"int_tp channel_id = (index / size_a) % channels_b;",    // NOLINT
"int_tp bidx = (batch_id * channels_b + channel_id) * size_b;",    // NOLINT
"int_tp btemp = 1;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"bidx += btemp * (tmp_idx[i] + pad[i]);",    // NOLINT
"btemp *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"bottom_b[bidx] = (backward_b == 1) ? top[index] : 0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(merge_copy_forward_add, Dtype)(const int_tp nthreads,",    // NOLINT
"const int_tp dims,",    // NOLINT
"__global const Dtype* bottom_a,",    // NOLINT
"const int_tp forward_a,",    // NOLINT
"__global const Dtype* bottom_b,",    // NOLINT
"const int_tp forward_b,",    // NOLINT
"__global Dtype* top,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"__global const int_tp* shape_a,",    // NOLINT
"__global const int_tp* shape_b) {",    // NOLINT
"int_tp pad[6];",    // NOLINT
"int_tp tmp_idx[6];",    // NOLINT
"int_tp size_a = 1;",    // NOLINT
"int_tp size_b = 1;",    // NOLINT
"",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"pad[i] = (shape_b[i] - shape_a[i]) / 2;",    // NOLINT
"size_a *= shape_a[i];",    // NOLINT
"size_b *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp batch_id = index / (channels * size_a);",    // NOLINT
"int_tp counter = index;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"tmp_idx[i] = counter % shape_a[i];",    // NOLINT
"counter /= shape_a[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"top[index] = 0;",    // NOLINT
"int_tp channel_id = (index / size_a) % channels;",    // NOLINT
"int_tp aidx = batch_id * channels + channel_id;",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"aidx *= shape_a[i];",    // NOLINT
"aidx += tmp_idx[i];",    // NOLINT
"}",    // NOLINT
"top[index] = forward_a ? top[index] + bottom_a[aidx] : top[index];",    // NOLINT
"int_tp bidx = (batch_id * channels + channel_id) * size_b;",    // NOLINT
"int_tp btemp = 1;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"bidx += btemp * (tmp_idx[i] + pad[i]);",    // NOLINT
"btemp *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"top[index] = forward_b ? top[index] + bottom_b[bidx] : top[index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(merge_copy_backward_add,Dtype)(const int_tp nthreads,",    // NOLINT
"const int_tp dims,",    // NOLINT
"__global Dtype* bottom_a,",    // NOLINT
"const int_tp backward_a,",    // NOLINT
"__global Dtype* bottom_b,",    // NOLINT
"const int_tp backward_b,",    // NOLINT
"__global const Dtype* top,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"__global const int_tp* shape_a,",    // NOLINT
"__global const int_tp* shape_b) {",    // NOLINT
"int_tp pad[6];",    // NOLINT
"int_tp tmp_idx[6];",    // NOLINT
"int_tp size_a = 1;",    // NOLINT
"int_tp size_b = 1;",    // NOLINT
"",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"pad[i] = (shape_b[i] - shape_a[i]) / 2;",    // NOLINT
"size_a *= shape_a[i];",    // NOLINT
"size_b *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp batch_id = index / (channels * size_a);",    // NOLINT
"int_tp counter = index;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"tmp_idx[i] = counter % shape_a[i];",    // NOLINT
"counter /= shape_a[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"int_tp channel_id = (index / size_a) % channels;",    // NOLINT
"int_tp aidx = batch_id * channels + channel_id;",    // NOLINT
"for (int_tp i = 0; i < dims; ++i) {",    // NOLINT
"aidx *= shape_a[i];",    // NOLINT
"aidx += tmp_idx[i];",    // NOLINT
"}",    // NOLINT
"bottom_a[aidx] = backward_a ? top[index] : 0;",    // NOLINT
"int_tp bidx = (batch_id * channels + channel_id) * size_b;",    // NOLINT
"int_tp btemp = 1;",    // NOLINT
"for (int_tp i = dims - 1; i >= 0; --i) {",    // NOLINT
"bidx += btemp * (tmp_idx[i] + pad[i]);",    // NOLINT
"btemp *= shape_b[i];",    // NOLINT
"}",    // NOLINT
"bottom_b[bidx] = backward_b ? top[index] : 0;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_forward,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* top_data,",    // NOLINT
"const int use_mask, __global int_tp* mask, __global Dtype* top_mask) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp pw = index % pooled_width;",    // NOLINT
"const int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"const int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"const int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"int_tp hstart = ph * stride_h - pad_h;",    // NOLINT
"int_tp wstart = pw * stride_w - pad_w;",    // NOLINT
"const int_tp hend = min(hstart + kernel_h, height);",    // NOLINT
"const int_tp wend = min(wstart + kernel_w, width);",    // NOLINT
"hstart = max(hstart, (int_tp)0);",    // NOLINT
"wstart = max(wstart, (int_tp)0);",    // NOLINT
"Dtype maxval = -FLT_MAX;",    // NOLINT
"int_tp maxidx = -1;",    // NOLINT
"__global const Dtype* bottom_slice = bottom_data",    // NOLINT
"+ (n * channels + c) * height * width;",    // NOLINT
"for (int_tp h = hstart; h < hend; ++h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; ++w) {",    // NOLINT
"if (bottom_slice[h * width + w] > maxval) {",    // NOLINT
"maxidx = h * width + w;",    // NOLINT
"maxval = bottom_slice[maxidx];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"top_data[index] = maxval;",    // NOLINT
"if (use_mask == 1) {",    // NOLINT
"mask[index] = maxidx;",    // NOLINT
"} else {",    // NOLINT
"top_mask[index] = maxidx;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ave_pool_forward,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w, __global Dtype* top_data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"{",    // NOLINT
"const int_tp pw = index % pooled_width;",    // NOLINT
"const int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"const int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"const int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"int_tp hstart = ph * stride_h - pad_h;",    // NOLINT
"int_tp wstart = pw * stride_w - pad_w;",    // NOLINT
"int_tp hend = min(hstart + kernel_h, height + pad_h);",    // NOLINT
"int_tp wend = min(wstart + kernel_w, width + pad_w);",    // NOLINT
"const int_tp pool_size = (hend - hstart) * (wend - wstart);",    // NOLINT
"hstart = max(hstart, (int_tp)0);",    // NOLINT
"wstart = max(wstart, (int_tp)0);",    // NOLINT
"hend = min(hend, height);",    // NOLINT
"wend = min(wend, width);",    // NOLINT
"Dtype aveval = 0;",    // NOLINT
"__global const Dtype* bottom_slice = bottom_data",    // NOLINT
"+ (n * channels + c) * height * width;",    // NOLINT
"for (int_tp h = hstart; h < hend; ++h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; ++w) {",    // NOLINT
"aveval += bottom_slice[h * width + w];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"top_data[index] = aveval / pool_size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sto_pool_forward_train,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"__global Dtype* rand_idx,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp pw = index % pooled_width;",    // NOLINT
"const int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"const int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"const int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"const int_tp hstart = ph * stride_h;",    // NOLINT
"const int_tp hend = min(hstart + kernel_h, height);",    // NOLINT
"const int_tp wstart = pw * stride_w;",    // NOLINT
"const int_tp wend = min(wstart + kernel_w, width);",    // NOLINT
"Dtype cumsum = 0.;",    // NOLINT
"__global const Dtype* bottom_slice = bottom_data",    // NOLINT
"+ (n * channels + c) * height * width;",    // NOLINT
"// First pass: get sum",    // NOLINT
"for (int_tp h = hstart; h < hend; ++h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; ++w) {",    // NOLINT
"cumsum += bottom_slice[h * width + w];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"const float thres = rand_idx[index] * cumsum;",    // NOLINT
"// Second pass: get value, and set index.",    // NOLINT
"cumsum = 0;",    // NOLINT
"for (int_tp h = hstart; h < hend; ++h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; ++w) {",    // NOLINT
"cumsum += bottom_slice[h * width + w];",    // NOLINT
"if (cumsum >= thres) {",    // NOLINT
"rand_idx[index] = ((n * channels + c) * height + h) * width + w;",    // NOLINT
"top_data[index] = bottom_slice[h * width + w];",    // NOLINT
"h = hend;",    // NOLINT
"w = wend;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sto_pool_forward_test,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* const bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp pw = index % pooled_width;",    // NOLINT
"const int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"const int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"const int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"const int_tp hstart = ph * stride_h;",    // NOLINT
"const int_tp hend = min(hstart + kernel_h, height);",    // NOLINT
"const int_tp wstart = pw * stride_w;",    // NOLINT
"const int_tp wend = min(wstart + kernel_w, width);",    // NOLINT
"// We set cumsum to be 0 to avoid divide-by-zero problems",    // NOLINT
"Dtype cumsum = FLT_MIN;",    // NOLINT
"Dtype cumvalues = 0.;",    // NOLINT
"__global const Dtype* bottom_slice = bottom_data",    // NOLINT
"+ (n * channels + c) * height * width;",    // NOLINT
"// First pass: get sum",    // NOLINT
"for (int_tp h = hstart; h < hend; ++h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; ++w) {",    // NOLINT
"cumsum += bottom_slice[h * width + w];",    // NOLINT
"cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"top_data[index] = cumvalues / cumsum;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_backward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* top_diff,",    // NOLINT
"const int use_mask,",    // NOLINT
"__global const int_tp* mask,",    // NOLINT
"__global const Dtype* top_mask,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp height,",    // NOLINT
"const int_tp width,",    // NOLINT
"const int_tp pooled_height,",    // NOLINT
"const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w,",    // NOLINT
"const int_tp stride_h,",    // NOLINT
"const int_tp stride_w,",    // NOLINT
"const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local index",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp c = (index / width / height) % channels;",    // NOLINT
"const int_tp n = index / width / height / channels;",    // NOLINT
"const int_tp phstart =",    // NOLINT
"(h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;",    // NOLINT
"const int_tp phend = min((h + pad_h) / stride_h + 1, pooled_height);",    // NOLINT
"const int_tp pwstart =",    // NOLINT
"(w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;",    // NOLINT
"const int_tp pwend = min((w + pad_w) / stride_w + 1, pooled_width);",    // NOLINT
"Dtype gradient = 0;",    // NOLINT
"const int_tp offset = (n * channels + c) * pooled_height * pooled_width;",    // NOLINT
"__global const Dtype* top_diff_slice = top_diff + offset;",    // NOLINT
"if (use_mask == 1) {",    // NOLINT
"__global const int_tp* mask_slice = mask + offset;",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"if (mask_slice[ph * pooled_width + pw] == h * width + w) {",    // NOLINT
"gradient += top_diff_slice[ph * pooled_width + pw];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"} else {",    // NOLINT
"__global const Dtype* top_mask_slice = top_mask + offset;",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {",    // NOLINT
"gradient += top_diff_slice[ph * pooled_width + pw];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ave_pool_backward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* top_diff,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp height,",    // NOLINT
"const int_tp width,",    // NOLINT
"const int_tp pooled_height,",    // NOLINT
"const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w,",    // NOLINT
"const int_tp stride_h,",    // NOLINT
"const int_tp stride_w,",    // NOLINT
"const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local index",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width + pad_w;",    // NOLINT
"const int_tp h = (index / width) % height + pad_h;",    // NOLINT
"const int_tp c = (index / width / height) % channels;",    // NOLINT
"const int_tp n = index / width / height / channels;",    // NOLINT
"const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;",    // NOLINT
"const int_tp phend = min(h / stride_h + 1, pooled_height);",    // NOLINT
"const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;",    // NOLINT
"const int_tp pwend = min(w / stride_w + 1, pooled_width);",    // NOLINT
"Dtype gradient = 0.0;",    // NOLINT
"__global const Dtype* const top_diff_slice = top_diff",    // NOLINT
"+ (n * channels + c) * pooled_height * pooled_width;",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"// figure out the pooling size",    // NOLINT
"int_tp hstart = ph * stride_h - pad_h;",    // NOLINT
"int_tp wstart = pw * stride_w - pad_w;",    // NOLINT
"int_tp hend = min(hstart + kernel_h, height + pad_h);",    // NOLINT
"int_tp wend = min(wstart + kernel_w, width + pad_w);",    // NOLINT
"int_tp pool_size = (hend - hstart) * (wend - wstart);",    // NOLINT
"gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sto_pool_backward,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* rand_idx,",    // NOLINT
"__global const Dtype* const top_diff, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h, const int_tp kernel_w, const int_tp stride_h,",    // NOLINT
"const int_tp stride_w, __global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"// find out the local index",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp c = (index / width / height) % channels;",    // NOLINT
"const int_tp n = index / width / height / channels;",    // NOLINT
"const int_tp phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;",    // NOLINT
"const int_tp phend = min(h / stride_h + 1, pooled_height);",    // NOLINT
"const int_tp pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;",    // NOLINT
"const int_tp pwend = min(w / stride_w + 1, pooled_width);",    // NOLINT
"Dtype gradient = 0.0;",    // NOLINT
"__global const Dtype* rand_idx_slice = rand_idx",    // NOLINT
"+ (n * channels + c) * pooled_height * pooled_width;",    // NOLINT
"__global const Dtype* top_diff_slice = top_diff",    // NOLINT
"+ (n * channels + c) * pooled_height * pooled_width;",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"gradient += top_diff_slice[ph * pooled_width + pw]",    // NOLINT
"* (index == (int_tp) (rand_idx_slice[ph * pooled_width + pw])?1.0:0.0);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_forward_nd, Dtype)(const int_tp n,",    // NOLINT
"const int_tp num_axes,",    // NOLINT
"__global const Dtype* bottom_data,",    // NOLINT
"const int_tp channels,",    // NOLINT
"__global const int_tp* size,",    // NOLINT
"__global const int_tp* pooled_size,",    // NOLINT
"__global const int_tp* kernel_size,",    // NOLINT
"__global const int_tp* ext_kernel_size,",    // NOLINT
"__global const int_tp* stride,",    // NOLINT
"__global const int_tp* dilation,",    // NOLINT
"__global const int_tp* pad,",    // NOLINT
"__global Dtype* top_data,",    // NOLINT
"const int use_mask,",    // NOLINT
"__global int_tp* mask, __global Dtype* top_mask) {",    // NOLINT
"int_tp d_idx[6];",    // NOLINT
"int_tp d_start[6];",    // NOLINT
"int_tp d_end[6];",    // NOLINT
"int_tp d_iter[6];",    // NOLINT
"int_tp i;",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"int_tp offset = 1;",    // NOLINT
"int_tp num = index;",    // NOLINT
"",    // NOLINT
"bool do_continue = false;",    // NOLINT
"",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"d_idx[i] = num % pooled_size[i];",    // NOLINT
"d_start[i] = d_idx[i] * stride[i] - pad[i];",    // NOLINT
"d_end[i] = min(d_start[i] + ext_kernel_size[i], size[i]);",    // NOLINT
"while (d_start[i] < 0) {",    // NOLINT
"d_start[i] += dilation[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"num /= pooled_size[i];",    // NOLINT
"offset *= size[i];",    // NOLINT
"d_iter[i] = d_start[i];",    // NOLINT
"",    // NOLINT
"if (d_start[i] >= d_end[i]) {",    // NOLINT
"top_data[index] = -FLT_MAX;",    // NOLINT
"if (use_mask) {",    // NOLINT
"mask[index] = -1;",    // NOLINT
"} else {",    // NOLINT
"top_mask[index] = -1;",    // NOLINT
"}",    // NOLINT
"do_continue = true;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(do_continue) {",    // NOLINT
"continue;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"int_tp chan = num % channels;",    // NOLINT
"num /= channels;",    // NOLINT
"offset *= (num * channels + chan);",    // NOLINT
"",    // NOLINT
"Dtype maxval = -FLT_MAX;",    // NOLINT
"int_tp maxidx = -1;",    // NOLINT
"int_tp final_offset = 0;",    // NOLINT
"",    // NOLINT
"bool incremented;",    // NOLINT
"do {",    // NOLINT
"final_offset = 0;",    // NOLINT
"int_tp size_prod = 1;",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"final_offset += d_iter[i] * size_prod;",    // NOLINT
"size_prod *= size[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if (bottom_data[final_offset + offset] > maxval) {",    // NOLINT
"maxidx = final_offset;",    // NOLINT
"maxval = bottom_data[offset + final_offset];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"incremented = false;",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"if (d_iter[i] >= d_end[i] - dilation[i]) {",    // NOLINT
"d_iter[i] = d_start[i];",    // NOLINT
"} else {",    // NOLINT
"d_iter[i] += dilation[i];",    // NOLINT
"incremented = true;",    // NOLINT
"break;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"} while (incremented);",    // NOLINT
"",    // NOLINT
"top_data[index] = maxval;",    // NOLINT
"if (use_mask == 1) {",    // NOLINT
"mask[index] = maxidx;",    // NOLINT
"} else {",    // NOLINT
"top_mask[index] = maxidx;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_backward_nd, Dtype)(const int_tp n,",    // NOLINT
"const int_tp num_axes,",    // NOLINT
"__global const Dtype* top_diff,",    // NOLINT
"const int use_mask,",    // NOLINT
"__global const int_tp* mask,",    // NOLINT
"__global const Dtype* top_mask,",    // NOLINT
"const int_tp channels,",    // NOLINT
"__global const int_tp* size,",    // NOLINT
"__global const int_tp* pooled_size,",    // NOLINT
"__global const int_tp* kernel_size,",    // NOLINT
"__global const int_tp* ext_kernel_size,",    // NOLINT
"__global const int_tp* stride,",    // NOLINT
"__global const int_tp* dilation,",    // NOLINT
"__global const int_tp* pad,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"int_tp d_idx[6];",    // NOLINT
"int_tp d_start[6];",    // NOLINT
"int_tp d_end[6];",    // NOLINT
"int_tp d_iter[6];",    // NOLINT
"int_tp i;",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"// find out the local index",    // NOLINT
"// find out the local offset",    // NOLINT
"int_tp offset = 1;",    // NOLINT
"int_tp num = index;",    // NOLINT
"",    // NOLINT
"bool do_continue = false;",    // NOLINT
"",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"d_idx[i] = num % size[i];",    // NOLINT
"d_start[i] =",    // NOLINT
"(d_idx[i] + pad[i] < ext_kernel_size[i]) ?",    // NOLINT
"0 : (d_idx[i] + pad[i] - ext_kernel_size[i]) / stride[i] + 1;",    // NOLINT
"d_end[i] = min((int_tp) ((d_idx[i] + pad[i]) / stride[i]),",    // NOLINT
"(int_tp) (pooled_size[i] - 1));",    // NOLINT
"num /= size[i];",    // NOLINT
"offset *= pooled_size[i];",    // NOLINT
"d_iter[i] = d_start[i];",    // NOLINT
"",    // NOLINT
"if (d_start[i] > d_end[i]) {",    // NOLINT
"bottom_diff[index] = 0;",    // NOLINT
"do_continue = true;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if (do_continue) {",    // NOLINT
"continue;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"int_tp chan = num % channels;",    // NOLINT
"num /= channels;",    // NOLINT
"offset *= (num * channels + chan);",    // NOLINT
"",    // NOLINT
"Dtype gradient = 0.0;",    // NOLINT
"int_tp final_offset = 0;",    // NOLINT
"int_tp im_offset = 0;",    // NOLINT
"",    // NOLINT
"bool incremented;",    // NOLINT
"do {",    // NOLINT
"final_offset = offset;",    // NOLINT
"im_offset = 0;",    // NOLINT
"int_tp size_prod = 1;",    // NOLINT
"int_tp pooled_size_prod = 1;",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"final_offset += d_iter[i] * pooled_size_prod;",    // NOLINT
"im_offset += d_idx[i] * size_prod;",    // NOLINT
"size_prod *= size[i];",    // NOLINT
"pooled_size_prod *= pooled_size[i];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if (use_mask) {",    // NOLINT
"if (mask[final_offset] == im_offset) {",    // NOLINT
"gradient += top_diff[final_offset];",    // NOLINT
"}",    // NOLINT
"} else {",    // NOLINT
"if (top_mask[final_offset] == im_offset) {",    // NOLINT
"gradient += top_diff[final_offset];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"incremented = false;",    // NOLINT
"for (i = num_axes - 1; i >= 0; --i) {",    // NOLINT
"if (d_iter[i] >= d_end[i]) {",    // NOLINT
"d_iter[i] = d_start[i];",    // NOLINT
"} else {",    // NOLINT
"++d_iter[i];",    // NOLINT
"incremented = true;",    // NOLINT
"break;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"} while (incremented);",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_forward_sk,Dtype)(const int_tp nthreads,",    // NOLINT
"__global Dtype* bottom_data,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp height,",    // NOLINT
"const int_tp width,",    // NOLINT
"const int_tp pooled_height,",    // NOLINT
"const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w,",    // NOLINT
"const int_tp ext_kernel_h,",    // NOLINT
"const int_tp ext_kernel_w,",    // NOLINT
"const int_tp stride_h,",    // NOLINT
"const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h,",    // NOLINT
"const int_tp dilation_w,",    // NOLINT
"const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* top_data,",    // NOLINT
"const int use_mask,",    // NOLINT
"__global int_tp* mask,",    // NOLINT
"__global Dtype* top_mask) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp pw = index % pooled_width;",    // NOLINT
"int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"int_tp hstart = ph * stride_h - pad_h;",    // NOLINT
"int_tp wstart = pw * stride_w - pad_w;",    // NOLINT
"int_tp hend = min(hstart + ext_kernel_h, height);",    // NOLINT
"int_tp wend = min(wstart + ext_kernel_w, width);",    // NOLINT
"while (hstart < 0) {",    // NOLINT
"hstart += dilation_h;",    // NOLINT
"}",    // NOLINT
"while (wstart < 0) {",    // NOLINT
"wstart += dilation_w;",    // NOLINT
"}",    // NOLINT
"Dtype maxval = -FLT_MAX;",    // NOLINT
"int_tp maxidx = -1;",    // NOLINT
"__global Dtype* bottom_data_ptr = bottom_data",    // NOLINT
"+ (n * channels + c) * height * width;",    // NOLINT
"for (int_tp h = hstart; h < hend; h += dilation_h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; w += dilation_w) {",    // NOLINT
"if (bottom_data_ptr[h * width + w] > maxval) {",    // NOLINT
"maxidx = h * width + w;",    // NOLINT
"maxval = bottom_data_ptr[maxidx];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"top_data[index] = maxval;",    // NOLINT
"if (use_mask == 1) {",    // NOLINT
"mask[index] = maxidx;",    // NOLINT
"} else {",    // NOLINT
"top_mask[index] = maxidx;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_backward_sk,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* top_diff, const int use_mask,",    // NOLINT
"__global const int_tp* mask, __global const Dtype* top_mask,",    // NOLINT
"const int_tp num, const int_tp channels, const int_tp height,",    // NOLINT
"const int_tp width, const int_tp pooled_height, const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,",    // NOLINT
"const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h, const int_tp dilation_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"",    // NOLINT
"__global const int_tp* mask_ptr = mask;",    // NOLINT
"__global const Dtype* top_diff_ptr = top_diff;",    // NOLINT
"",    // NOLINT
"// find out the local index",    // NOLINT
"// find out the local offset",    // NOLINT
"int_tp w = index % width;",    // NOLINT
"int_tp h = (index / width) % height;",    // NOLINT
"int_tp c = (index / width / height) % channels;",    // NOLINT
"int_tp n = index / width / height / channels;",    // NOLINT
"",    // NOLINT
"int_tp phstart =",    // NOLINT
"(h + pad_h < ext_kernel_h) ? 0 : (h + pad_h - ext_kernel_h) / stride_h + 1;",    // NOLINT
"int_tp phend = min(((h + pad_h) / stride_h + 1),",    // NOLINT
"pooled_height);",    // NOLINT
"int_tp pwstart =",    // NOLINT
"(w + pad_w < ext_kernel_w) ? 0 : (w + pad_w - ext_kernel_w) / stride_w + 1;",    // NOLINT
"int_tp pwend = min(((w + pad_w) / stride_w + 1),",    // NOLINT
"pooled_width);",    // NOLINT
"",    // NOLINT
"Dtype gradient = 0.0;",    // NOLINT
"int_tp offset = (n * channels + c) * pooled_height * pooled_width;",    // NOLINT
"top_diff_ptr += offset;",    // NOLINT
"if (use_mask == 1) {",    // NOLINT
"mask_ptr += offset;",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"if (mask_ptr[ph * pooled_width + pw] == h * width + w) {",    // NOLINT
"gradient += top_diff_ptr[ph * pooled_width + pw];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"} else {",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"if (top_mask[ph * pooled_width + pw] == h * width + w) {",    // NOLINT
"gradient += top_diff_ptr[ph * pooled_width + pw];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ave_pool_forward_sk,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,",    // NOLINT
"const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h, const int_tp dilation_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"",    // NOLINT
"int_tp pool_size = 0;",    // NOLINT
"int_tp pw = index % pooled_width;",    // NOLINT
"int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"int_tp hstart = ph * stride_h - pad_h;",    // NOLINT
"int_tp wstart = pw * stride_w - pad_w;",    // NOLINT
"int_tp hend = hstart + ext_kernel_h;",    // NOLINT
"int_tp wend = wstart + ext_kernel_w;",    // NOLINT
"// Overspill over the image + pad does",    // NOLINT
"// not contribute to pool size",    // NOLINT
"while (hend > height + pad_h) {",    // NOLINT
"hend -= dilation_h;",    // NOLINT
"}",    // NOLINT
"while (wend > width + pad_w) {",    // NOLINT
"wend -= dilation_w;",    // NOLINT
"}",    // NOLINT
"Dtype aveval = 0;",    // NOLINT
"__global const Dtype* bottom_data_ptr = bottom_data;",    // NOLINT
"bottom_data_ptr += (n * channels + c) * height * width;",    // NOLINT
"for (int_tp h = hstart; h < hend; h += dilation_h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; w += dilation_w) {",    // NOLINT
"if (h >= 0 && h < height && w >= 0 && w < width) {",    // NOLINT
"aveval += bottom_data_ptr[h * width + w];",    // NOLINT
"}",    // NOLINT
"++pool_size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"top_data[index] = aveval / pool_size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ave_pool_backward_sk,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* top_diff,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp height,",    // NOLINT
"const int_tp width,",    // NOLINT
"const int_tp pooled_height,",    // NOLINT
"const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w,",    // NOLINT
"const int_tp ext_kernel_h,",    // NOLINT
"const int_tp ext_kernel_w,",    // NOLINT
"const int_tp stride_h,",    // NOLINT
"const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h,",    // NOLINT
"const int_tp dilation_w,",    // NOLINT
"const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"// find out the local index",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp w = index % width;",    // NOLINT
"const int_tp h = (index / width) % height;",    // NOLINT
"const int_tp c = (index / width / height) % channels;",    // NOLINT
"const int_tp n = index / width / height / channels;",    // NOLINT
"int_tp phstart =",    // NOLINT
"(h + pad_h < ext_kernel_h) ? 0 : (h + pad_h - ext_kernel_h) / stride_h + 1;",    // NOLINT
"int_tp phend = min(((h + pad_h) / stride_h + 1),",    // NOLINT
"pooled_height);",    // NOLINT
"int_tp pwstart =",    // NOLINT
"(w + pad_w < ext_kernel_w) ? 0 : (w + pad_w - ext_kernel_w) / stride_w + 1;",    // NOLINT
"int_tp pwend = min(((w + pad_w) / stride_w + 1),",    // NOLINT
"pooled_width);",    // NOLINT
"Dtype gradient = 0.0;",    // NOLINT
"__global const Dtype* const top_diff_slice = top_diff",    // NOLINT
"+ (n * channels + c) * pooled_height * pooled_width;",    // NOLINT
"for (int_tp ph = phstart; ph < phend; ++ph) {",    // NOLINT
"for (int_tp pw = pwstart; pw < pwend; ++pw) {",    // NOLINT
"// figure out the pooling size",    // NOLINT
"int_tp hstart = ph * stride_h - pad_h;",    // NOLINT
"int_tp wstart = pw * stride_w - pad_w;",    // NOLINT
"int_tp hend = min(hstart + ext_kernel_h, height + pad_h);",    // NOLINT
"int_tp wend = min(wstart + ext_kernel_w, width + pad_w);",    // NOLINT
"int_tp pool_size =",    // NOLINT
"((hend - hstart - 1) / dilation_h + 1) *",    // NOLINT
"((wend - wstart - 1) / dilation_w + 1);",    // NOLINT
"if (h >= hstart && h < hend &&",    // NOLINT
"(h - hstart) % dilation_h == 0 &&",    // NOLINT
"w >= wstart && w < wend &&",    // NOLINT
"(w - wstart) % dilation_w == 0) {",    // NOLINT
"gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"bottom_diff[index] = gradient;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sto_pool_forward_train_sk,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,",    // NOLINT
"const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h, const int_tp dilation_w, __global Dtype* rand_idx,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp pw = index % pooled_width;",    // NOLINT
"int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"int_tp hstart = ph * stride_h;",    // NOLINT
"int_tp hend = min(hstart + ext_kernel_h, height);",    // NOLINT
"int_tp wstart = pw * stride_w;",    // NOLINT
"int_tp wend = min(wstart + ext_kernel_w, width);",    // NOLINT
"Dtype cumsum = 0.;",    // NOLINT
"__global const Dtype* bottom_data_ptr = bottom_data;",    // NOLINT
"bottom_data_ptr += (n * channels + c) * height * width;",    // NOLINT
"// First pass: get sum",    // NOLINT
"for (int_tp h = hstart; h < hend; h += dilation_h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; w += dilation_w) {",    // NOLINT
"cumsum += bottom_data_ptr[h * width + w];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"float thres = rand_idx[index] * cumsum;",    // NOLINT
"// Second pass: get value, and set index.",    // NOLINT
"cumsum = 0;",    // NOLINT
"for (int_tp h = hstart; h < hend; h += dilation_h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; w += dilation_w) {",    // NOLINT
"cumsum += bottom_data_ptr[h * width + w];",    // NOLINT
"if (cumsum >= thres) {",    // NOLINT
"rand_idx[index] = ((n * channels + c) * height + h) * width + w;",    // NOLINT
"top_data[index] = bottom_data_ptr[h * width + w];",    // NOLINT
"h = hend;",    // NOLINT
"w = wend;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sto_pool_forward_test_sk,Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width,",    // NOLINT
"const int_tp kernel_h, const int_tp kernel_w, const int_tp ext_kernel_h,",    // NOLINT
"const int_tp ext_kernel_w, const int_tp stride_h, const int_tp stride_w,",    // NOLINT
"const int_tp dilation_h, const int_tp dilation_w,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp pw = index % pooled_width;",    // NOLINT
"int_tp ph = (index / pooled_width) % pooled_height;",    // NOLINT
"int_tp c = (index / pooled_width / pooled_height) % channels;",    // NOLINT
"int_tp n = index / pooled_width / pooled_height / channels;",    // NOLINT
"int_tp hstart = ph * stride_h;",    // NOLINT
"int_tp hend = min(hstart + ext_kernel_h, height);",    // NOLINT
"int_tp wstart = pw * stride_w;",    // NOLINT
"int_tp wend = min(wstart + ext_kernel_w, width);",    // NOLINT
"// We set cumsum to be 0 to avoid divide-by-zero problems",    // NOLINT
"Dtype cumsum = FLT_MIN;",    // NOLINT
"Dtype cumvalues = 0.;",    // NOLINT
"__global const Dtype* bottom_data_ptr = bottom_data;",    // NOLINT
"bottom_data_ptr += (n * channels + c) * height * width;",    // NOLINT
"// First pass: get sum",    // NOLINT
"for (int_tp h = hstart; h < hend; h += dilation_h) {",    // NOLINT
"for (int_tp w = wstart; w < wend; w += dilation_w) {",    // NOLINT
"cumsum += bottom_data_ptr[h * width + w];",    // NOLINT
"cumvalues += bottom_data_ptr[h * width + w]",    // NOLINT
"* bottom_data_ptr[h * width + w];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"top_data[index] = cumvalues / cumsum;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(slice,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"const int forward, const int_tp num_slices,",    // NOLINT
"const int_tp slice_size,",    // NOLINT
"const int_tp bottom_slice_axis,",    // NOLINT
"const int_tp top_slice_axis,",    // NOLINT
"const int_tp offset_slice_axis,",    // NOLINT
"__global Dtype* out_data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp total_slice_size = slice_size * top_slice_axis;",    // NOLINT
"const int_tp slice_num = index / total_slice_size;",    // NOLINT
"const int_tp slice_index = index % total_slice_size;",    // NOLINT
"const int_tp bottom_index = slice_index",    // NOLINT
"+ (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;",    // NOLINT
"if (forward == 1) {",    // NOLINT
"out_data[index] = in_data[bottom_index];",    // NOLINT
"} else {",    // NOLINT
"out_data[bottom_index] = in_data[index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if defined(cl_intel_subgroups)",    // NOLINT
"#pragma OPENCL EXTENSION  cl_intel_subgroups : enable",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(softmax_forward_slm,Dtype)(const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"__global Dtype* scale,",    // NOLINT
"__global const Dtype* data,",    // NOLINT
"__global Dtype* out,",    // NOLINT
"__local Dtype *out_tmp,",    // NOLINT
"__local Dtype *scale_tmp,",    // NOLINT
"__local Dtype *group_tmp) {",    // NOLINT
"",    // NOLINT
"int_tp n = get_global_id(1);",    // NOLINT
"for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=",    // NOLINT
"get_global_size(0), ++s) {",    // NOLINT
"float maxval = -FLT_MAX;",    // NOLINT
"for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {",    // NOLINT
"Dtype tmp = data[(n * channels + c) * spatial_dim + s];",    // NOLINT
"maxval = max((Dtype)tmp, (Dtype)maxval);",    // NOLINT
"}",    // NOLINT
"maxval = sub_group_reduce_max(maxval);",    // NOLINT
"//if (get_sub_group_local_id() == 0)",    // NOLINT
"group_tmp[get_sub_group_id() * spatial_dim + s] = maxval;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp s = index / get_max_sub_group_size();",    // NOLINT
"Dtype maxval = sub_group_reduce_max(group_tmp[get_sub_group_local_id() * spatial_dim + s]);",    // NOLINT
"//if (get_sub_group_local_id() == 0)",    // NOLINT
"scale_tmp[s] = maxval;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < channels * spatial_dim;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"out_tmp[index] = exp(data[n * channels * spatial_dim + index] - scale_tmp[s]);",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=",    // NOLINT
"get_global_size(0), ++s) {",    // NOLINT
"Dtype sum = 0;",    // NOLINT
"for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {",    // NOLINT
"sum += out_tmp[c * spatial_dim + s];",    // NOLINT
"}",    // NOLINT
"sum = sub_group_reduce_add(sum);",    // NOLINT
"group_tmp[get_sub_group_id() * spatial_dim + s] = sum;",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp s = index / get_max_sub_group_size();",    // NOLINT
"Dtype sum = sub_group_reduce_add(group_tmp[get_sub_group_local_id() * spatial_dim + s]);",    // NOLINT
"//if (get_sub_group_local_id() == 0)",    // NOLINT
"scale_tmp[s] = sum;",    // NOLINT
"}",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < channels * spatial_dim;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"out[n * channels * spatial_dim + index] = out_tmp[index] / scale_tmp[s];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(softmax_forward,Dtype)(const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"__global Dtype* scale,",    // NOLINT
"__global const Dtype* data,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"",    // NOLINT
"int_tp n = get_global_id(1);",    // NOLINT
"__global Dtype *group_tmp = scale + spatial_dim * num + n * get_max_sub_group_size() * spatial_dim;",    // NOLINT
"for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=",    // NOLINT
"get_global_size(0), ++s) {",    // NOLINT
"float maxval = -FLT_MAX;",    // NOLINT
"for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {",    // NOLINT
"Dtype tmp = data[(n * channels + c) * spatial_dim + s];",    // NOLINT
"maxval = max((Dtype)tmp, (Dtype)maxval);",    // NOLINT
"}",    // NOLINT
"maxval = sub_group_reduce_max(maxval);",    // NOLINT
"//if (get_sub_group_local_id() == 0)",    // NOLINT
"group_tmp[get_sub_group_id() * spatial_dim + s] = maxval;",    // NOLINT
"}",    // NOLINT
"barrier(CLK_GLOBAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp s = index / get_max_sub_group_size();",    // NOLINT
"Dtype maxval = sub_group_reduce_max(group_tmp[get_sub_group_local_id() * spatial_dim + s]);",    // NOLINT
"//if (get_sub_group_local_id() == 0)",    // NOLINT
"scale[n * spatial_dim + s] = maxval;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"barrier(CLK_GLOBAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < channels * spatial_dim;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"out[n * channels * spatial_dim + index] = exp(data[n * channels * spatial_dim + index] - scale[n * spatial_dim + s]);",    // NOLINT
"}",    // NOLINT
"barrier(CLK_GLOBAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0), s = 0; index < spatial_dim * get_local_size(0); index +=",    // NOLINT
"get_global_size(0), ++s) {",    // NOLINT
"Dtype sum = 0;",    // NOLINT
"for (int_tp c = get_global_id(0); c < channels; c += get_global_size(0)) {",    // NOLINT
"sum += out[n * channels * spatial_dim + c * spatial_dim + s];",    // NOLINT
"}",    // NOLINT
"sum = sub_group_reduce_add(sum);",    // NOLINT
"group_tmp[get_sub_group_id() * spatial_dim + s] = sum;",    // NOLINT
"}",    // NOLINT
"barrier(CLK_GLOBAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < spatial_dim * get_max_sub_group_size(); index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"int_tp s = index / get_max_sub_group_size();",    // NOLINT
"Dtype sum = sub_group_reduce_add(group_tmp[get_sub_group_local_id() * spatial_dim + s]);",    // NOLINT
"//if (get_sub_group_local_id() == 0)",    // NOLINT
"scale[n * spatial_dim + s] = sum;",    // NOLINT
"}",    // NOLINT
"barrier(CLK_GLOBAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < channels * spatial_dim;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"int_tp s = index % spatial_dim;",    // NOLINT
"out[n * channels * spatial_dim + index] /= scale[n * spatial_dim + s];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Copied from caffe.pb.h, must keep consistent with the original definition",    // NOLINT
"#ifndef __SOFTMAX_LOSS_CL__",    // NOLINT
"#define __SOFTMAX_LOSS_CL__",    // NOLINT
"enum LossParameter_NormalizationMode {",    // NOLINT
"LossParameter_NormalizationMode_FULL = 0,",    // NOLINT
"LossParameter_NormalizationMode_VALID = 1,",    // NOLINT
"LossParameter_NormalizationMode_BATCH_SIZE = 2,",    // NOLINT
"LossParameter_NormalizationMode_NONE = 3",    // NOLINT
"};",    // NOLINT
"#endif",    // NOLINT
"// Copied from softmax_loss_layer.cpp, must keep consistent with the original implementation",    // NOLINT
"Dtype TEMPLATE(get_normalizer, Dtype)(",    // NOLINT
"enum LossParameter_NormalizationMode normalization_mode, int_tp valid_count,",    // NOLINT
"int_tp outer_num_, int_tp inner_num_) {",    // NOLINT
"Dtype normalizer;",    // NOLINT
"switch (normalization_mode) {",    // NOLINT
"case LossParameter_NormalizationMode_FULL:",    // NOLINT
"normalizer = (Dtype)(outer_num_ * inner_num_);",    // NOLINT
"break;",    // NOLINT
"case LossParameter_NormalizationMode_VALID:",    // NOLINT
"if (valid_count == -1) {",    // NOLINT
"normalizer = (Dtype)(outer_num_ * inner_num_);",    // NOLINT
"} else {",    // NOLINT
"normalizer = (Dtype)(valid_count);",    // NOLINT
"}",    // NOLINT
"break;",    // NOLINT
"case LossParameter_NormalizationMode_BATCH_SIZE:",    // NOLINT
"normalizer = (Dtype)(outer_num_);",    // NOLINT
"break;",    // NOLINT
"case LossParameter_NormalizationMode_NONE:",    // NOLINT
"normalizer = (Dtype)(1);",    // NOLINT
"break;",    // NOLINT
"default:",    // NOLINT
"normalizer = (Dtype)(0);",    // NOLINT
"}",    // NOLINT
"// Some users will have no labels for some examples in order to 'turn off' a",    // NOLINT
"// particular loss in a multi-task setup. The max prevents NaNs in that case.",    // NOLINT
"return fmax((Dtype)(1.0), normalizer);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"Dtype TEMPLATE(asum, Dtype)(int_tp n, __global const Dtype *data, __local Dtype *sum_tmp) {",    // NOLINT
"Dtype sum = 0;",    // NOLINT
"for(int_tp i = get_global_id(0); i < n; i += get_global_size(0)) {",    // NOLINT
"sum += data[i];",    // NOLINT
"}",    // NOLINT
"sum = sub_group_reduce_add(sum);",    // NOLINT
"sum_tmp[get_sub_group_id()] = sum;",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if (get_sub_group_id() == 0)",    // NOLINT
"sum = sub_group_reduce_add(sum_tmp[get_sub_group_local_id()]);",    // NOLINT
"return sum;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(softmax_loss_forward_asum, Dtype)(",    // NOLINT
"int_tp n, int_tp outer_num_, int_tp inner_num_,",    // NOLINT
"int_tp compute_count_sum, int_tp normalization_type,",    // NOLINT
"__global const Dtype *loss,",    // NOLINT
"__global const Dtype *counts, __global Dtype *out) {",    // NOLINT
"__local Dtype sum_tmp[16];",    // NOLINT
"",    // NOLINT
"Dtype loss_sum = TEMPLATE(asum, Dtype)(n, loss, sum_tmp);",    // NOLINT
"Dtype counts_sum = -1;",    // NOLINT
"if (compute_count_sum)",    // NOLINT
"counts_sum = TEMPLATE(asum, Dtype)(n, counts, sum_tmp);",    // NOLINT
"",    // NOLINT
"if (get_global_id(0) == 0)",    // NOLINT
"out[0] = loss_sum / TEMPLATE(get_normalizer, Dtype)(normalization_type, counts_sum, outer_num_, inner_num_);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(softmax_loss_forward,Dtype)(",    // NOLINT
"int_tp n, __global const Dtype* prob_data, __global const Dtype* label,",    // NOLINT
"__global Dtype* loss,",    // NOLINT
"const int_tp num, const int_tp dim, const int_tp spatial_dim,",    // NOLINT
"const int has_ignore_label_, const int_tp ignore_label_,",    // NOLINT
"__global Dtype* counts) {",    // NOLINT
"",    // NOLINT
"for (int_tp index = get_global_id(0); index < n;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp n = index / spatial_dim;",    // NOLINT
"const int_tp s = index % spatial_dim;",    // NOLINT
"const int_tp label_value = (int_tp) (label[n * spatial_dim + s]);",    // NOLINT
"if (has_ignore_label_ == 1 && label_value == ignore_label_) {",    // NOLINT
"loss[index] = 0;",    // NOLINT
"counts[index] = 0;",    // NOLINT
"} else {",    // NOLINT
"loss[index] = -log((Dtype)(",    // NOLINT
"max((Dtype) (prob_data[n * dim + label_value * spatial_dim + s]),",    // NOLINT
"(Dtype) FLT_MIN)));",    // NOLINT
"counts[index] = 1;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(softmax_loss_backward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* top,",    // NOLINT
"__global const Dtype* label,",    // NOLINT
"__global Dtype* bottom_diff,",    // NOLINT
"const int_tp num,",    // NOLINT
"const int_tp dim,",    // NOLINT
"const int_tp spatial_dim,",    // NOLINT
"const int has_ignore_label_,",    // NOLINT
"const int_tp ignore_label_,",    // NOLINT
"__global Dtype* counts) {",    // NOLINT
"const int_tp channels = dim / spatial_dim;",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads; index +=",    // NOLINT
"get_global_size(0)) {",    // NOLINT
"const int_tp n = index / spatial_dim;",    // NOLINT
"const int_tp s = index % spatial_dim;",    // NOLINT
"const int_tp label_value = (int_tp) (label[n * spatial_dim + s]);",    // NOLINT
"if (has_ignore_label_ == 1 && label_value == ignore_label_) {",    // NOLINT
"for (int_tp c = 0; c < channels; ++c) {",    // NOLINT
"bottom_diff[n * dim + c * spatial_dim + s] = 0;",    // NOLINT
"}",    // NOLINT
"counts[index] = 0;",    // NOLINT
"} else {",    // NOLINT
"bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;",    // NOLINT
"counts[index] = 1;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ada_delta_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"__global Dtype* h2,",    // NOLINT
"Dtype momentum,",    // NOLINT
"Dtype delta,",    // NOLINT
"Dtype local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype gi = g[i];",    // NOLINT
"Dtype hi = h[i] = momentum * h[i] + (1.0 - momentum) * gi * gi;",    // NOLINT
"gi = gi * sqrt((h2[i] + delta) / (hi + delta));",    // NOLINT
"h2[i] = momentum * h2[i] + (1.0 - momentum) * gi * gi;",    // NOLINT
"g[i] = local_rate * gi;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ada_grad_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"Dtype delta,",    // NOLINT
"Dtype local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype gi = g[i];",    // NOLINT
"Dtype hi = h[i] = h[i] + gi * gi;",    // NOLINT
"g[i] = local_rate * gi / (sqrt(hi) + delta);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(adam_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* m,",    // NOLINT
"__global Dtype* v,",    // NOLINT
"Dtype beta1,",    // NOLINT
"Dtype beta2,",    // NOLINT
"Dtype eps_hat,",    // NOLINT
"Dtype corrected_local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype gi = g[i];",    // NOLINT
"Dtype mi = m[i] = m[i] * beta1 + gi * (1 - beta1);",    // NOLINT
"Dtype vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);",    // NOLINT
"g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(nesterov_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"Dtype momentum,",    // NOLINT
"Dtype local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype hi = h[i];",    // NOLINT
"Dtype hi_new = h[i] = momentum * hi + local_rate * g[i];",    // NOLINT
"g[i] = (1 + momentum) * hi_new - momentum * hi;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(rms_prop_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"Dtype rms_decay,",    // NOLINT
"Dtype delta,",    // NOLINT
"Dtype local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype gi = g[i];",    // NOLINT
"Dtype hi = h[i] = rms_decay * h[i] + (1 - rms_decay) * gi * gi;",    // NOLINT
"g[i] = local_rate * g[i] / (sqrt(hi) + delta);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sgd_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"Dtype momentum,",    // NOLINT
"Dtype local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"g[i] = h[i] = momentum * h[i] + local_rate * g[i];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(tile,Dtype)(const int_tp nthreads, __global const Dtype* bottom_data,",    // NOLINT
"const int_tp tile_size, const int_tp num_tiles,",    // NOLINT
"const int_tp bottom_tile_axis,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp d = index % tile_size;",    // NOLINT
"const int_tp b = (index / tile_size / num_tiles) % bottom_tile_axis;",    // NOLINT
"const int_tp n = index / tile_size / num_tiles / bottom_tile_axis;",    // NOLINT
"const int_tp bottom_index = (n * bottom_tile_axis + b) * tile_size + d;",    // NOLINT
"top_data[index] = bottom_data[bottom_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(tile_backward,Dtype)(const int_tp nthreads,",    // NOLINT
"__global const Dtype* top_diff,",    // NOLINT
"const int_tp tile_size,",    // NOLINT
"const int_tp num_tiles,",    // NOLINT
"const int_tp bottom_tile_axis,",    // NOLINT
"__global Dtype* bottom_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"const int_tp d = index % tile_size;",    // NOLINT
"const int_tp b = (index / tile_size) % bottom_tile_axis;",    // NOLINT
"const int_tp n = index / tile_size / bottom_tile_axis;",    // NOLINT
"bottom_diff[index] = 0;",    // NOLINT
"int_tp top_index = (n * num_tiles * bottom_tile_axis + b) * tile_size + d;",    // NOLINT
"for (int_tp t = 0; t < num_tiles; ++t) {",    // NOLINT
"bottom_diff[index] += top_diff[top_index];",    // NOLINT
"top_index += bottom_tile_axis * tile_size;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
""}   // NOLINT
};
static std::string cl_kernel_names[] = {
    "activation",   // NOLINT
    "auxiliary",   // NOLINT
    "batch_norm",   // NOLINT
    "batch_reindex",   // NOLINT
    "benchmark",   // NOLINT
    "bias",   // NOLINT
    "bnll",   // NOLINT
    "channel",   // NOLINT
    "concat",   // NOLINT
    "contrastive_loss",   // NOLINT
    "conv_layer_spatial",   // NOLINT
    "conv_spatial_helper",   // NOLINT
    "crop",   // NOLINT
    "dropout",   // NOLINT
    "eltwise",   // NOLINT
    "elu",   // NOLINT
    "embed",   // NOLINT
    "fft",   // NOLINT
    "fillbuffer",   // NOLINT
    "im2col",   // NOLINT
    "im2col_nd",   // NOLINT
    "lrn",   // NOLINT
    "lstm_unit",   // NOLINT
    "math",   // NOLINT
    "mergecrop",   // NOLINT
    "pooling",   // NOLINT
    "pooling_nd",   // NOLINT
    "pooling_sk",   // NOLINT
    "slice",   // NOLINT
    "softmax_loss",   // NOLINT
    "solvers",   // NOLINT
    "tile"   // NOLINT
};
viennacl::ocl::program & RegisterKernels(viennacl::ocl::context *ctx) {
  std::stringstream ss;
#ifdef USE_INDEX_64
  ss << header << "\n\n";  // NOLINT
  ss << definitions_64 << "\n\n";  // NOLINT
#else
  ss << header << "\n\n";  // NOLINT
  ss << definitions_32 << "\n\n";  // NOLINT
#endif
  ss << "#define Dtype float" << "\n\n";  // NOLINT
  ss << "#define Dtype2 float2" << "\n\n";  // NOLINT
  ss << "#define Dtype4 float4" << "\n\n";  // NOLINT
  ss << "#define Dtype8 float8" << "\n\n";  // NOLINT
  ss << "#define Dtype16 float16" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_FLOAT" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    for (int j = 0; j < cl_kernels[i].size(); ++j) {
      ss << cl_kernels[i][j] << "\n\n";
    }
  }
  ss << "#ifdef DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  ss << "#undef Dtype" << "\n\n";  // NOLINT
  ss << "#undef Dtype2" << "\n\n";  // NOLINT
  ss << "#undef Dtype4" << "\n\n";  // NOLINT
  ss << "#undef Dtype8" << "\n\n";  // NOLINT
  ss << "#undef Dtype16" << "\n\n";  // NOLINT
  ss << "#define Dtype double" << "\n\n";  // NOLINT
  ss << "#define Dtype2 double2" << "\n\n";  // NOLINT
  ss << "#define Dtype4 double4" << "\n\n";  // NOLINT
  ss << "#define Dtype8 double8" << "\n\n";  // NOLINT
  ss << "#define Dtype16 double16" << "\n\n";  // NOLINT
  ss << "#undef TYPE" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_DOUBLE" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] != std::string("fft")) {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  ss << "#endif  // DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  std::string kernel_string = ss.str();
  const char* kernel_program = kernel_string.c_str();
  // ctx->build_options("-cl-fast-relaxed-math -cl-mad-enable");
#ifdef USE_FFT
  ctx->build_options("-DFFT");
#endif
  viennacl::ocl::program &program = ctx->add_program(kernel_program,
      "kernel_program");
  return program;
}
viennacl::ocl::program & submit_conv_spatial_program(
viennacl::ocl::context *ctx, string name, string options) {
  static const char* core_defines =
  "#define Dtype float\n"
  "#define Dtype2 float2\n"
  "#define Dtype4 float4\n"
  "#define Dtype8 float8\n"
  "#define Dtype16 float16\n"
  "#define OCL_KERNEL_LOOP(i, n)"
  " for (int i = get_global_id(0); i < (n); i += get_global_size(0))\n";
  std::stringstream ss;
  ss << core_defines;
#ifdef USE_INDEX_64
  ss << header + "\n";
  ss << definitions_64 + "\n";
#else
  ss << header + "\n";
  ss << definitions_32 + "\n";
#endif
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] == "conv_layer_spatial") {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  ctx->build_options(options);
  viennacl::ocl::program &program = ctx->add_program(ss.str(), name);
  return program;
}
int getKernelBundleCount() {
  return cl_kernels.size();
}
template<typename Dtype>
std::string getKernelBundleSource(int index) {
  std::stringstream ss;
#ifdef USE_INDEX_64
  ss << header << "\n\n";  // NOLINT
  ss << definitions_64 << "\n\n";  // NOLINT
#else
  ss << header << "\n\n";  // NOLINT
  ss << definitions_32 << "\n\n";  // NOLINT
#endif
  if (std::is_same<Dtype, float>::value) {
    ss << "#define Dtype float" << "\n\n";  // NOLINT
    ss << "#define Dtype2 float2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 float4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 float8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 float16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_FLOAT" << "\n\n";  // NOLINT
  } else {
    ss << "#ifdef DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
    ss << "#define Dtype double" << "\n\n";  // NOLINT
    ss << "#define Dtype2 double2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 double4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 double8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 double16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_DOUBLE" << "\n\n";  // NOLINT
  }
  for (int j = 0; j < cl_kernels[index].size(); ++j) {
    ss << cl_kernels[index][j] << "\n\n";
  }
  if (std::is_same<Dtype, float>::value) {
  } else {
    ss << "#endif" << "\n\n";  // NOLINT
  }
  return ss.str();
}
template std::string getKernelBundleSource<float>(int index);
template std::string getKernelBundleSource<double>(int index);
std::string getKernelBundleName(int index) {
  return cl_kernel_names[index];
}
}  // namespace caffe
#endif
