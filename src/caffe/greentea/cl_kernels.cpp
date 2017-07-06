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
static std::string header = DOUBLE_SUPPORT "#ifndef __OPENCL_VERSION__\n#define __kernel\n#define __global\n#define __constant\n#define __local\n#define get_global_id(x) 0\n#define get_global_size(x) 0\n#define get_local_id(x) 0\n#define get_local_size(x) 0\n#define FLT_MAX 0\n#define FLT_MIN 0\n#define cl_khr_fp64\n#define cl_amd_fp64\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#define CLK_LOCAL_MEM_FENCE\n#define CLK_GLOBAL_MEM_FENCE\n#define Dtype float\n#define barrier(x)\n#define atomic_cmpxchg(x, y, z) x\n#define signbit(x) x\n#define int_tp long\n#define uint_tp unsigned long\n#define int_tpc long\n#define uint_tpc unsigned long\n#endif\n\n#define CONCAT(A,B) A##_##B\n#define TEMPLATE(name,type) CONCAT(name,type)\n\n#define TYPE_FLOAT 1\n#define TYPE_DOUBLE 2\n#define TYPE_HALF 3\n\n#if defined(cl_khr_fp64)\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#elif defined(cl_amd_fp64)\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#endif\n\n#if defined(cl_khr_fp16)\n#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define HALF_SUPPORT_AVAILABLE\n#endif\n\n#if defined(cl_khr_int64_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n#define ATOMICS_64_AVAILABLE\n#endif\n\n#if defined(cl_khr_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if defined(cl_khr_global_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif";  // NOLINT
static std::string definitions_64 = DOUBLE_SUPPORT "// Types used for parameters, offset computations and so on\n#define int_tp long\n#define uint_tp unsigned long\n\n// Definitions used to cast the types above as needed\n#define int_tpc long\n#define uint_tpc unsigned long";  // NOLINT
#else
static std::string header = DOUBLE_SUPPORT "#ifndef __OPENCL_VERSION__\n#define __kernel\n#define __global\n#define __constant\n#define __local\n#define get_global_id(x) 0\n#define get_global_size(x) 0\n#define get_local_id(x) 0\n#define get_local_size(x) 0\n#define FLT_MAX 0\n#define FLT_MIN 0\n#define cl_khr_fp64\n#define cl_amd_fp64\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#define CLK_LOCAL_MEM_FENCE\n#define CLK_GLOBAL_MEM_FENCE\n#define Dtype float\n#define barrier(x)\n#define atomic_cmpxchg(x, y, z) x\n#define signbit(x) x\n#define int_tp long\n#define uint_tp unsigned long\n#define int_tpc long\n#define uint_tpc unsigned long\n#endif\n\n#define CONCAT(A,B) A##_##B\n#define TEMPLATE(name,type) CONCAT(name,type)\n\n#define TYPE_FLOAT 1\n#define TYPE_DOUBLE 2\n#define TYPE_HALF 3\n\n#if defined(cl_khr_fp64)\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#elif defined(cl_amd_fp64)\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#endif\n\n#if defined(cl_khr_fp16)\n#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define HALF_SUPPORT_AVAILABLE\n#endif\n\n#if defined(cl_khr_int64_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n#define ATOMICS_64_AVAILABLE\n#endif\n\n#if defined(cl_khr_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if defined(cl_khr_global_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif";  // NOLINT
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
"KERNEL_ARG_DTYPE negative_slope) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(relu_backward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff,",    // NOLINT
"KERNEL_ARG_DTYPE negative_slope) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out_diff[index] = in_diff[index]",    // NOLINT
"* ((Dtype)(in_data[index] > 0?1.0:0.0) + (Dtype)(in_data[index] <= 0?1.0:0.0) * negative_slope);",    // NOLINT
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
"out_diff[index] = in_diff[index] * ((Dtype)1 - tanhx * tanhx);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sigmoid_forward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = (Dtype)1.0 / ((Dtype)1.0 + exp(-in[index]));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sigmoid_backward,Dtype)(const int_tp n,",    // NOLINT
"__global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* out_data,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"const Dtype sigmoid_x = out_data[index];",    // NOLINT
"out_diff[index] = in_diff[index] * sigmoid_x * ((Dtype)1 - sigmoid_x);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(threshold,Dtype)(const int_tp n, const KERNEL_ARG_DTYPE threshold,",    // NOLINT
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
"out_diff[index] = in_diff[index] * in_data[index] * (Dtype)(in_data[index] <= 0?1.0:0.0);",    // NOLINT
"for (int k = 1; k < rows; k++) {",    // NOLINT
"out_diff[index] += in_diff[index + k * rowPitch]",    // NOLINT
"* in_data[index + k * rowPitch]",    // NOLINT
"* (Dtype)(in_data[index + k * rowPitch] <= 0?1.0:0.0);",    // NOLINT
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
"__kernel void TEMPLATE(gpu_set,Dtype)(const int_tp n, const KERNEL_ARG_DTYPE alpha, __global Dtype* y) {",    // NOLINT
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
"Dtype TEMPLATE(bn_common,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const Dtype scale, const Dtype eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global const Dtype* data,",    // NOLINT
"int_tp *out_off) {",    // NOLINT
"const int_tp idx_num = get_global_id(0);",    // NOLINT
"const int_tp idx_chans = get_global_id(1);",    // NOLINT
"const int_tp idx_spatial_dim = get_global_id(2);",    // NOLINT
"",    // NOLINT
"Dtype m = mean[idx_chans];",    // NOLINT
"Dtype v = variance[idx_chans];",    // NOLINT
"",    // NOLINT
"m = -scale * m;",    // NOLINT
"v = (Dtype)native_powr((Dtype)mad(scale, v, eps), (Dtype)-0.5);",    // NOLINT
"",    // NOLINT
"*out_off = (idx_num * channels + idx_chans) * spatial_dim + idx_spatial_dim;",    // NOLINT
"return (v * (data[*out_off] + m));",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bn_use_global_stats_in_place,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global Dtype* top) {",    // NOLINT
"int_tp out_off;",    // NOLINT
"Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, top, &out_off);",    // NOLINT
"top[out_off] = val;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bn_use_global_stats_in_place_fused_relu,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global Dtype* top) {",    // NOLINT
"int_tp out_off;",    // NOLINT
"Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, top, &out_off);",    // NOLINT
"top[out_off] = val > 0.0f ? val : 0.0f;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bn_use_global_stats,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global const Dtype* bottom,",    // NOLINT
"__global Dtype* top) {",    // NOLINT
"int_tp out_off;",    // NOLINT
"Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, bottom, &out_off);",    // NOLINT
"top[out_off] = val;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(bn_use_global_stats_fused_relu,Dtype)(const int_tp num, const int_tp channels, const int_tp spatial_dim,",    // NOLINT
"const KERNEL_ARG_DTYPE scale, const KERNEL_ARG_DTYPE eps,",    // NOLINT
"__global const Dtype* mean,",    // NOLINT
"__global const Dtype* variance,",    // NOLINT
"__global const Dtype* bottom,",    // NOLINT
"__global Dtype* top) {",    // NOLINT
"int_tp out_off;",    // NOLINT
"Dtype val = TEMPLATE(bn_common,Dtype)(num, channels, spatial_dim, scale, eps, mean, variance, bottom, &out_off);",    // NOLINT
"top[out_off] =  val > 0.0f ? val : 0.0f;",    // NOLINT
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
"__kernel void TEMPLATE(DecodeBBoxesCORNER, Dtype)(const int nthreads,",    // NOLINT
"__global const Dtype* loc_data, __global const Dtype* prior_data,",    // NOLINT
"const int variance_encoded_in_target,",    // NOLINT
"const int num_priors, const int share_location,",    // NOLINT
"const int num_loc_classes, const int background_label_id,",    // NOLINT
"const int clip_bbox, __global Dtype* bbox_data) {",    // NOLINT
"",    // NOLINT
"for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {",    // NOLINT
"const int i = index % 4;",    // NOLINT
"const int c = (index / 4) % num_loc_classes;",    // NOLINT
"const int d = (index / 4 / num_loc_classes) % num_priors;",    // NOLINT
"if (!share_location && c == background_label_id) {",    // NOLINT
"// Ignore background class if not share_location.",    // NOLINT
"return;",    // NOLINT
"}",    // NOLINT
"const int pi = d * 4;",    // NOLINT
"const int vi = pi + num_priors * 4;",    // NOLINT
"if (variance_encoded_in_target) {",    // NOLINT
"// variance is encoded in target, we simply need to add the offset",    // NOLINT
"// predictions.",    // NOLINT
"bbox_data[index] = prior_data[pi + i] + loc_data[index];",    // NOLINT
"} else {",    // NOLINT
"// variance is encoded in bbox, we need to scale the offset accordingly.",    // NOLINT
"bbox_data[index] =",    // NOLINT
"prior_data[pi + i] + loc_data[index] * prior_data[vi + i];",    // NOLINT
"}",    // NOLINT
"if (clip_bbox) {",    // NOLINT
"bbox_data[index] = max(min(bbox_data[index], (Dtype)1.), (Dtype)0.);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(DecodeBBoxesCENTER_SIZE, Dtype)(const int nthreads,",    // NOLINT
"__global const Dtype* loc_data, __global const Dtype* prior_data,",    // NOLINT
"const int variance_encoded_in_target,",    // NOLINT
"const int num_priors, const int share_location,",    // NOLINT
"const int num_loc_classes, const int background_label_id,",    // NOLINT
"const int clip_bbox, __global Dtype* bbox_data) {",    // NOLINT
"",    // NOLINT
"for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {",    // NOLINT
"const int i = index % 4;",    // NOLINT
"const int c = (index / 4) % num_loc_classes;",    // NOLINT
"const int d = (index / 4 / num_loc_classes) % num_priors;",    // NOLINT
"if (!share_location && c == background_label_id) {",    // NOLINT
"// Ignore background class if not share_location.",    // NOLINT
"return;",    // NOLINT
"}",    // NOLINT
"const int pi = d * 4;",    // NOLINT
"const int vi = pi + num_priors * 4;",    // NOLINT
"const Dtype p_xmin = prior_data[pi];",    // NOLINT
"const Dtype p_ymin = prior_data[pi + 1];",    // NOLINT
"const Dtype p_xmax = prior_data[pi + 2];",    // NOLINT
"const Dtype p_ymax = prior_data[pi + 3];",    // NOLINT
"const Dtype prior_width = p_xmax - p_xmin;",    // NOLINT
"const Dtype prior_height = p_ymax - p_ymin;",    // NOLINT
"const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;",    // NOLINT
"const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;",    // NOLINT
"",    // NOLINT
"const Dtype xmin = loc_data[index - i];",    // NOLINT
"const Dtype ymin = loc_data[index - i + 1];",    // NOLINT
"const Dtype xmax = loc_data[index - i + 2];",    // NOLINT
"const Dtype ymax = loc_data[index - i + 3];",    // NOLINT
"",    // NOLINT
"Dtype decode_bbox_center_x, decode_bbox_center_y;",    // NOLINT
"Dtype decode_bbox_width, decode_bbox_height;",    // NOLINT
"if (variance_encoded_in_target) {",    // NOLINT
"// variance is encoded in target, we simply need to retore the offset",    // NOLINT
"// predictions.",    // NOLINT
"decode_bbox_center_x = xmin * prior_width + prior_center_x;",    // NOLINT
"decode_bbox_center_y = ymin * prior_height + prior_center_y;",    // NOLINT
"decode_bbox_width = exp(xmax) * prior_width;",    // NOLINT
"decode_bbox_height = exp(ymax) * prior_height;",    // NOLINT
"} else {",    // NOLINT
"// variance is encoded in bbox, we need to scale the offset accordingly.",    // NOLINT
"decode_bbox_center_x =",    // NOLINT
"prior_data[vi] * xmin * prior_width + prior_center_x;",    // NOLINT
"decode_bbox_center_y =",    // NOLINT
"prior_data[vi + 1] * ymin * prior_height + prior_center_y;",    // NOLINT
"decode_bbox_width =",    // NOLINT
"exp(prior_data[vi + 2] * xmax) * prior_width;",    // NOLINT
"decode_bbox_height =",    // NOLINT
"exp(prior_data[vi + 3] * ymax) * prior_height;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"switch (i) {",    // NOLINT
"case 0:",    // NOLINT
"bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;",    // NOLINT
"break;",    // NOLINT
"case 1:",    // NOLINT
"bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;",    // NOLINT
"break;",    // NOLINT
"case 2:",    // NOLINT
"bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;",    // NOLINT
"break;",    // NOLINT
"case 3:",    // NOLINT
"bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;",    // NOLINT
"break;",    // NOLINT
"}",    // NOLINT
"if (clip_bbox) {",    // NOLINT
"bbox_data[index] = max(min(bbox_data[index], (Dtype)1.), (Dtype)0.);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(DecodeBBoxesCORNER_SIZE, Dtype)(const int nthreads,",    // NOLINT
"__global const Dtype* loc_data, __global const Dtype* prior_data,",    // NOLINT
"const int variance_encoded_in_target,",    // NOLINT
"const int num_priors, const int share_location,",    // NOLINT
"const int num_loc_classes, const int background_label_id,",    // NOLINT
"const int clip_bbox, __global Dtype* bbox_data) {",    // NOLINT
"",    // NOLINT
"for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {",    // NOLINT
"const int i = index % 4;",    // NOLINT
"const int c = (index / 4) % num_loc_classes;",    // NOLINT
"const int d = (index / 4 / num_loc_classes) % num_priors;",    // NOLINT
"if (!share_location && c == background_label_id) {",    // NOLINT
"// Ignore background class if not share_location.",    // NOLINT
"return;",    // NOLINT
"}",    // NOLINT
"const int pi = d * 4;",    // NOLINT
"const int vi = pi + num_priors * 4;",    // NOLINT
"const Dtype p_xmin = prior_data[pi];",    // NOLINT
"const Dtype p_ymin = prior_data[pi + 1];",    // NOLINT
"const Dtype p_xmax = prior_data[pi + 2];",    // NOLINT
"const Dtype p_ymax = prior_data[pi + 3];",    // NOLINT
"const Dtype prior_width = p_xmax - p_xmin;",    // NOLINT
"const Dtype prior_height = p_ymax - p_ymin;",    // NOLINT
"Dtype p_size;",    // NOLINT
"if (i == 0 || i == 2) {",    // NOLINT
"p_size = prior_width;",    // NOLINT
"} else {",    // NOLINT
"p_size = prior_height;",    // NOLINT
"}",    // NOLINT
"if (variance_encoded_in_target) {",    // NOLINT
"// variance is encoded in target, we simply need to add the offset",    // NOLINT
"// predictions.",    // NOLINT
"bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;",    // NOLINT
"} else {",    // NOLINT
"// variance is encoded in bbox, we need to scale the offset accordingly.",    // NOLINT
"bbox_data[index] =",    // NOLINT
"prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;",    // NOLINT
"}",    // NOLINT
"if (clip_bbox) {",    // NOLINT
"bbox_data[index] = max(min(bbox_data[index], (Dtype)1.), (Dtype)0.);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(PermuteData, Dtype)(const int nthreads,",    // NOLINT
"__global const Dtype* data, const int num_classes, const int num_data,",    // NOLINT
"const int num_dim, __global Dtype* new_data) {",    // NOLINT
"for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {",    // NOLINT
"const int i = index % num_dim;",    // NOLINT
"const int c = (index / num_dim) % num_classes;",    // NOLINT
"const int d = (index / num_dim / num_classes) % num_data;",    // NOLINT
"const int n = index / num_dim / num_classes / num_data;",    // NOLINT
"const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;",    // NOLINT
"new_data[new_index] = data[index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(null_kernel,Dtype)(KERNEL_ARG_DTYPE arg) {",    // NOLINT
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
"out[index] = in[index] + log((Dtype) ((Dtype)1.0 + exp(-in[index])));",    // NOLINT
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
"out_diff[index] = in_diff[index] * expval / (expval + (Dtype)1.);",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"const KERNEL_ARG_DTYPE margin, const KERNEL_ARG_DTYPE alpha, __global const Dtype* y,",    // NOLINT
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
"beta = -alpha * mdist / (dist + (Dtype)1e-4) * diff[i];",    // NOLINT
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
"const KERNEL_ARG_DTYPE margin, const KERNEL_ARG_DTYPE alpha, __global Dtype* y,",    // NOLINT
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
"__kernel void TEMPLATE(conv_layer_spatial_phony,Dtype)(KERNEL_ARG_DTYPE arg) {",    // NOLINT
"Dtype out = arg;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#ifdef FUSED_CONV_RELU",    // NOLINT
"#define ACTIVATION_RELU_FUNCTION(x) ((Dtype)(x) > 0 ? (Dtype)(x) : ((Dtype)(x) * (Dtype)(negative_slope)))",    // NOLINT
"#define NEGATIVE_SLOPE_ARG KERNEL_ARG_DTYPE negative_slope,",    // NOLINT
"#else",    // NOLINT
"#define ACTIVATION_RELU_FUNCTION(x) (x)",    // NOLINT
"#define NEGATIVE_SLOPE_ARG",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#ifdef FUSED_CONV_ELTWISE",    // NOLINT
"#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) do { (_dst_)[(_offset_)] = ACTIVATION_RELU_FUNCTION(eltwise_data[(_offset_)] + (_data_));} while(0)",    // NOLINT
"#define ELTWISE_DATA_ARG __global Dtype* eltwise_data,",    // NOLINT
"#else",    // NOLINT
"#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) do { (_dst_)[(_offset_)] = ACTIVATION_RELU_FUNCTION(_data_);} while(0)",    // NOLINT
"#define ELTWISE_DATA_ARG",    // NOLINT
"#endif",    // NOLINT
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
"ELTWISE_DATA_ARG",    // NOLINT
"NEGATIVE_SLOPE_ARG",    // NOLINT
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
"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if defined(convolve_simd) || defined(Conv_Interleaved)",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define INT_TYPE ushort",    // NOLINT
"#define INT_TYPE2 ushort2",    // NOLINT
"#define INT_TYPE4 ushort4",    // NOLINT
"#define INT_TYPE8 ushort8",    // NOLINT
"#define SUB_GROUP_BLOCK_READ2 intel_sub_group_block_read_us2",    // NOLINT
"#define SUB_GROUP_BLOCK_READ4 intel_sub_group_block_read_us4",    // NOLINT
"#define SUB_GROUP_BLOCK_READ8 intel_sub_group_block_read_us8",    // NOLINT
"#define SUB_GROUP_BLOCK_READ intel_sub_group_block_read_us",    // NOLINT
"#else",    // NOLINT
"#define INT_TYPE uint",    // NOLINT
"#define INT_TYPE2 uint2",    // NOLINT
"#define INT_TYPE4 uint4",    // NOLINT
"#define INT_TYPE8 uint8",    // NOLINT
"#define SUB_GROUP_BLOCK_READ2 intel_sub_group_block_read2",    // NOLINT
"#define SUB_GROUP_BLOCK_READ4 intel_sub_group_block_read4",    // NOLINT
"#define SUB_GROUP_BLOCK_READ8 intel_sub_group_block_read8",    // NOLINT
"#define SUB_GROUP_BLOCK_READ intel_sub_group_block_read",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"//Begin IDLF kernels below here",    // NOLINT
"#ifdef IDLF",    // NOLINT
"",    // NOLINT
"#define OUT_BLOCK_SIZE (OUT_BLOCK_WIDTH*OUT_BLOCK_HEIGHT)",    // NOLINT
"",    // NOLINT
"// Each work-item computes a OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT region of one output map.",    // NOLINT
"// Each work-group (which will be mapped to 1 SIMD16/SIMD8 EU thread) will compute 16/8 different feature maps, but each feature map is for the same region of the imput image.",    // NOLINT
"// NDRange:  (output_width+pad)/ OUT_BLOCK_WIDTH, (output_height+pad)/OUT_BLOCK_HEIGHT, NUM_FILTERS/OUT_BLOCK_DEPTH",    // NOLINT
"",    // NOLINT
"// NOTE: for beignet this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.",    // NOLINT
"#ifndef __BEIGNET__",    // NOLINT
"__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))",    // NOLINT
"#endif",    // NOLINT
"__kernel void",    // NOLINT
"convolve_simd(",    // NOLINT
"ELTWISE_DATA_ARG",    // NOLINT
"NEGATIVE_SLOPE_ARG",    // NOLINT
"__global Dtype* inputs_base,",    // NOLINT
"filter_qualifier Dtype* weights_base,",    // NOLINT
"__global Dtype* biases_base,",    // NOLINT
"__global Dtype* outputs_base,",    // NOLINT
"const ushort input_width,",    // NOLINT
"const ushort input_height,",    // NOLINT
"const ushort output_width,",    // NOLINT
"const ushort output_height)",    // NOLINT
"{",    // NOLINT
"__global Dtype* outputs = outputs_base;",    // NOLINT
"__global Dtype* inputs = inputs_base;",    // NOLINT
"filter_qualifier Dtype* weights = weights_base;",    // NOLINT
"__global Dtype* biases = biases_base;",    // NOLINT
"",    // NOLINT
"uint_tp oc = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column",    // NOLINT
"uint_tp or = get_global_id(1) * OUT_BLOCK_HEIGHT;// or = Output Row",    // NOLINT
"uint_tp fm = get_global_id(2);// fm = Feature Map = od = Output Depth",    // NOLINT
"uint_tp fmg = get_group_id(2);",    // NOLINT
"uint_tp lid = get_local_id(2);",    // NOLINT
"",    // NOLINT
"Dtype out[OUT_BLOCK_SIZE];",    // NOLINT
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
"int curr_local_y = ( lid / ( TILE_X / 4 ) );",    // NOLINT
"int curr_local_x = ( lid % ( TILE_X / 4 ) ) * 4;",    // NOLINT
"int curr_y = or * STRIDEY + INPUT_START_Y + curr_local_y;",    // NOLINT
"int curr_x = oc * STRIDEX + INPUT_START_X + curr_local_x;",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0",    // NOLINT
"int saved_y = curr_y;",    // NOLINT
"#endif",    // NOLINT
"in_addr = input_batch_offset + INPUT_START_Z * input_height * input_width",    // NOLINT
"+  (curr_y - INPUT_PAD_H) * input_width             // y tile offset",    // NOLINT
"+   curr_x - INPUT_PAD_W;                        // x tile offset",    // NOLINT
"union {",    // NOLINT
"Dtype4 in_vec[INVEC_SIZE];",    // NOLINT
"Dtype in_array[INVEC_SIZE * 4];",    // NOLINT
"} in_buf;",    // NOLINT
"",    // NOLINT
"for(int_tp kd = 0; kd < INPUT_DEPTH; kd++)",    // NOLINT
"{",    // NOLINT
"int_tp in_offset = in_addr;",    // NOLINT
"int_tp reg = 0;",    // NOLINT
"LOOP(INVEC_SIZE, reg,",    // NOLINT
"{",    // NOLINT
"if (curr_local_y + reg * TILE_Y_STRIDE < TILE_Y || INVEC_SIZE * TILE_Y_STRIDE == TILE_Y || reg < INVEC_SIZE - 1) {",    // NOLINT
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
"in_buf.in_vec[reg] = vload4(0, (inputs + in_offset)); // read 16 elements",    // NOLINT
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
"in_buf.in_vec[reg] = vload4(0, (inputs + in_offset)); // read 16 elements",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
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
"Dtype w[WEIGHT_PREF];",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT != 1",    // NOLINT
"INT_TYPE8 ui8;",    // NOLINT
"#endif",    // NOLINT
"} weight_buf;",    // NOLINT
"int_tp w_idx=0;",    // NOLINT
"",    // NOLINT
"uint_tp orig_weight_addr = weight_addr;",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT != 1",    // NOLINT
"weight_buf.ui8 = SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);",    // NOLINT
"weight_addr += SIMD_SIZE * WEIGHT_PREF;",    // NOLINT
"#else",    // NOLINT
"weight_buf.w[0] = as_Dtype(SUB_GROUP_BLOCK_READ((__global INT_TYPE *)&weights[weight_addr]));",    // NOLINT
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
"Dtype input = BLOCK_IN((br * STRIDEY + kr * DILATION_Y) * TILE_X + bc * STRIDEX + kc * DILATION_X);",    // NOLINT
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
"weight_buf.ui8 = SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);",    // NOLINT
"weight_addr += SIMD_SIZE * WEIGHT_PREF;  // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.",    // NOLINT
"}",    // NOLINT
"#if KERNEL_WIDTH*KERNEL_HEIGHT % 8 == 0",    // NOLINT
"// need to do nothing",    // NOLINT
"#else",    // NOLINT
"else if ((w_idx + 1) %  WEIGHT_PREF == 0 && ((w_idx + 1) > (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF)))",    // NOLINT
"#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 1",    // NOLINT
"weight_buf.w[0] = weights[weight_addr];",    // NOLINT
"#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 2",    // NOLINT
"weight_buf.ui8.s01 = SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)&weights[weight_addr]);",    // NOLINT
"#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 <= 4",    // NOLINT
"weight_buf.ui8.s0123 = SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)&weights[weight_addr]);",    // NOLINT
"#else",    // NOLINT
"weight_buf.ui8 = SUB_GROUP_BLOCK_READ8((__global INT_TYPE *)&weights[weight_addr]);",    // NOLINT
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
"fm = fm % ALIGNED_NUM_FILTERS;",    // NOLINT
"",    // NOLINT
"if ((ALIGNED_NUM_FILTERS == NUM_FILTERS || fm < NUM_FILTERS)) {",    // NOLINT
"uint_tp out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH + fm ) * output_width * output_height;",    // NOLINT
"out_addr += or * output_width + oc;",    // NOLINT
"// we need this address calculation for biases because we support views and batching",    // NOLINT
"Dtype bias = biases[fm];",    // NOLINT
"for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {",    // NOLINT
"if (r + or >= output_height) break;",    // NOLINT
"for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {",    // NOLINT
"if (c + oc >= output_width) break;",    // NOLINT
"// this does a scattered write to SIMD_SIZE different feature maps, so that data within one map is contiguous, thus ready for input to next layer.",    // NOLINT
"ACTIVATION_FUNCTION(outputs, out_addr + r * output_width + c, bias + out[r * OUT_BLOCK_WIDTH + c]);",    // NOLINT
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
"typedef struct half1 { half s0; } half1;",    // NOLINT
"typedef struct half5 { half s0; half s1; half s2; half s3; half s4; } half5;",    // NOLINT
"typedef struct half6 { half s0; half s1; half s2; half s3; half s4; half s5; } half6;",    // NOLINT
"typedef struct half7 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; } half7;",    // NOLINT
"typedef struct half9 { half s0; half s1; half s2; half s3; half s4; half s5; half s6; half s7; half s8; } half9;",    // NOLINT
"typedef struct half10 { half s0; half s1; half s2; half s3; half s4; half s5;",    // NOLINT
"half s6; half s7; half s8; half s9; } half10;",    // NOLINT
"typedef struct half11 { half s0; half s1; half s2; half s3; half s4; half s5;",    // NOLINT
"half s6; half s7; half s8; half s9; half sa; } half11;",    // NOLINT
"typedef struct half12 { half s0; half s1; half s2; half s3; half s4; half s5;",    // NOLINT
"half s6; half s7; half s8; half s9; half sa; half sb; } half12;",    // NOLINT
"typedef struct half13 { half s0; half s1; half s2; half s3; half s4; half s5;",    // NOLINT
"half s6; half s7; half s8; half s9; half sa; half sb; half sc; } half13;",    // NOLINT
"typedef struct half14 { half s0; half s1; half s2; half s3; half s4; half s5;",    // NOLINT
"half s6; half s7; half s8; half s9; half sa; half sb; half sc; half sd; } half14;",    // NOLINT
"typedef struct half15 { half s0; half s1; half s2; half s3; half s4; half s5;",    // NOLINT
"half s6; half s7; half s8; half s9; half sa; half sb; half sc; half sd; half se; } half15;",    // NOLINT
"typedef struct half0 { half s0; } half0; //never used but makes compiler happy.",    // NOLINT
"",    // NOLINT
"#define OUT_PITCH_X output_width",    // NOLINT
"#define ROW_PITCH input_width",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"#define GEMM_LIKE_KERNEL_ARGS         ELTWISE_DATA_ARG                  NEGATIVE_SLOPE_ARG                const __global Dtype *src0,       const __global Dtype *src1,       const __global Dtype *biases,     __global Dtype *dst,              const ushort input_width,         const ushort input_height,        const ushort output_width,        const ushort output_height,       const int_tp out_pitch_y,         const int_tp out_pitch_z,         const int_tp aligned_input_size,     const int_tp slice_pitch",    // NOLINT
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
"#ifndef __BEIGNET__",    // NOLINT
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
"typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;",    // NOLINT
"",    // NOLINT
"// True for all threads if filter_width is multiple of TILE_N",    // NOLINT
"// else, true for all but right-most column of threads.",    // NOLINT
"if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N )",    // NOLINT
"{",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"Dtype8  blockC00 = 0.f;",    // NOLINT
"Dtype8  blockC10 = 0.f;",    // NOLINT
"Dtype8  blockC20 = 0.f;",    // NOLINT
"Dtype8  blockC30 = 0.f;",    // NOLINT
"",    // NOLINT
"// Src0 (patch input) is directly used as atile.",    // NOLINT
"// Each work item points to the start of a different patch.",    // NOLINT
"// atile is M rows x K columns.",    // NOLINT
"int curr_x = ( global_y % output_width ) * STRIDE_X;",    // NOLINT
"int curr_y = ( global_y / output_width ) * STRIDE_Y;",    // NOLINT
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"int saved_y = curr_y;",    // NOLINT
"#endif",    // NOLINT
"const __global Dtype *src0_read = src0",    // NOLINT
"+ aligned_input_size * global_z                            // batch offset",    // NOLINT
"+ (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset",    // NOLINT
"+ (curr_x - INPUT_PAD_W);                 // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global Dtype *src1_read = src1 + ( global_x * TILE_N  * 2);",    // NOLINT
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
"// Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype8 granularity.",    // NOLINT
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
"src0_read += (ROW_PITCH * DILATION_Y);",    // NOLINT
"",    // NOLINT
"Dtype blockB00[KERNEL_WIDTH*4];",    // NOLINT
"Dtype8* p8BlockB00 = (Dtype8*)blockB00;",    // NOLINT
"Dtype4* p4BlockB00 = (Dtype4*)blockB00;",    // NOLINT
"Dtype*  pBlockB00 =  (Dtype* )blockB00;",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p8BlockB00[interleaved_y] = as_Dtype8( SUB_GROUP_BLOCK_READ8( (const __global INT_TYPE *)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p4BlockB00[KERNEL_WIDTH - 1] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE *)src1_read ) );",    // NOLINT
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
"int_tp out_offset = global_z * out_pitch_z                                                   // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                       // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset",    // NOLINT
"+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"__global Dtype *out = dst + out_offset;",    // NOLINT
"Dtype bias[4];",    // NOLINT
"Dtype4 *bias_vec;",    // NOLINT
"bias_vec = (Dtype4*)bias;",    // NOLINT
"*bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"if (global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < 8; i++)",    // NOLINT
"{",    // NOLINT
"ACTIVATION_FUNCTION(dst, out_offset + ( 0 + i ) * out_pitch_y, blockC00[i] + intel_sub_group_shuffle(bias[0], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out_offset + ( 8 + i ) * out_pitch_y, blockC10[i] + intel_sub_group_shuffle(bias[1], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out_offset + ( 16 + i ) * out_pitch_y, blockC20[i] + intel_sub_group_shuffle(bias[2], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out_offset + ( 24 + i ) * out_pitch_y, blockC30[i] + intel_sub_group_shuffle(bias[3], i));",    // NOLINT
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
"Dtype8  blockC[TILE_N_LAST_DIV8];",    // NOLINT
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
"const __global Dtype *src0_read = src0",    // NOLINT
"+ aligned_input_size * global_z                            // batch offset",    // NOLINT
"+ (curr_y - INPUT_PAD_H) * ROW_PITCH      // y offset",    // NOLINT
"+ (curr_x - INPUT_PAD_W);                 // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global Dtype *src1_read = src1 + ( global_x * TILE_N  * 2);",    // NOLINT
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
"src0_read += (ROW_PITCH * DILATION_Y);",    // NOLINT
"Dtype blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"Dtype2* p2BlockB = (Dtype2* )blockB;",    // NOLINT
"p2BlockB[interleaved_y] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"Dtype4* p4BlockB = (Dtype4* )blockB;",    // NOLINT
"p4BlockB[interleaved_y] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"//TODO: broken.  No block_read6",    // NOLINT
"Dtype6* p6BlockB = (Dtype6* )blockB;",    // NOLINT
"(*((Dtype8*)(&p6BlockB[interleaved_y]))).s0123 = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"(*((Dtype8*)(&p6BlockB[interleaved_y]))).s45 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)(src1_read + 4 * 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"Dtype* pBlockB = (Dtype* )blockB;",    // NOLINT
"pBlockB[KERNEL_WIDTH - 1] = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"Dtype2* p2BlockB = (Dtype2* )blockB;",    // NOLINT
"p2BlockB[KERNEL_WIDTH - 1] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"Dtype3* p3BlockB = (Dtype3* )blockB;",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s01 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s2 = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*) (src1_read + 2 * 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"Dtype* pBlockB = (Dtype*)blockB;",    // NOLINT
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
"int_tp out_offset = global_z * out_pitch_z                                                   // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                       // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset",    // NOLINT
"+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"__global Dtype *out = dst + out_offset;",    // NOLINT
"Dtype bias[4];",    // NOLINT
"Dtype4 *bias_vec;",    // NOLINT
"bias_vec = (Dtype4*)bias;",    // NOLINT
"*bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"if (global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for (int i = 0; i < 8; i++)",    // NOLINT
"{",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 0 ) ACTIVATION_FUNCTION(dst, out_offset + ( 0+i) * out_pitch_y, blockC[0][i] + intel_sub_group_shuffle(bias[0], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 1 ) ACTIVATION_FUNCTION(dst, out_offset + ( 8+i) * out_pitch_y, blockC[1][i] + intel_sub_group_shuffle(bias[1], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 2 ) ACTIVATION_FUNCTION(dst, out_offset + (16+i) * out_pitch_y, blockC[2][i] + intel_sub_group_shuffle(bias[2], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 3 ) ACTIVATION_FUNCTION(dst, out_offset + (24+i) * out_pitch_y, blockC[3][i] + intel_sub_group_shuffle(bias[3], i));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
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
"#ifndef __BEIGNET__",    // NOLINT
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
"typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;",    // NOLINT
"",    // NOLINT
"// True for all threads if filter_width is multiple of TILE_N",    // NOLINT
"// else, true for all but right-most column of threads.",    // NOLINT
"if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N )",    // NOLINT
"{",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"Dtype8  blockC00 = 0.f;",    // NOLINT
"Dtype8  blockC10 = 0.f;",    // NOLINT
"Dtype8  blockC20 = 0.f;",    // NOLINT
"Dtype8  blockC30 = 0.f;",    // NOLINT
"Dtype8  blockC01 = 0.f;",    // NOLINT
"Dtype8  blockC11 = 0.f;",    // NOLINT
"Dtype8  blockC21 = 0.f;",    // NOLINT
"Dtype8  blockC31 = 0.f;",    // NOLINT
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
"const __global Dtype *src0_read0 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x0 - INPUT_PAD_W;                // x offset",    // NOLINT
"const __global Dtype *src0_read1 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x1 - INPUT_PAD_W;                // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global Dtype *src1_read = src1 + ( global_x * TILE_N * 2);",    // NOLINT
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
"// Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype8 granularity.",    // NOLINT
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
"Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;",    // NOLINT
"Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"Dtype*  pblockA01 = (Dtype*)(&blockA01);",    // NOLINT
"#else",    // NOLINT
"Dtype_t blockA00;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read0[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y0 += DILATION_Y;",    // NOLINT
"Dtype_t blockA01;",    // NOLINT
"Dtype*  pblockA01 = (Dtype*)(&blockA01);",    // NOLINT
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
"Dtype blockB00[KERNEL_WIDTH*4];",    // NOLINT
"Dtype8* p8BlockB00 = (Dtype8*)blockB00;",    // NOLINT
"Dtype4* p4BlockB00 = (Dtype4*)blockB00;",    // NOLINT
"Dtype*  pBlockB00 =  (Dtype* )blockB00;",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p8BlockB00[interleaved_y] = as_Dtype8( SUB_GROUP_BLOCK_READ8( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p4BlockB00[KERNEL_WIDTH - 1] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
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
"int_tp out0_offset = global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"int_tp out1_offset = global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"Dtype bias[4];",    // NOLINT
"Dtype4 *bias_vec;",    // NOLINT
"bias_vec = (Dtype4*)bias;",    // NOLINT
"*bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"if( global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"ACTIVATION_FUNCTION(dst, out0_offset + ( 0+i) * out_pitch_y, blockC00[i] + intel_sub_group_shuffle(bias[0], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out0_offset + ( 8+i) * out_pitch_y, blockC10[i] + intel_sub_group_shuffle(bias[1], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out0_offset + (16+i) * out_pitch_y, blockC20[i] + intel_sub_group_shuffle(bias[2], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out0_offset + (24+i) * out_pitch_y, blockC30[i] + intel_sub_group_shuffle(bias[3], i));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if( global_y * TILE_M + 1 < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"ACTIVATION_FUNCTION(dst, out1_offset + ( 0+i) * out_pitch_y, blockC01[i] + intel_sub_group_shuffle(bias[0], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out1_offset + ( 8+i) * out_pitch_y, blockC11[i] + intel_sub_group_shuffle(bias[1], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out1_offset + (16+i) * out_pitch_y, blockC21[i] + intel_sub_group_shuffle(bias[2], i));",    // NOLINT
"ACTIVATION_FUNCTION(dst, out1_offset + (24+i) * out_pitch_y, blockC31[i] + intel_sub_group_shuffle(bias[3], i));",    // NOLINT
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
"Dtype8  blockC0[TILE_N_LAST_DIV8];",    // NOLINT
"Dtype8  blockC1[TILE_N_LAST_DIV8];",    // NOLINT
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
"const __global Dtype *src0_read0 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x0 - INPUT_PAD_W;                // x offset",    // NOLINT
"const __global Dtype *src0_read1 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x1 - INPUT_PAD_W;                // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global Dtype *src1_read = src1 + ( global_x * TILE_N  * 2);",    // NOLINT
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
"Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;",    // NOLINT
"Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"Dtype*  pblockA01 = (Dtype*)(&blockA01);",    // NOLINT
"#else",    // NOLINT
"Dtype_t blockA00;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read0[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y0 += DILATION_Y;",    // NOLINT
"Dtype_t blockA01;",    // NOLINT
"Dtype*  pblockA01 = (Dtype*)(&blockA01);",    // NOLINT
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
"Dtype blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"Dtype2* p2BlockB = (Dtype2* )blockB;",    // NOLINT
"p2BlockB[interleaved_y] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"Dtype4* p4BlockB = (Dtype4* )blockB;",    // NOLINT
"p4BlockB[interleaved_y] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"//TODO: broken.  No block_read6",    // NOLINT
"Dtype6* p6BlockB = (Dtype6* )blockB;",    // NOLINT
"(*((Dtype8*)(&p6BlockB[interleaved_y]))).s0123 = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"(*((Dtype8*)(&p6BlockB[interleaved_y]))).s45 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)(src1_read + 4 * 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"#if TILE_N_LAST_DIV8 == 1",    // NOLINT
"Dtype* pBlockB = (Dtype* )blockB;",    // NOLINT
"pBlockB[KERNEL_WIDTH - 1] = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 2",    // NOLINT
"Dtype2* p2BlockB = (Dtype2* )blockB;",    // NOLINT
"p2BlockB[KERNEL_WIDTH - 1] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"#elif TILE_N_LAST_DIV8 == 3",    // NOLINT
"Dtype3* p3BlockB = (Dtype3* )blockB;",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s01 = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"p3BlockB[KERNEL_WIDTH - 1].s2 = as_Dtype( SUB_GROUP_BLOCK_READ( (const __global INT_TYPE*) (src1_read + 8) ) );",    // NOLINT
"#endif",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"// Perform MADs",    // NOLINT
"Dtype* pBlockB = (Dtype*)blockB;",    // NOLINT
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
"int_tp out0_offset = global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"int_tp out1_offset = global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"__global Dtype *out1 = dst + out1_offset;",    // NOLINT
"",    // NOLINT
"Dtype bias[4];",    // NOLINT
"Dtype4 *bias_vec;",    // NOLINT
"bias_vec = (Dtype4*)bias;",    // NOLINT
"*bias_vec = as_Dtype4(SUB_GROUP_BLOCK_READ4((__global INT_TYPE *)biases + group_x * TILE_N));",    // NOLINT
"if( global_y * TILE_M < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 0 ) ACTIVATION_FUNCTION(dst, out0_offset + ( 0+i) * out_pitch_y, blockC0[0][i] + intel_sub_group_shuffle(bias[0], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 1 ) ACTIVATION_FUNCTION(dst, out0_offset + ( 8+i) * out_pitch_y, blockC0[1][i] + intel_sub_group_shuffle(bias[1], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 2 ) ACTIVATION_FUNCTION(dst, out0_offset + (16+i) * out_pitch_y, blockC0[2][i] + intel_sub_group_shuffle(bias[2], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 3 ) ACTIVATION_FUNCTION(dst, out0_offset + (24+i) * out_pitch_y, blockC0[3][i] + intel_sub_group_shuffle(bias[3], i));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if( global_y * TILE_M + 1 < output_width * output_height )",    // NOLINT
"{",    // NOLINT
"for( int i = 0; i < 8; i++ )",    // NOLINT
"{",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 0 ) ACTIVATION_FUNCTION(dst, out1_offset + ( 0+i) * out_pitch_y, blockC1[0][i] + intel_sub_group_shuffle(bias[0], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 1 ) ACTIVATION_FUNCTION(dst, out1_offset + ( 8+i) * out_pitch_y, blockC1[1][i] + intel_sub_group_shuffle(bias[1], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 2 ) ACTIVATION_FUNCTION(dst, out1_offset + (16+i) * out_pitch_y, blockC1[2][i] + intel_sub_group_shuffle(bias[2], i));",    // NOLINT
"if ( TILE_N_LAST_DIV8 > 3 ) ACTIVATION_FUNCTION(dst, out1_offset + (24+i) * out_pitch_y, blockC1[3][i] + intel_sub_group_shuffle(bias[3], i));",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"#if defined(GEMM_LIKE_CONV_32_2_SIMD16) || defined(GEMM_LIKE_CONV_32_1_SIMD16)",    // NOLINT
"",    // NOLINT
"#define INTERLEAVED_SIMD16_OUTPUT(_out_, _offset_,  _m_) do {    if (global_y * TILE_M < output_width * output_height )     {       if ( ( OUT_DEPTH % TILE_N ) == 0 ) {        for (int i = 0; i < 16; i++)         {           ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_ [i] + intel_sub_group_shuffle(bias[0], i));           ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_ [i] + intel_sub_group_shuffle(bias[1], i));         }       }       else if( ( OUT_DEPTH % 16 ) == 0 ) {         if ( ( global_x + 1 ) < get_global_size(0) ) {           for ( int i = 0; i < 16; i++ )           {             ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_ [i] + intel_sub_group_shuffle(bias[0], i));             ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_ [i] + intel_sub_group_shuffle(bias[1], i));           }         }         else {           for (int i = 0; i < 16; i++)           {             ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_ [i] + intel_sub_group_shuffle(bias[0], i));           }         }       }       else {         if ( ( global_x + 1 ) < get_global_size(0) )         {           for ( int i = 0; i < 16; i++ )           {             ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_[i] + intel_sub_group_shuffle(bias[0], i));             ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_[i] + intel_sub_group_shuffle(bias[1], i));           }         }         else {           if ( (OUT_DEPTH % TILE_N) > 16 ) {             for (int i = 0; i < 16 ; i++)             {               ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_[i] + intel_sub_group_shuffle(bias[0], i));             }             for (int i = 0; i < OUT_DEPTH % 16 ; i++)             {               ACTIVATION_FUNCTION(_out_, _offset_ + (16+i) * out_pitch_y, blockC1 ##_m_[i] + intel_sub_group_shuffle(bias[1], i));             }           }           else {             for (int i = 0; i < OUT_DEPTH % 16 ; i++)             {               ACTIVATION_FUNCTION(_out_, _offset_ + ( 0+i) * out_pitch_y, blockC0 ##_m_[i] + intel_sub_group_shuffle(bias[0], i));             }           }         }       }     }  }while(0)",    // NOLINT
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
"#if INPUT_PAD_H != 0 || INPUT_PAD_W != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
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
"src0_read += ROW_PITCH * DILATION_Y;",    // NOLINT
"INT_TYPE blockB00[KERNEL_WIDTH * 2];",    // NOLINT
"INT_TYPE4* p4BlockB00 = (INT_TYPE4*)blockB00;",    // NOLINT
"INT_TYPE2* p2BlockB00 = (INT_TYPE2*)blockB00;",    // NOLINT
"Dtype* pBlockB00  = (Dtype*)blockB00;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p4BlockB00[interleaved_y] = SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p2BlockB00[KERNEL_WIDTH - 1] = SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read );",    // NOLINT
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
"int_tp out_offset = global_z * out_pitch_z                                                   // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                       // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset",    // NOLINT
"+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"__global Dtype *out = dst + out_offset;",    // NOLINT
"",    // NOLINT
"Dtype bias[2];",    // NOLINT
"Dtype2 *bias_vec;",    // NOLINT
"bias_vec = (Dtype2*)bias;",    // NOLINT
"*bias_vec = as_Dtype2(SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)biases + group_x * TILE_N));",    // NOLINT
"// Work around a potential compiler bug.",    // NOLINT
"if (group_x > 0xFFFFFFFEul) {",    // NOLINT
"out[0] = bias[0] + bias[1];",    // NOLINT
"}",    // NOLINT
"INTERLEAVED_SIMD16_OUTPUT(dst, out_offset, 0);",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#ifdef GEMM_LIKE_CONV_32_2_SIMD16",    // NOLINT
"",    // NOLINT
"//////////////////////////////////////////////////////////////////////////////",    // NOLINT
"// Conv_Interleaved_32_2_SIMD16",    // NOLINT
"//",    // NOLINT
"// Convolution: each workitem computes 1 patch x 32 filters worth of output",    // NOLINT
"// data.",    // NOLINT
"#define TILE_M          2",    // NOLINT
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
"#define DOT_PRODUCT_16( _result, _rowA, colB )        {           _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );          _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );          _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );          _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );          _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );          _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );          _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );          _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );          _result.s8 = mad( _rowA, sub_group_broadcast( colB, 8 ), _result.s8 );          _result.s9 = mad( _rowA, sub_group_broadcast( colB, 9 ), _result.s9 );          _result.sa = mad( _rowA, sub_group_broadcast( colB, 10 ), _result.sa );          _result.sb = mad( _rowA, sub_group_broadcast( colB, 11 ), _result.sb );          _result.sc = mad( _rowA, sub_group_broadcast( colB, 12 ), _result.sc );          _result.sd = mad( _rowA, sub_group_broadcast( colB, 13 ), _result.sd );          _result.se = mad( _rowA, sub_group_broadcast( colB, 14 ), _result.se );          _result.sf = mad( _rowA, sub_group_broadcast( colB, 15 ), _result.sf );      }",    // NOLINT
"typedef CAT( Dtype, KERNEL_WIDTH ) Dtype_t;",    // NOLINT
"",    // NOLINT
"// True for all threads if filter_width is multiple of TILE_N",    // NOLINT
"// else, true for all but right-most column of threads.",    // NOLINT
"{",    // NOLINT
"// Result ctile (*dst) is M rows x N columns",    // NOLINT
"// LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.",    // NOLINT
"Dtype16  blockC00 = 0.f;",    // NOLINT
"Dtype16  blockC10 = 0.f;",    // NOLINT
"Dtype16  blockC01 = 0.f;",    // NOLINT
"Dtype16  blockC11 = 0.f;",    // NOLINT
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
"const __global Dtype *src0_read0 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y0 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x0 - INPUT_PAD_W;                // x offset",    // NOLINT
"const __global Dtype *src0_read1 = src0",    // NOLINT
"+ aligned_input_size * global_z                                            // batch offset",    // NOLINT
"+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH   // y offset",    // NOLINT
"+ curr_x1 - INPUT_PAD_W;                // x offset",    // NOLINT
"",    // NOLINT
"// Src1 (filter) is directly used as btile.",    // NOLINT
"// It starts at the top of src1 and walks down.",    // NOLINT
"// btile is K rows x N columns.",    // NOLINT
"const __global Dtype *src1_read = src1 + ( global_x * TILE_N * 2);",    // NOLINT
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
"// Kernel data is partially interleaved.  Every 2 rows are interleaved at Dtype8 granularity.",    // NOLINT
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
"Dtype_t blockA00 = ( (const __global Dtype_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;",    // NOLINT
"Dtype_t blockA01 = ( (const __global Dtype_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"Dtype*  pblockA01 = (Dtype*)(&blockA01);",    // NOLINT
"#else",    // NOLINT
"Dtype_t blockA00;",    // NOLINT
"Dtype*  pblockA00 = (Dtype*)(&blockA00);",    // NOLINT
"int pos = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH, pos,",    // NOLINT
"{",    // NOLINT
"if (curr_y0 >= INPUT_PAD_H && curr_y0 < input_height + INPUT_PAD_H && curr_x0 + pos * DILATION_X >= INPUT_PAD_W && curr_x0 + pos * DILATION_X < input_width + INPUT_PAD_W)",    // NOLINT
"pblockA00[pos] = src0_read0[pos * DILATION_X];",    // NOLINT
"else",    // NOLINT
"pblockA00[pos] = 0;",    // NOLINT
"})",    // NOLINT
"curr_y0 += DILATION_Y;",    // NOLINT
"Dtype_t blockA01;",    // NOLINT
"Dtype*  pblockA01 = (Dtype*)(&blockA01);",    // NOLINT
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
"Dtype blockB00[KERNEL_WIDTH*2];",    // NOLINT
"Dtype4* p4BlockB00 = (Dtype4*)blockB00;",    // NOLINT
"Dtype2* p2BlockB00 = (Dtype2*)blockB00;",    // NOLINT
"Dtype*  pBlockB00 =  (Dtype* )blockB00;",    // NOLINT
"",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"p4BlockB00[interleaved_y] = as_Dtype4( SUB_GROUP_BLOCK_READ4( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"p2BlockB00[KERNEL_WIDTH - 1] = as_Dtype2( SUB_GROUP_BLOCK_READ2( (const __global INT_TYPE*)src1_read ) );",    // NOLINT
"src1_read += WIDTH1 * 2;",    // NOLINT
"}",    // NOLINT
"// Perform MADs",    // NOLINT
"kernel_idx = 0;",    // NOLINT
"interleaved_y = 0;",    // NOLINT
"LOOP(KERNEL_WIDTH_DIV2, interleaved_y,",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_16( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_16( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_16( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_16( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_16( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"} )",    // NOLINT
"if ( kernel_width_is_odd )",    // NOLINT
"{",    // NOLINT
"kernel_y = interleaved_y * 2;",    // NOLINT
"DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_16( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );",    // NOLINT
"DOT_PRODUCT_16( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"//while( ++patch_row < 1 ); //debug",    // NOLINT
"while( ++patch_row < KERNEL_HEIGHT );",    // NOLINT
"#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0 || DILATION_X != 1 || DILATION_Y != 1",    // NOLINT
"curr_y0 = saved_y0;",    // NOLINT
"curr_y1 = saved_y1;",    // NOLINT
"#endif",    // NOLINT
"src0_read0 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y); // reset to start of next slice of patch",    // NOLINT
"src0_read1 += slice_pitch - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y);",    // NOLINT
"}",    // NOLINT
"//while ( ++patch_depth < 1 );  //debug",    // NOLINT
"while ( ++patch_depth < INPUT_DEPTH );",    // NOLINT
"",    // NOLINT
"// Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:",    // NOLINT
"// (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.",    // NOLINT
"int_tp out0_offset = global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 0 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"int_tp out1_offset = global_z * out_pitch_z                                                       // batch offset",    // NOLINT
"+ ( group_x * TILE_N ) * out_pitch_y                                           // channel offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) / output_width + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset",    // NOLINT
"+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;               // x offset",    // NOLINT
"",    // NOLINT
"Dtype bias[2];",    // NOLINT
"Dtype2 *bias_vec;",    // NOLINT
"bias_vec = (Dtype2*)bias;",    // NOLINT
"*bias_vec = as_Dtype2(SUB_GROUP_BLOCK_READ2((__global INT_TYPE *)biases + group_x * TILE_N));",    // NOLINT
"",    // NOLINT
"INTERLEAVED_SIMD16_OUTPUT(dst, out0_offset, 0);",    // NOLINT
"INTERLEAVED_SIMD16_OUTPUT(dst, out1_offset, 1);",    // NOLINT
"}",    // NOLINT
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
"const KERNEL_ARG_DTYPE scale,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] * (Dtype)((mask[index] > threshold)?1.0:0.0) * scale;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(dropout_backward,Dtype)(",    // NOLINT
"const int_tp n, __global const Dtype* in_diff,",    // NOLINT
"__global const uint_tp* mask, const uint_tp threshold,",    // NOLINT
"const KERNEL_ARG_DTYPE scale,",    // NOLINT
"__global Dtype* out_diff) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out_diff[index] = in_diff[index] * (Dtype)((mask[index] > threshold)?1.0:0.0) * scale;",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"KERNEL_ARG_DTYPE alpha) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] > 0 ? in[index] : alpha * (exp(in[index]) - (Dtype)1.0);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(elu_backward,Dtype)(const int n, __global const Dtype* in_diff,",    // NOLINT
"__global const Dtype* out_data,",    // NOLINT
"__global const Dtype* in_data,",    // NOLINT
"__global Dtype* out_diff,",    // NOLINT
"KERNEL_ARG_DTYPE alpha) {",    // NOLINT
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
"",    // NOLINT
"// atomic_add fddrom: http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html",    // NOLINT
"#if (TYPE == TYPE_HALF)",    // NOLINT
"",    // NOLINT
"// FIXME, has bug which may hang GPU.",    // NOLINT
"inline void TEMPLATE(atomic_add,Dtype)(volatile __global Dtype *source, const Dtype operand) {",    // NOLINT
"union {",    // NOLINT
"uint_tp intVal;",    // NOLINT
"Dtype floatVal[2];",    // NOLINT
"} newVal;",    // NOLINT
"union {",    // NOLINT
"uint_tp intVal;",    // NOLINT
"Dtype floatVal[2];",    // NOLINT
"} prevVal;",    // NOLINT
"do {",    // NOLINT
"// FIXME, need to consider buffer overflow.",    // NOLINT
"prevVal.floatVal[0] = *source;",    // NOLINT
"prevVal.floatVal[1] = *(source+1);",    // NOLINT
"newVal.floatVal[0] = prevVal.floatVal[0] + operand;",    // NOLINT
"newVal.floatVal[1] = prevVal.floatVal[1];",    // NOLINT
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
"",    // NOLINT
"",    // NOLINT
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
"__kernel void TEMPLATE(fft_phony,Dtype)(KERNEL_ARG_DTYPE arg) {",    // NOLINT
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
"cdotc.x += dot(cdotc4.xz, (Dtype2)(1));",    // NOLINT
"cdotc.y += dot(cdotc4.yw, (Dtype2)(1));",    // NOLINT
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
"__kernel void TEMPLATE(fill,Dtype)(const int_tp n, const KERNEL_ARG_DTYPE alpha, __global Dtype* x,",    // NOLINT
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
"#if defined(cl_intel_subgroups)",    // NOLINT
"#pragma OPENCL EXTENSION  cl_intel_subgroups : enable",    // NOLINT
"",    // NOLINT
"#if TYPE != TYPE_DOUBLE",    // NOLINT
"",    // NOLINT
"#define TILE_M          32",    // NOLINT
"#define TILE_K          8",    // NOLINT
"",    // NOLINT
"// common block to calculate (alpha * AxB + beta * C) and output to destination image.",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define SUBGROUP_BLOCK_READ8( __image, __coord ) intel_sub_group_block_read_us8( __image, __coord )",    // NOLINT
"#define SHUFFLE_TYPE2(val) as_ushort2(val)",    // NOLINT
"#define SHUFFLE_TYPE8(val) as_ushort8(val)",    // NOLINT
"#define READ_IMAGE(__image, __coord) read_imageh(__image, sampler, __coord)",    // NOLINT
"#define SIZE_OF_ELEMENT sizeof(ushort)",    // NOLINT
"#define SIMD_SIZE_GEMM 16",    // NOLINT
"#define TILE_N 16",    // NOLINT
"#else",    // NOLINT
"#define SUBGROUP_BLOCK_READ8( __image, __coord ) intel_sub_group_block_read8( __image, __coord )",    // NOLINT
"#define SHUFFLE_TYPE2(val) val",    // NOLINT
"#define SHUFFLE_TYPE8(val) val",    // NOLINT
"#define READ_IMAGE(__image, __coord) read_imagef(__image, sampler, __coord)",    // NOLINT
"#define SIZE_OF_ELEMENT sizeof(uint)",    // NOLINT
"#define SIMD_SIZE_GEMM 8",    // NOLINT
"#define TILE_N 8",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"//#define USE_IMAGE_C",    // NOLINT
"#ifdef USE_IMAGE_C",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define BLOCKC_READ8( _C, _coordC ) as_Dtype8( intel_sub_group_block_read_us8( _C, _coordC ) )",    // NOLINT
"#define BLOCKC_WRITE8( _C, _coordC, _val ) intel_sub_group_block_write_us8( _C, _coordC, as_ushort8( _val ) )",    // NOLINT
"#else",    // NOLINT
"#define BLOCKC_READ8( _C, _coordC ) as_Dtype8( intel_sub_group_block_read8( _C, _coordC ) )",    // NOLINT
"#define BLOCKC_WRITE8( _C, _coordC, _val ) intel_sub_group_block_write8( _C, _coordC, as_uint8( _val ) )",    // NOLINT
"#endif",    // NOLINT
"#define MATC_PARAMETER __read_only image2d_t C, __write_only image2d_t dst",    // NOLINT
"#define GEMM_OUTPUT(ALPHA1, BETA_NOT0) GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, C, dst, sizeof(uint))",    // NOLINT
"#else",    // NOLINT
"#define BLOCKC_READ8( _C, _coordC )           (Dtype8) ( (_coordC.x + get_local_id(0) < N && _coordC.y < M) ? _C[ _coordC.y * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 1 < M) ? _C[ ( _coordC.y + 1 ) * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 2 < M) ? _C[ ( _coordC.y + 2 ) * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 3 < M) ? _C[ ( _coordC.y + 3 ) * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 4 < M) ? _C[ ( _coordC.y + 4 ) * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 5 < M) ? _C[ ( _coordC.y + 5 ) * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 6 < M) ? _C[ ( _coordC.y + 6 ) * ldc + _coordC.x + get_local_id(0) ] : 0,                      (_coordC.x + get_local_id(0) < N && _coordC.y + 7 < M) ? _C[ ( _coordC.y + 7 ) * ldc + _coordC.x + get_local_id(0) ] : 0)",    // NOLINT
"",    // NOLINT
"#define BLOCKC_WRITE8( _C, _coordC, _val) do {                     if (_coordC.x + get_local_id(0) < N) {                        if (_coordC.y < M)                          _C[ _coordC.y * ldc + _coordC.x + get_local_id(0) ] = _val.s0;                        if (_coordC.y + 1 < M)                          _C[ ( _coordC.y + 1 )* ldc + _coordC.x + get_local_id(0) ] = _val.s1;                        if (_coordC.y + 2 < M)                          _C[ ( _coordC.y + 2 )* ldc + _coordC.x + get_local_id(0) ] = _val.s2;                        if (_coordC.y + 3 < M)                          _C[ ( _coordC.y + 3 )* ldc + _coordC.x + get_local_id(0) ] = _val.s3;                        if (_coordC.y + 4 < M)                          _C[ ( _coordC.y + 4 )* ldc + _coordC.x + get_local_id(0) ] = _val.s4;                        if (_coordC.y + 5 < M)                          _C[ ( _coordC.y + 5 )* ldc + _coordC.x + get_local_id(0) ] = _val.s5;                        if (_coordC.y + 6 < M)                          _C[ ( _coordC.y + 6 )* ldc + _coordC.x + get_local_id(0) ] = _val.s6;                        if (_coordC.y + 7 < M)                          _C[ ( _coordC.y + 7 )* ldc + _coordC.x + get_local_id(0) ] = _val.s7;                      }} while(0)",    // NOLINT
"#define MATC_PARAMETER __global Dtype * C, const int offC, const int M, const int N, const int ldc",    // NOLINT
"#define GEMM_OUTPUT(ALPHA1, BETA_NOT0) GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, (C + offC), (C + offC), 1)",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, _C, _dst, _C_step)     int2    coordDst = (int2)( ( group_x * TILE_N ) * _C_step, ( group_y * TILE_M ) );     int2    coordC = coordDst;     Dtype8 blockC00;     Dtype8 blockC01;     Dtype8 blockC02;     Dtype8 blockC03;     if (BETA_NOT0) {         blockC00 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8;         blockC01 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8;         blockC02 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8;         blockC03 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );         if (!ALPHA1) {             blockC00 = mad(blockAxB00, (Dtype8)alpha, blockC00);             blockC01 = mad(blockAxB01, (Dtype8)alpha, blockC01);             blockC02 = mad(blockAxB02, (Dtype8)alpha, blockC02);             blockC03 = mad(blockAxB03, (Dtype8)alpha, blockC03);         } else {             blockC00 += blockAxB00;             blockC01 += blockAxB01;             blockC02 += blockAxB02;             blockC03 += blockAxB03;         }     } else {         blockC00 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8;         blockC01 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8;         blockC02 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8;         blockC03 = isFirstColBlock ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );         if (!ALPHA1) {           blockC00 = mad(blockAxB00, (Dtype8)alpha, blockC00);           blockC01 = mad(blockAxB01, (Dtype8)alpha, blockC01);           blockC02 = mad(blockAxB02, (Dtype8)alpha, blockC02);           blockC03 = mad(blockAxB03, (Dtype8)alpha, blockC03);         } else {           blockC00 += blockAxB00;           blockC01 += blockAxB01;           blockC02 += blockAxB02;           blockC03 += blockAxB03;         }     }     BLOCKC_WRITE8( _dst, coordDst, blockC00 );    coordDst.y += 8;     BLOCKC_WRITE8( _dst, coordDst, blockC01 );    coordDst.y += 8;     BLOCKC_WRITE8( _dst, coordDst, blockC02 );    coordDst.y += 8;     BLOCKC_WRITE8( _dst, coordDst, blockC03 );",    // NOLINT
"",    // NOLINT
"// Get the specified column of the block of the block",    // NOLINT
"#define TRANSPOSE_BLOCK_8( _block, _col )           (Dtype8)( intel_sub_group_shuffle( _block.s0, _col ),                     intel_sub_group_shuffle( _block.s1, _col ),                     intel_sub_group_shuffle( _block.s2, _col ),                     intel_sub_group_shuffle( _block.s3, _col ),                     intel_sub_group_shuffle( _block.s4, _col ),                     intel_sub_group_shuffle( _block.s5, _col ),                     intel_sub_group_shuffle( _block.s6, _col ),                     intel_sub_group_shuffle( _block.s7, _col ) );",    // NOLINT
"",    // NOLINT
"// A's column block multiply B 's row block.",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB00, _blockB01 )            {               const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );                const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );                const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );                const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );                const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );                const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );                const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );                const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );                const Dtype8    acol8 = TRANSPOSE_BLOCK_8( _blockA, 8 );                const Dtype8    acol9 = TRANSPOSE_BLOCK_8( _blockA, 9 );                const Dtype8    acola = TRANSPOSE_BLOCK_8( _blockA, 10 );                const Dtype8    acolb = TRANSPOSE_BLOCK_8( _blockA, 11 );                const Dtype8    acolc = TRANSPOSE_BLOCK_8( _blockA, 12 );                const Dtype8    acold = TRANSPOSE_BLOCK_8( _blockA, 13 );                const Dtype8    acole = TRANSPOSE_BLOCK_8( _blockA, 14 );                const Dtype8    acolf = TRANSPOSE_BLOCK_8( _blockA, 15 );                _result = mad( (Dtype8)(_blockB00.s0), acol0, _result );                  _result = mad( (Dtype8)(_blockB00.s1), acol1, _result );                  _result = mad( (Dtype8)(_blockB00.s2), acol2, _result );                  _result = mad( (Dtype8)(_blockB00.s3), acol3, _result );                  _result = mad( (Dtype8)(_blockB00.s4), acol4, _result );                  _result = mad( (Dtype8)(_blockB00.s5), acol5, _result );                  _result = mad( (Dtype8)(_blockB00.s6), acol6, _result );                  _result = mad( (Dtype8)(_blockB00.s7), acol7, _result );                  _result = mad( (Dtype8)(_blockB01.s0), acol8, _result );                  _result = mad( (Dtype8)(_blockB01.s1), acol9, _result );                  _result = mad( (Dtype8)(_blockB01.s2), acola, _result );                  _result = mad( (Dtype8)(_blockB01.s3), acolb, _result );                  _result = mad( (Dtype8)(_blockB01.s4), acolc, _result );                  _result = mad( (Dtype8)(_blockB01.s5), acold, _result );                  _result = mad( (Dtype8)(_blockB01.s6), acole, _result );                  _result = mad( (Dtype8)(_blockB01.s7), acolf, _result );              }",    // NOLINT
"#else",    // NOLINT
"#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )            {               const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );                const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );                const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );                const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );                const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );                const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );                const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );                const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );                _result = mad( (Dtype8)(_blockB.s0), acol0, _result );                  _result = mad( (Dtype8)(_blockB.s1), acol1, _result );                  _result = mad( (Dtype8)(_blockB.s2), acol2, _result );                  _result = mad( (Dtype8)(_blockB.s3), acol3, _result );                  _result = mad( (Dtype8)(_blockB.s4), acol4, _result );                  _result = mad( (Dtype8)(_blockB.s5), acol5, _result );                  _result = mad( (Dtype8)(_blockB.s6), acol6, _result );                  _result = mad( (Dtype8)(_blockB.s7), acol7, _result );              }",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define GEMM_NN(ALPHA1, BETA_NOT0) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_NN_ ##ALPHA1 ##_ ##BETA_NOT0, Dtype)(     __read_only image2d_t A,     __read_only image2d_t B,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int width0,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);     const int group_y = get_group_id(1);     Dtype8 blockAxB00 = 0;     Dtype8 blockAxB01 = 0;     Dtype8 blockAxB02 = 0;     Dtype8 blockAxB03 = 0;     int2    coordA = (int2)( 0, group_y * TILE_M );     int2    coordB = (int2)( ( group_x * TILE_N ) * SIZE_OF_ELEMENT, 0 );     do     {          int2    coordBTemp = coordB;         Dtype8  blockB00 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K;         Dtype8  blockB01 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K;         int2    coordATemp = coordA;         Dtype8  blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8  blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8  blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8  blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.x += TILE_K * SIZE_OF_ELEMENT * 2;         MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00, blockB01 );         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00, blockB01 );         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00, blockB01 );         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00, blockB01 );     }     while( coordB.y < width0 );     GEMM_OUTPUT(ALPHA1, BETA_NOT0);  }",    // NOLINT
"#else",    // NOLINT
"#define GEMM_NN(ALPHA1, BETA_NOT0) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_NN_ ##ALPHA1 ##_ ##BETA_NOT0, Dtype)(     __read_only image2d_t A,     __read_only image2d_t B,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int width0,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);     const int group_y = get_group_id(1);     Dtype8 blockAxB00 = 0.0f;     Dtype8 blockAxB01 = 0.0f;     Dtype8 blockAxB02 = 0.0f;     Dtype8 blockAxB03 = 0.0f;     int2    coordA = (int2)( 0, group_y * TILE_M );     int2    coordB = (int2)( ( group_x * TILE_N ) * SIZE_OF_ELEMENT, 0 );     do     {          int2    coordBTemp = coordB;         Dtype8  blockB00 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K;         int2    coordATemp = coordA;         Dtype8  blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8  blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8  blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8  blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.x += TILE_K * SIZE_OF_ELEMENT;         MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 );     }     while( coordB.y < width0 );     GEMM_OUTPUT(ALPHA1, BETA_NOT0); }",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"GEMM_NN(1, 0) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_NN(1, 1) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_NN(0, 0) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_NN(0, 1) // ALPHA != 1, BETA != 0",    // NOLINT
"",    // NOLINT
"#undef TRANSPOSE_BLOCK_8",    // NOLINT
"#undef MULTIPLY_BLOCKS_8x8",    // NOLINT
"#undef GEMM_NN",    // NOLINT
"",    // NOLINT
"// replicate the first row to column block.",    // NOLINT
"#define TRANSPOSE_BLOCK_8(_vec, _col)         (Dtype8)( intel_sub_group_shuffle(_vec, _col + 0),                   intel_sub_group_shuffle(_vec, _col + 1),                   intel_sub_group_shuffle(_vec, _col + 2),                   intel_sub_group_shuffle(_vec, _col + 3),                   intel_sub_group_shuffle(_vec, _col + 4),                   intel_sub_group_shuffle(_vec, _col + 5),                   intel_sub_group_shuffle(_vec, _col + 6),                   intel_sub_group_shuffle(_vec, _col + 7) )",    // NOLINT
"",    // NOLINT
"#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB, _col )            {               _result = mad( (Dtype8)(_blockB.s0), TRANSPOSE_BLOCK_8(_blockA.s0, _col), _result );                  _result = mad( (Dtype8)(_blockB.s1), TRANSPOSE_BLOCK_8(_blockA.s1, _col), _result );                  _result = mad( (Dtype8)(_blockB.s2), TRANSPOSE_BLOCK_8(_blockA.s2, _col), _result );                  _result = mad( (Dtype8)(_blockB.s3), TRANSPOSE_BLOCK_8(_blockA.s3, _col), _result );                  _result = mad( (Dtype8)(_blockB.s4), TRANSPOSE_BLOCK_8(_blockA.s4, _col), _result );                  _result = mad( (Dtype8)(_blockB.s5), TRANSPOSE_BLOCK_8(_blockA.s5, _col), _result );                  _result = mad( (Dtype8)(_blockB.s6), TRANSPOSE_BLOCK_8(_blockA.s6, _col), _result );                  _result = mad( (Dtype8)(_blockB.s7), TRANSPOSE_BLOCK_8(_blockA.s7, _col), _result );              }",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define GEMM_TN(ALPHA1, BETA_NOT0) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_TN_ ##ALPHA1 ##_ ##BETA_NOT0,Dtype)(     __read_only image2d_t A,     __read_only image2d_t B,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int width0,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);    const int group_y = get_group_id(1);    Dtype8 blockAxB00 = 0;    Dtype8 blockAxB01 = 0;    Dtype8 blockAxB02 = 0;    Dtype8 blockAxB03 = 0;    int2    coordA = (int2)( group_y * TILE_M * SIZE_OF_ELEMENT, 0 );    int2    coordB = (int2)( ( group_x * TILE_N ) * SIZE_OF_ELEMENT, 0 );    do    {        int2    coordBTemp = coordB;        Dtype8 blockB00 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K;        int2    coordATemp = coordA;        Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 16 * SIZE_OF_ELEMENT;        Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.y += TILE_K;        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00, 0);         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA00, blockB00, 8);         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA01, blockB00, 0);         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA01, blockB00, 8);     }     while( coordB.y < width0 );     GEMM_OUTPUT(ALPHA1, BETA_NOT0); }",    // NOLINT
"#else",    // NOLINT
"#define GEMM_TN(ALPHA1, BETA_NOT0) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_TN_ ##ALPHA1 ##_ ##BETA_NOT0,Dtype)(     __read_only image2d_t A,     __read_only image2d_t B,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int width0,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);    const int group_y = get_group_id(1);    Dtype8 blockAxB00 = 0.0f;    Dtype8 blockAxB01 = 0.0f;    Dtype8 blockAxB02 = 0.0f;    Dtype8 blockAxB03 = 0.0f;    int2    coordA = (int2)( group_y * TILE_M * SIZE_OF_ELEMENT, 0 );    int2    coordB = (int2)( ( group_x * TILE_N ) * SIZE_OF_ELEMENT, 0 );    do    {        int2    coordBTemp = coordB;        Dtype8 blockB00 = as_Dtype8( SUBGROUP_BLOCK_READ8( B, coordBTemp ) );    coordB.y += TILE_K;        int2    coordATemp = coordA;        Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;        Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;        Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;        Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.y += TILE_K;        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00, 0 );         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00, 0 );         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00, 0 );         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00, 0 );     }     while( coordB.y < width0 );     GEMM_OUTPUT(ALPHA1, BETA_NOT0); }",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"GEMM_TN(1, 0) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_TN(1, 1) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_TN(0, 0) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_TN(0, 1) // ALPHA != 1, BETA != 0",    // NOLINT
"",    // NOLINT
"#undef MULTIPLY_BLOCKS_8x8",    // NOLINT
"#undef TRANSPOSE_BLOCK_8",    // NOLINT
"#undef GEMM_TN",    // NOLINT
"",    // NOLINT
"// The same as GEMM_NN",    // NOLINT
"#define TRANSPOSE_BLOCK_8( _block, _col )           (Dtype8)( intel_sub_group_shuffle( _block.s0, _col),                     intel_sub_group_shuffle( _block.s1, _col),                     intel_sub_group_shuffle( _block.s2, _col),                     intel_sub_group_shuffle( _block.s3, _col),                     intel_sub_group_shuffle( _block.s4, _col),                     intel_sub_group_shuffle( _block.s5, _col),                     intel_sub_group_shuffle( _block.s6, _col),                     intel_sub_group_shuffle( _block.s7, _col) )",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )            {               const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );                const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );                const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );                const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );                const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );                const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );                const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );                const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );                const Dtype8    acol8 = TRANSPOSE_BLOCK_8( _blockA, 8 );                const Dtype8    acol9 = TRANSPOSE_BLOCK_8( _blockA, 9 );                const Dtype8    acola = TRANSPOSE_BLOCK_8( _blockA, 10 );                const Dtype8    acolb = TRANSPOSE_BLOCK_8( _blockA, 11 );                const Dtype8    acolc = TRANSPOSE_BLOCK_8( _blockA, 12 );                const Dtype8    acold = TRANSPOSE_BLOCK_8( _blockA, 13 );                const Dtype8    acole = TRANSPOSE_BLOCK_8( _blockA, 14 );                const Dtype8    acolf = TRANSPOSE_BLOCK_8( _blockA, 15 );                _result = mad( (Dtype8)_blockB.s0, acol0, _result );                  _result = mad( (Dtype8)_blockB.s1, acol1, _result );                  _result = mad( (Dtype8)_blockB.s2, acol2, _result );                  _result = mad( (Dtype8)_blockB.s3, acol3, _result );                  _result = mad( (Dtype8)_blockB.s4, acol4, _result );                  _result = mad( (Dtype8)_blockB.s5, acol5, _result );                  _result = mad( (Dtype8)_blockB.s6, acol6, _result );                  _result = mad( (Dtype8)_blockB.s7, acol7, _result );                  _result = mad( (Dtype8)_blockB.s8, acol8, _result );                  _result = mad( (Dtype8)_blockB.s9, acol9, _result );                  _result = mad( (Dtype8)_blockB.sa, acola, _result );                  _result = mad( (Dtype8)_blockB.sb, acolb, _result );                  _result = mad( (Dtype8)_blockB.sc, acolc, _result );                  _result = mad( (Dtype8)_blockB.sd, acold, _result );                  _result = mad( (Dtype8)_blockB.se, acole, _result );                  _result = mad( (Dtype8)_blockB.sf, acolf, _result );              }",    // NOLINT
"#else",    // NOLINT
"#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )            {               const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );                const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );                const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );                const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );                const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );                const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );                const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );                const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );                _result = mad( (Dtype8)_blockB.s0, acol0, _result );                  _result = mad( (Dtype8)_blockB.s1, acol1, _result );                  _result = mad( (Dtype8)_blockB.s2, acol2, _result );                  _result = mad( (Dtype8)_blockB.s3, acol3, _result );                  _result = mad( (Dtype8)_blockB.s4, acol4, _result );                  _result = mad( (Dtype8)_blockB.s5, acol5, _result );                  _result = mad( (Dtype8)_blockB.s6, acol6, _result );                  _result = mad( (Dtype8)_blockB.s7, acol7, _result );              }",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define GEMM_NT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_NT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0,Dtype)(     __read_only image2d_t A,     MATB_PARAMETER,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int padded_k,     int k,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);     const int group_y = get_group_id(1);     Dtype8 blockAxB00 = 0;     Dtype8 blockAxB01 = 0;     Dtype8 blockAxB02 = 0;     Dtype8 blockAxB03 = 0;     int2    coordA = (int2)( 0, group_y * TILE_M );     int2    coordB = (int2)( 0, ( group_x * TILE_N ));     const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;     do     {         Dtype16 blockB00;         BLOCKB_READ8(blockB00, B, coordB);         int2    coordATemp = coordA;         Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.x += TILE_K * SIZE_OF_ELEMENT * 2;         MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 );     }     while( coordB.x < padded_k / VECSIZE );     GEMM_OUTPUT(ALPHA1, BETA_NOT0); }",    // NOLINT
"#else",    // NOLINT
"#define GEMM_NT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_NT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0,Dtype)(     __read_only image2d_t A,     MATB_PARAMETER,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int padded_k,     int k,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);     const int group_y = get_group_id(1);     Dtype8 blockAxB00 = 0.0f;     Dtype8 blockAxB01 = 0.0f;     Dtype8 blockAxB02 = 0.0f;     Dtype8 blockAxB03 = 0.0f;     int2    coordA = (int2)( 0, group_y * TILE_M );     int2    coordB = (int2)( 0, ( group_x * TILE_N ));     const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;     do     {         Dtype8 blockB00;          BLOCKB_READ8(blockB00, B, coordB);         int2    coordATemp = coordA;         Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.y += 8;         Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.x += TILE_K * SIZE_OF_ELEMENT;         MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 );         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 );     }     while( coordB.x < padded_k / VECSIZE );     GEMM_OUTPUT(ALPHA1, BETA_NOT0); }",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         _blockb.s0123 = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s4567 = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s89ab = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.scdef = READ_IMAGE(_B, _coordBTemp); _coordB.x += 4;",    // NOLINT
"#else",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         _blockb.s0123 = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s4567 = READ_IMAGE(_B, _coordBTemp); _coordB.x += 2;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define MATB_PARAMETER __read_only image2d_t B",    // NOLINT
"",    // NOLINT
"GEMM_NT(1, 0, VEC4, 4) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_NT(1, 1, VEC4, 4) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_NT(0, 0, VEC4, 4) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_NT(0, 1, VEC4, 4) // ALPHA != 1, BETA != 0",    // NOLINT
"#undef BLOCKB_READ8",    // NOLINT
"#undef MATB_PARAMETER",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         const __global float *B_read = (__global float *)(_B + (_coordBTemp.y * ldb) + _coordBTemp.x + offB);         _blockb = as_Dtype16(as_ushort16(vload8(0, B_read)));         _coordB.x += TILE_K * 2;",    // NOLINT
"#else",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         const __global Dtype *B_read = (__global Dtype *)(_B + (_coordBTemp.y * ldb) + _coordBTemp.x + offB);         _blockb = vload8(0, B_read);         _coordB.x += TILE_K;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define MATB_PARAMETER __global Dtype *B, int offB, int ldb",    // NOLINT
"",    // NOLINT
"GEMM_NT(1, 0, BUFFER, 1) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_NT(1, 1, BUFFER, 1) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_NT(0, 0, BUFFER, 1) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_NT(0, 1, BUFFER, 1) // ALPHA != 1, BETA != 0",    // NOLINT
"#undef BLOCKB_READ8",    // NOLINT
"#undef MATB_PARAMETER",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         Dtype4 temp;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s0 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s1 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s2 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s3 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s4 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s5 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s6 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s7 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s8 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s9 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.sa = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.sb = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;          _blockb.sc = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.sd = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.se = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.sf = temp.s0;         _coordB.x += 16;",    // NOLINT
"#else",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         Dtype4 temp;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s0 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s1 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s2 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s3 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s4 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s5 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s6 = temp.s0;         temp = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s7 = temp.s0;         _coordB.x += 8;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define MATB_PARAMETER __read_only image2d_t B",    // NOLINT
"",    // NOLINT
"GEMM_NT(1, 0, SCALAR, 1) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_NT(1, 1, SCALAR, 1) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_NT(0, 0, SCALAR, 1) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_NT(0, 1, SCALAR, 1) // ALPHA != 1, BETA != 0",    // NOLINT
"#undef BLOCKB_READ8",    // NOLINT
"#undef MATB_PARAMETER",    // NOLINT
"",    // NOLINT
"#undef MULTIPLY_BLOCKS_8x8",    // NOLINT
"#undef TRANSPOSE_BLOCK_8",    // NOLINT
"#undef GEMM_NT",    // NOLINT
"",    // NOLINT
"//The same as GEMM_TN.",    // NOLINT
"#define TRANSPOSE_BLOCK_8(_vec, _col)         (Dtype8)( intel_sub_group_shuffle(_vec, _col + 0),                   intel_sub_group_shuffle(_vec, _col + 1),                   intel_sub_group_shuffle(_vec, _col + 2),                   intel_sub_group_shuffle(_vec, _col + 3),                   intel_sub_group_shuffle(_vec, _col + 4),                   intel_sub_group_shuffle(_vec, _col + 5),                   intel_sub_group_shuffle(_vec, _col + 6),                   intel_sub_group_shuffle(_vec, _col + 7) );",    // NOLINT
"",    // NOLINT
"#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB, _col )            {               const Dtype8    acol0 = TRANSPOSE_BLOCK_8( _blockA.s0, _col );                const Dtype8    acol1 = TRANSPOSE_BLOCK_8( _blockA.s1, _col );                const Dtype8    acol2 = TRANSPOSE_BLOCK_8( _blockA.s2, _col );                const Dtype8    acol3 = TRANSPOSE_BLOCK_8( _blockA.s3, _col );                const Dtype8    acol4 = TRANSPOSE_BLOCK_8( _blockA.s4, _col );                const Dtype8    acol5 = TRANSPOSE_BLOCK_8( _blockA.s5, _col );                const Dtype8    acol6 = TRANSPOSE_BLOCK_8( _blockA.s6, _col );                const Dtype8    acol7 = TRANSPOSE_BLOCK_8( _blockA.s7, _col );                _result = mad( (Dtype8)_blockB.s0, acol0, _result );                  _result = mad( (Dtype8)_blockB.s1, acol1, _result );                  _result = mad( (Dtype8)_blockB.s2, acol2, _result );                  _result = mad( (Dtype8)_blockB.s3, acol3, _result );                  _result = mad( (Dtype8)_blockB.s4, acol4, _result );                  _result = mad( (Dtype8)_blockB.s5, acol5, _result );                  _result = mad( (Dtype8)_blockB.s6, acol6, _result );                  _result = mad( (Dtype8)_blockB.s7, acol7, _result );              }",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define GEMM_TT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_TT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0, Dtype)(     __read_only image2d_t A,     MATB_PARAMETER,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int padded_k,     int k,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);     const int group_y = get_group_id(1);     Dtype8 blockAxB00 = 0;     Dtype8 blockAxB01 = 0;     Dtype8 blockAxB02 = 0;     Dtype8 blockAxB03 = 0;     int2    coordA = (int2)( group_y * TILE_M * SIZE_OF_ELEMENT, 0 );     int2    coordB = (int2)( 0, ( group_x * TILE_N ));     const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;     do     {         Dtype8 blockB00;                     BLOCKB_READ8(blockB00, B, coordB);         int2    coordATemp = coordA;         Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 16 * SIZE_OF_ELEMENT;        Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.y += TILE_K;        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00, 0);         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA00, blockB00, 8);         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA01, blockB00, 0);         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA01, blockB00, 8);     }     while( coordB.x < padded_k / VECSIZE );     GEMM_OUTPUT(ALPHA1, BETA_NOT0);}",    // NOLINT
"#else",    // NOLINT
"#define GEMM_TT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) __attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM))) __attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, 1, 1))) __kernel void TEMPLATE(gemm_32_1_TT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0, Dtype)(     __read_only image2d_t A,     MATB_PARAMETER,     MATC_PARAMETER,     KERNEL_ARG_DTYPE alpha_in,     KERNEL_ARG_DTYPE beta_in,     int padded_k,     int k,     int isFirstColBlock) {     const Dtype alpha = (Dtype)alpha_in;     const Dtype beta = (Dtype)beta_in;     const int group_x = get_group_id(0);     const int group_y = get_group_id(1);     Dtype8 blockAxB00 = 0.0f;     Dtype8 blockAxB01 = 0.0f;     Dtype8 blockAxB02 = 0.0f;     Dtype8 blockAxB03 = 0.0f;     int2    coordA = (int2)( group_y * TILE_M * SIZE_OF_ELEMENT, 0 );     int2    coordB = (int2)( 0, ( group_x * TILE_N ));     const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;     do     {         Dtype8 blockB00;                     BLOCKB_READ8(blockB00, B, coordB);         int2    coordATemp = coordA;         Dtype8 blockA00 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;         Dtype8 blockA01 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;         Dtype8 blockA02 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordATemp.x += 8 * SIZE_OF_ELEMENT;         Dtype8 blockA03 = as_Dtype8( SUBGROUP_BLOCK_READ8( A, coordATemp ) );    coordA.y += TILE_K;         MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00 , blockB00, 0 );         MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01 , blockB00, 0 );         MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02 , blockB00, 0 );         MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03 , blockB00, 0 );     }     while( coordB.x < padded_k / VECSIZE );     GEMM_OUTPUT(ALPHA1, BETA_NOT0);}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         _blockb.s0123 = READ_IMAGE(_B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s4567 = READ_IMAGE(_B, _coordBTemp); _coordB.x += 2;",    // NOLINT
"",    // NOLINT
"#define MATB_PARAMETER __read_only image2d_t B",    // NOLINT
"",    // NOLINT
"GEMM_TT(1, 0, VEC4, 4) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_TT(1, 1, VEC4, 4) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_TT(0, 0, VEC4, 4) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_TT(0, 1, VEC4, 4) // ALPHA != 1, BETA != 0",    // NOLINT
"#undef BLOCKB_READ8",    // NOLINT
"#undef MATB_PARAMETER",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         const __global float *B_read = (__global float *)(_B + (_coordBTemp.y * k) + _coordBTemp.x + offB);         _blockb = as_Dtype8(as_ushort8(vload4(0, B_read)));         _coordB.x += TILE_K;",    // NOLINT
"#else",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         const __global Dtype *B_read = (__global Dtype *)(_B + (_coordBTemp.y * k) + _coordBTemp.x + offB);         _blockb = vload8(0, B_read);         _coordB.x += TILE_K;",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define MATB_PARAMETER __global Dtype *B, int offB, int ldb",    // NOLINT
"",    // NOLINT
"GEMM_TT(1, 0, BUFFER, 1) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_TT(1, 1, BUFFER, 1) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_TT(0, 0, BUFFER, 1) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_TT(0, 1, BUFFER, 1) // ALPHA != 1, BETA != 0",    // NOLINT
"#undef BLOCKB_READ8",    // NOLINT
"#undef MATB_PARAMETER",    // NOLINT
"",    // NOLINT
"#define BLOCKB_READ8(_blockb, _B, _coordB)         int2 _coordBTemp = _coordB;         _coordBTemp.y += get_local_id(0);         Dtype4 temp;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s0 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s1 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s2 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s3 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s4 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s5 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s6 = temp.s0;         temp = READ_IMAGE(B, _coordBTemp); _coordBTemp.x += 1;         _blockb.s7 = temp.s0;         _coordB.x += 8;",    // NOLINT
"",    // NOLINT
"#define MATB_PARAMETER __read_only image2d_t B",    // NOLINT
"",    // NOLINT
"GEMM_TT(1, 0, SCALAR, 1) // ALPHA == 1, BETA == 0",    // NOLINT
"GEMM_TT(1, 1, SCALAR, 1) // ALPHA == 1, BETA != 0",    // NOLINT
"GEMM_TT(0, 0, SCALAR, 1) // ALPHA != 1, BETA == 0",    // NOLINT
"GEMM_TT(0, 1, SCALAR, 1) // ALPHA != 1, BETA != 0",    // NOLINT
"#undef BLOCKB_READ8",    // NOLINT
"#undef MATB_PARAMETER",    // NOLINT
"",    // NOLINT
"#undef MULTIPLY_BLOCKS_8x8",    // NOLINT
"#undef TRANSPOSE_BLOCK_8",    // NOLINT
"#undef GEMM_TT",    // NOLINT
"",    // NOLINT
"#undef TILE_M",    // NOLINT
"#undef TILE_K",    // NOLINT
"#undef TILE_N",    // NOLINT
"#undef SUBGROUP_BLOCK_READ8",    // NOLINT
"#undef READ_IMAGE",    // NOLINT
"#undef SIZE_OF_ELEMENT",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_copy_image_transpose, Dtype)(",    // NOLINT
"__global Dtype* A,",    // NOLINT
"__write_only image2d_t ImA,",    // NOLINT
"int offA,",    // NOLINT
"int width,",    // NOLINT
"int height,",    // NOLINT
"int ldA)",    // NOLINT
"{",    // NOLINT
"const int gidx = get_global_id(0);",    // NOLINT
"const int gidy = get_global_id(1);",    // NOLINT
"int2 coord_dst = (int2)(gidx, gidy);",    // NOLINT
"__global Dtype* A_off = A + offA;",    // NOLINT
"Dtype srcA = A_off[gidy * ldA + gidx];",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"write_imageh(ImA, coord_dst, (Dtype4)srcA);",    // NOLINT
"#else",    // NOLINT
"write_imagef(ImA, coord_dst, (Dtype4)srcA);",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_copy_image_no_transpose, Dtype)(",    // NOLINT
"__global Dtype* A,",    // NOLINT
"__write_only image2d_t ImA,",    // NOLINT
"int offA,",    // NOLINT
"int width,",    // NOLINT
"int height,",    // NOLINT
"int ldA)",    // NOLINT
"{",    // NOLINT
"const int gidx = get_global_id(0);",    // NOLINT
"const int gidy = get_global_id(1);",    // NOLINT
"int2 coord_dst = (int2)(gidx, gidy);",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"if (gidx >= width || gidy >= height) {",    // NOLINT
"write_imageh(ImA, coord_dst, 0);",    // NOLINT
"return;",    // NOLINT
"}",    // NOLINT
"__global Dtype* A_off = A + offA;",    // NOLINT
"write_imageh(ImA, coord_dst, A_off[gidy * ldA + gidx]);",    // NOLINT
"#else",    // NOLINT
"if (gidx >= width || gidy >= height) {",    // NOLINT
"write_imageui(ImA, coord_dst, (uint4)0);",    // NOLINT
"return;",    // NOLINT
"}",    // NOLINT
"__global Dtype* A_off = A + offA;",    // NOLINT
"uint4 srcA = convert_uint4(as_uchar4(A_off[gidy * ldA + gidx]));",    // NOLINT
"write_imageui(ImA, coord_dst, srcA);",    // NOLINT
"#endif",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"#define VEC_SIZE        4",    // NOLINT
"#define LWG_HEIGHT      4",    // NOLINT
"#define TILE_M          8",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define TILE_K          32",    // NOLINT
"#define TILE_N          64",    // NOLINT
"#else",    // NOLINT
"#define TILE_K          16",    // NOLINT
"#define TILE_N          32",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, LWG_HEIGHT, 1)))",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM)))",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_NN, Dtype)(",    // NOLINT
"const __global Dtype *src0, int off0,",    // NOLINT
"const __global Dtype *src1, int off1,",    // NOLINT
"__global Dtype *dst, int offd,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_in,",    // NOLINT
"KERNEL_ARG_DTYPE beta_in,",    // NOLINT
"int start_index)",    // NOLINT
"{",    // NOLINT
"const Dtype alpha = (Dtype)alpha_in;",    // NOLINT
"const Dtype beta = (Dtype)beta_in;",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int local_x = get_local_id(0);",    // NOLINT
"const int local_y = get_local_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"",    // NOLINT
"Dtype4 brow;",    // NOLINT
"Dtype2 arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7;",    // NOLINT
"",    // NOLINT
"__global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src0_read = src0 + local_x * (TILE_K / SIMD_SIZE_GEMM) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * K + start_index + off0;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src1_read0 = src1 + local_x * VEC_SIZE + (group_x * TILE_N) + start_index * N + off1;",    // NOLINT
"",    // NOLINT
"int border = -(group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M);",    // NOLINT
"",    // NOLINT
"int row0 = mad24(global_y, TILE_M, 0) < M ? 0 : border;",    // NOLINT
"int row1 = mad24(global_y, TILE_M, 1) < M ? 1 : border;",    // NOLINT
"int row2 = mad24(global_y, TILE_M, 2) < M ? 2 : border;",    // NOLINT
"int row3 = mad24(global_y, TILE_M, 3) < M ? 3 : border;",    // NOLINT
"int row4 = mad24(global_y, TILE_M, 4) < M ? 4 : border;",    // NOLINT
"int row5 = mad24(global_y, TILE_M, 5) < M ? 5 : border;",    // NOLINT
"int row6 = mad24(global_y, TILE_M, 6) < M ? 6 : border;",    // NOLINT
"int row7 = mad24(global_y, TILE_M, 7) < M ? 7 : border;",    // NOLINT
"",    // NOLINT
"Dtype4 dot00 = (start_index != 0) ? vload4(0, dst_write0) : beta * vload4(0, dst_write0);",    // NOLINT
"Dtype4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * N) : beta * vload4(0, dst_write0 + 1 * N);",    // NOLINT
"Dtype4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * N) : beta * vload4(0, dst_write0 + 2 * N);",    // NOLINT
"Dtype4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * N) : beta * vload4(0, dst_write0 + 3 * N);",    // NOLINT
"Dtype4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * N) : beta * vload4(0, dst_write0 + 4 * N);",    // NOLINT
"Dtype4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * N) : beta * vload4(0, dst_write0 + 5 * N);",    // NOLINT
"Dtype4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * N) : beta * vload4(0, dst_write0 + 6 * N);",    // NOLINT
"Dtype4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * N) : beta * vload4(0, dst_write0 + 7 * N);",    // NOLINT
"",    // NOLINT
"int end_index = min(start_index + 256, K);",    // NOLINT
"int w = start_index;",    // NOLINT
"while( w + TILE_K <= end_index ) {",    // NOLINT
"arow0 = alpha * vload2(0, src0_read + row0 * K);",    // NOLINT
"arow1 = alpha * vload2(0, src0_read + row1 * K);",    // NOLINT
"arow2 = alpha * vload2(0, src0_read + row2 * K);",    // NOLINT
"arow3 = alpha * vload2(0, src0_read + row3 * K);",    // NOLINT
"arow4 = alpha * vload2(0, src0_read + row4 * K);",    // NOLINT
"arow5 = alpha * vload2(0, src0_read + row5 * K);",    // NOLINT
"arow6 = alpha * vload2(0, src0_read + row6 * K);",    // NOLINT
"arow7 = alpha * vload2(0, src0_read + row7 * K);",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( index, suffix )           brow = vload4(0, src1_read0);  src1_read0 += N;         dot00 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow0), index )).s##suffix), brow, dot00 );         dot01 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow1), index )).s##suffix), brow, dot01 );         dot02 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow2), index )).s##suffix), brow, dot02 );         dot03 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow3), index )).s##suffix), brow, dot03 );         dot04 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow4), index )).s##suffix), brow, dot04 );         dot05 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow5), index )).s##suffix), brow, dot05 );         dot06 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow6), index )).s##suffix), brow, dot06 );         dot07 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow7), index )).s##suffix), brow, dot07 );",    // NOLINT
"MM_DOT_PRODUCT(0, 0);",    // NOLINT
"MM_DOT_PRODUCT(0, 1);",    // NOLINT
"MM_DOT_PRODUCT(1, 0);",    // NOLINT
"MM_DOT_PRODUCT(1, 1);",    // NOLINT
"MM_DOT_PRODUCT(2, 0);",    // NOLINT
"MM_DOT_PRODUCT(2, 1);",    // NOLINT
"MM_DOT_PRODUCT(3, 0);",    // NOLINT
"MM_DOT_PRODUCT(3, 1);",    // NOLINT
"MM_DOT_PRODUCT(4, 0);",    // NOLINT
"MM_DOT_PRODUCT(4, 1);",    // NOLINT
"MM_DOT_PRODUCT(5, 0);",    // NOLINT
"MM_DOT_PRODUCT(5, 1);",    // NOLINT
"MM_DOT_PRODUCT(6, 0);",    // NOLINT
"MM_DOT_PRODUCT(6, 1);",    // NOLINT
"MM_DOT_PRODUCT(7, 0);",    // NOLINT
"MM_DOT_PRODUCT(7, 1);",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"MM_DOT_PRODUCT(8, 0);",    // NOLINT
"MM_DOT_PRODUCT(8, 1);",    // NOLINT
"MM_DOT_PRODUCT(9, 0);",    // NOLINT
"MM_DOT_PRODUCT(9, 1);",    // NOLINT
"MM_DOT_PRODUCT(10, 0);",    // NOLINT
"MM_DOT_PRODUCT(10, 1);",    // NOLINT
"MM_DOT_PRODUCT(11, 0);",    // NOLINT
"MM_DOT_PRODUCT(11, 1);",    // NOLINT
"MM_DOT_PRODUCT(12, 0);",    // NOLINT
"MM_DOT_PRODUCT(12, 1);",    // NOLINT
"MM_DOT_PRODUCT(13, 0);",    // NOLINT
"MM_DOT_PRODUCT(13, 1);",    // NOLINT
"MM_DOT_PRODUCT(14, 0);",    // NOLINT
"MM_DOT_PRODUCT(14, 1);",    // NOLINT
"MM_DOT_PRODUCT(15, 0);",    // NOLINT
"MM_DOT_PRODUCT(15, 1);",    // NOLINT
"#endif",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"",    // NOLINT
"src0_read += TILE_K;",    // NOLINT
"w += TILE_K;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(w < end_index) {",    // NOLINT
"arow0.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row0 * K)[0] : 0.0f;",    // NOLINT
"arow0.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row0 * K)[1] : 0.0f;",    // NOLINT
"arow1.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row1 * K)[0] : 0.0f;",    // NOLINT
"arow1.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row1 * K)[1] : 0.0f;",    // NOLINT
"arow2.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row2 * K)[0] : 0.0f;",    // NOLINT
"arow2.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row2 * K)[1] : 0.0f;",    // NOLINT
"arow3.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row3 * K)[0] : 0.0f;",    // NOLINT
"arow3.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row3 * K)[1] : 0.0f;",    // NOLINT
"arow4.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row4 * K)[0] : 0.0f;",    // NOLINT
"arow4.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row4 * K)[1] : 0.0f;",    // NOLINT
"arow5.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row5 * K)[0] : 0.0f;",    // NOLINT
"arow5.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row5 * K)[1] : 0.0f;",    // NOLINT
"arow6.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row6 * K)[0] : 0.0f;",    // NOLINT
"arow6.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row6 * K)[1] : 0.0f;",    // NOLINT
"arow7.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row7 * K)[0] : 0.0f;",    // NOLINT
"arow7.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row7 * K)[1] : 0.0f;",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( index, suffix )           brow = (w < K) ? vload4(0, src1_read0) : (Dtype4)0.0f;  src1_read0 += N; w++;         dot00 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow0), index )).s##suffix), brow, dot00 );         dot01 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow1), index )).s##suffix), brow, dot01 );         dot02 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow2), index )).s##suffix), brow, dot02 );         dot03 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow3), index )).s##suffix), brow, dot03 );         dot04 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow4), index )).s##suffix), brow, dot04 );         dot05 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow5), index )).s##suffix), brow, dot05 );         dot06 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow6), index )).s##suffix), brow, dot06 );         dot07 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow7), index )).s##suffix), brow, dot07 );",    // NOLINT
"MM_DOT_PRODUCT(0, 0);",    // NOLINT
"MM_DOT_PRODUCT(0, 1);",    // NOLINT
"MM_DOT_PRODUCT(1, 0);",    // NOLINT
"MM_DOT_PRODUCT(1, 1);",    // NOLINT
"MM_DOT_PRODUCT(2, 0);",    // NOLINT
"MM_DOT_PRODUCT(2, 1);",    // NOLINT
"MM_DOT_PRODUCT(3, 0);",    // NOLINT
"MM_DOT_PRODUCT(3, 1);",    // NOLINT
"MM_DOT_PRODUCT(4, 0);",    // NOLINT
"MM_DOT_PRODUCT(4, 1);",    // NOLINT
"MM_DOT_PRODUCT(5, 0);",    // NOLINT
"MM_DOT_PRODUCT(5, 1);",    // NOLINT
"MM_DOT_PRODUCT(6, 0);",    // NOLINT
"MM_DOT_PRODUCT(6, 1);",    // NOLINT
"MM_DOT_PRODUCT(7, 0);",    // NOLINT
"MM_DOT_PRODUCT(7, 1);",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"MM_DOT_PRODUCT(8, 0);",    // NOLINT
"MM_DOT_PRODUCT(8, 1);",    // NOLINT
"MM_DOT_PRODUCT(9, 0);",    // NOLINT
"MM_DOT_PRODUCT(9, 1);",    // NOLINT
"MM_DOT_PRODUCT(10, 0);",    // NOLINT
"MM_DOT_PRODUCT(10, 1);",    // NOLINT
"MM_DOT_PRODUCT(11, 0);",    // NOLINT
"MM_DOT_PRODUCT(11, 1);",    // NOLINT
"MM_DOT_PRODUCT(12, 0);",    // NOLINT
"MM_DOT_PRODUCT(12, 1);",    // NOLINT
"MM_DOT_PRODUCT(13, 0);",    // NOLINT
"MM_DOT_PRODUCT(13, 1);",    // NOLINT
"MM_DOT_PRODUCT(14, 0);",    // NOLINT
"MM_DOT_PRODUCT(14, 1);",    // NOLINT
"MM_DOT_PRODUCT(15, 0);",    // NOLINT
"MM_DOT_PRODUCT(15, 1);",    // NOLINT
"#endif",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(global_x * 4 < N && global_y * 8 < M) {",    // NOLINT
"if(mad24(global_x, 4, 3) < N) {",    // NOLINT
"vstore4(dot00, 0, dst_write0); dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }",    // NOLINT
"} else if(mad24(global_x, 4, 2) < N) {",    // NOLINT
"vstore2(dot00.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot00.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) {",    // NOLINT
"vstore2(dot01.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot01.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) {",    // NOLINT
"vstore2(dot02.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot02.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) {",    // NOLINT
"vstore2(dot03.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot03.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) {",    // NOLINT
"vstore2(dot04.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot04.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) {",    // NOLINT
"vstore2(dot05.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot05.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) {",    // NOLINT
"vstore2(dot06.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot06.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) {",    // NOLINT
"vstore2(dot07.xy, 0, dst_write0);",    // NOLINT
"dst_write0[2] = dot07.z;",    // NOLINT
"}",    // NOLINT
"} else if(mad24(global_x, 4, 1) < N) {",    // NOLINT
"vstore2(dot00.xy, 0, dst_write0); dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }",    // NOLINT
"} else {",    // NOLINT
"dst_write0[0] = dot00.x; dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; }",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#undef VEC_SIZE",    // NOLINT
"#undef LWG_HEIGHT",    // NOLINT
"#undef TILE_M",    // NOLINT
"#undef TILE_K",    // NOLINT
"#undef TILE_N",    // NOLINT
"",    // NOLINT
"#define VEC_SIZE        1",    // NOLINT
"#define TILE_M          8",    // NOLINT
"#define TILE_N          8",    // NOLINT
"#define SLM_BLOCK       128",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define LWG_HEIGHT      2",    // NOLINT
"#define TILE_K          64",    // NOLINT
"#else",    // NOLINT
"#define LWG_HEIGHT      4",    // NOLINT
"#define TILE_K          32",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(8)))",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_NT, Dtype)(",    // NOLINT
"const __global Dtype *src0, int off0,",    // NOLINT
"const __global Dtype *src1, int off1,",    // NOLINT
"__global Dtype *dst, int offd,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_in,",    // NOLINT
"KERNEL_ARG_DTYPE beta_in)",    // NOLINT
"{",    // NOLINT
"const Dtype alpha = (Dtype)alpha_in;",    // NOLINT
"const Dtype beta = (Dtype)beta_in;",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int local_x = get_local_id(0);",    // NOLINT
"const int local_y = get_local_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"",    // NOLINT
"Dtype8 dot00 = 0.f;",    // NOLINT
"Dtype8 dot01 = 0.f;",    // NOLINT
"Dtype8 dot02 = 0.f;",    // NOLINT
"Dtype8 dot03 = 0.f;",    // NOLINT
"Dtype8 dot04 = 0.f;",    // NOLINT
"Dtype8 dot05 = 0.f;",    // NOLINT
"Dtype8 dot06 = 0.f;",    // NOLINT
"Dtype8 dot07 = 0.f;",    // NOLINT
"",    // NOLINT
"Dtype8 brow0;",    // NOLINT
"Dtype8 brow1;",    // NOLINT
"Dtype8 brow2;",    // NOLINT
"Dtype8 brow3;",    // NOLINT
"Dtype8 brow4;",    // NOLINT
"Dtype8 brow5;",    // NOLINT
"Dtype8 brow6;",    // NOLINT
"Dtype8 brow7;",    // NOLINT
"",    // NOLINT
"__global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src0_read = src0 + local_x * (TILE_K / 8) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * K + off0;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src1_read0 = src1 + (group_x * TILE_N) * K + off1;",    // NOLINT
"",    // NOLINT
"__local Dtype slm_brow[8 * SLM_BLOCK];",    // NOLINT
"__local Dtype* slm_brow0;",    // NOLINT
"",    // NOLINT
"int local_index = mad24(local_y, 8, local_x) * 8;",    // NOLINT
"int w;",    // NOLINT
"for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(0, K, local_index))), 0, (__local float *)(slm_brow + mad24(0, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(1, K, local_index))), 0, (__local float *)(slm_brow + mad24(1, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(2, K, local_index))), 0, (__local float *)(slm_brow + mad24(2, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(3, K, local_index))), 0, (__local float *)(slm_brow + mad24(3, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(4, K, local_index))), 0, (__local float *)(slm_brow + mad24(4, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(5, K, local_index))), 0, (__local float *)(slm_brow + mad24(5, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(6, K, local_index))), 0, (__local float *)(slm_brow + mad24(6, SLM_BLOCK, local_index)));",    // NOLINT
"vstore4(vload4(0, (__global float *)(src1_read0 + mad24(7, K, local_index))), 0, (__local float *)(slm_brow + mad24(7, SLM_BLOCK, local_index)));",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"slm_brow0 = slm_brow + local_x * (TILE_K / 8);",    // NOLINT
"w = b_tile;",    // NOLINT
"int end_w = min(b_tile + SLM_BLOCK, K);",    // NOLINT
"while( w + TILE_K <= end_w ) {",    // NOLINT
"Dtype8 arow;",    // NOLINT
"",    // NOLINT
"brow0 = as_half8(vload4(0, (__local float *)(slm_brow0 + 0 * SLM_BLOCK)));",    // NOLINT
"brow1 = as_half8(vload4(0, (__local float *)(slm_brow0 + 1 * SLM_BLOCK)));",    // NOLINT
"brow2 = as_half8(vload4(0, (__local float *)(slm_brow0 + 2 * SLM_BLOCK)));",    // NOLINT
"brow3 = as_half8(vload4(0, (__local float *)(slm_brow0 + 3 * SLM_BLOCK)));",    // NOLINT
"brow4 = as_half8(vload4(0, (__local float *)(slm_brow0 + 4 * SLM_BLOCK)));",    // NOLINT
"brow5 = as_half8(vload4(0, (__local float *)(slm_brow0 + 5 * SLM_BLOCK)));",    // NOLINT
"brow6 = as_half8(vload4(0, (__local float *)(slm_brow0 + 6 * SLM_BLOCK)));",    // NOLINT
"brow7 = as_half8(vload4(0, (__local float *)(slm_brow0 + 7 * SLM_BLOCK)));",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _row, _dot )               arow = as_half8(vload4(0, (__global float *)(src0_read + _row * K)));                                       _dot = mad( (Dtype8)(arow.s0), (Dtype8)(brow0.s0, brow1.s0, brow2.s0, brow3.s0, brow4.s0, brow5.s0, brow6.s0, brow7.s0), _dot );             _dot = mad( (Dtype8)(arow.s1), (Dtype8)(brow0.s1, brow1.s1, brow2.s1, brow3.s1, brow4.s1, brow5.s1, brow6.s1, brow7.s1), _dot );             _dot = mad( (Dtype8)(arow.s2), (Dtype8)(brow0.s2, brow1.s2, brow2.s2, brow3.s2, brow4.s2, brow5.s2, brow6.s2, brow7.s2), _dot );             _dot = mad( (Dtype8)(arow.s3), (Dtype8)(brow0.s3, brow1.s3, brow2.s3, brow3.s3, brow4.s3, brow5.s3, brow6.s3, brow7.s3), _dot );             _dot = mad( (Dtype8)(arow.s4), (Dtype8)(brow0.s4, brow1.s4, brow2.s4, brow3.s4, brow4.s4, brow5.s4, brow6.s4, brow7.s4), _dot );             _dot = mad( (Dtype8)(arow.s5), (Dtype8)(brow0.s5, brow1.s5, brow2.s5, brow3.s5, brow4.s5, brow5.s5, brow6.s5, brow7.s5), _dot );             _dot = mad( (Dtype8)(arow.s6), (Dtype8)(brow0.s6, brow1.s6, brow2.s6, brow3.s6, brow4.s6, brow5.s6, brow6.s6, brow7.s6), _dot );             _dot = mad( (Dtype8)(arow.s7), (Dtype8)(brow0.s7, brow1.s7, brow2.s7, brow3.s7, brow4.s7, brow5.s7, brow6.s7, brow7.s7), _dot );",    // NOLINT
"MM_DOT_PRODUCT( 0, dot00 );",    // NOLINT
"MM_DOT_PRODUCT( 1, dot01 );",    // NOLINT
"MM_DOT_PRODUCT( 2, dot02 );",    // NOLINT
"MM_DOT_PRODUCT( 3, dot03 );",    // NOLINT
"MM_DOT_PRODUCT( 4, dot04 );",    // NOLINT
"MM_DOT_PRODUCT( 5, dot05 );",    // NOLINT
"MM_DOT_PRODUCT( 6, dot06 );",    // NOLINT
"MM_DOT_PRODUCT( 7, dot07 );",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"",    // NOLINT
"src0_read += TILE_K;",    // NOLINT
"slm_brow0 += TILE_K;",    // NOLINT
"w += TILE_K;",    // NOLINT
"}",    // NOLINT
"src1_read0 += SLM_BLOCK;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(w < K) {",    // NOLINT
"Dtype8 arow;",    // NOLINT
"",    // NOLINT
"#define READ_BROW(_brow, _row)         _brow = as_half8(vload4(0, (__local float *)(slm_brow0 + _row * SLM_BLOCK)));         _brow.s0 = (mad24(local_x, 8, w) < K) ? _brow.s0 : 0.0f;         _brow.s1 = (mad24(local_x, 8, w + 1) < K) ? _brow.s1 : 0.0f;         _brow.s2 = (mad24(local_x, 8, w + 2) < K) ? _brow.s2 : 0.0f;         _brow.s3 = (mad24(local_x, 8, w + 3) < K) ? _brow.s3 : 0.0f;         _brow.s4 = (mad24(local_x, 8, w + 4) < K) ? _brow.s4 : 0.0f;         _brow.s5 = (mad24(local_x, 8, w + 5) < K) ? _brow.s5 : 0.0f;         _brow.s6 = (mad24(local_x, 8, w + 6) < K) ? _brow.s6 : 0.0f;         _brow.s7 = (mad24(local_x, 8, w + 7) < K) ? _brow.s7 : 0.0f;",    // NOLINT
"READ_BROW(brow0, 0);",    // NOLINT
"READ_BROW(brow1, 1);",    // NOLINT
"READ_BROW(brow2, 2);",    // NOLINT
"READ_BROW(brow3, 3);",    // NOLINT
"READ_BROW(brow4, 4);",    // NOLINT
"READ_BROW(brow5, 5);",    // NOLINT
"READ_BROW(brow6, 6);",    // NOLINT
"READ_BROW(brow7, 7);",    // NOLINT
"",    // NOLINT
"#undef READ_BROW",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _row, _dot )           arow = as_half8(vload4(0, (__global float *)(src0_read + _row * K)));                                   arow.s0 = (mad24(local_x, 8, w) < K) ? arow.s0 : 0.0f;         arow.s1 = (mad24(local_x, 8, w + 1) < K) ? arow.s1 : 0.0f;         arow.s2 = (mad24(local_x, 8, w + 2) < K) ? arow.s2 : 0.0f;         arow.s3 = (mad24(local_x, 8, w + 3) < K) ? arow.s3 : 0.0f;         arow.s4 = (mad24(local_x, 8, w + 4) < K) ? arow.s4 : 0.0f;         arow.s5 = (mad24(local_x, 8, w + 5) < K) ? arow.s5 : 0.0f;         arow.s6 = (mad24(local_x, 8, w + 6) < K) ? arow.s6 : 0.0f;         arow.s7 = (mad24(local_x, 8, w + 7) < K) ? arow.s7 : 0.0f;         _dot = mad( (Dtype8)(arow.s0), (Dtype8)(brow0.s0, brow1.s0, brow2.s0, brow3.s0, brow4.s0, brow5.s0, brow6.s0, brow7.s0), _dot );         _dot = mad( (Dtype8)(arow.s1), (Dtype8)(brow0.s1, brow1.s1, brow2.s1, brow3.s1, brow4.s1, brow5.s1, brow6.s1, brow7.s1), _dot );         _dot = mad( (Dtype8)(arow.s2), (Dtype8)(brow0.s2, brow1.s2, brow2.s2, brow3.s2, brow4.s2, brow5.s2, brow6.s2, brow7.s2), _dot );         _dot = mad( (Dtype8)(arow.s3), (Dtype8)(brow0.s3, brow1.s3, brow2.s3, brow3.s3, brow4.s3, brow5.s3, brow6.s3, brow7.s3), _dot );         _dot = mad( (Dtype8)(arow.s4), (Dtype8)(brow0.s4, brow1.s4, brow2.s4, brow3.s4, brow4.s4, brow5.s4, brow6.s4, brow7.s4), _dot );         _dot = mad( (Dtype8)(arow.s5), (Dtype8)(brow0.s5, brow1.s5, brow2.s5, brow3.s5, brow4.s5, brow5.s5, brow6.s5, brow7.s5), _dot );         _dot = mad( (Dtype8)(arow.s6), (Dtype8)(brow0.s6, brow1.s6, brow2.s6, brow3.s6, brow4.s6, brow5.s6, brow6.s6, brow7.s6), _dot );         _dot = mad( (Dtype8)(arow.s7), (Dtype8)(brow0.s7, brow1.s7, brow2.s7, brow3.s7, brow4.s7, brow5.s7, brow6.s7, brow7.s7), _dot );",    // NOLINT
"MM_DOT_PRODUCT( 0, dot00 );",    // NOLINT
"MM_DOT_PRODUCT( 1, dot01 );",    // NOLINT
"MM_DOT_PRODUCT( 2, dot02 );",    // NOLINT
"MM_DOT_PRODUCT( 3, dot03 );",    // NOLINT
"MM_DOT_PRODUCT( 4, dot04 );",    // NOLINT
"MM_DOT_PRODUCT( 5, dot05 );",    // NOLINT
"MM_DOT_PRODUCT( 6, dot06 );",    // NOLINT
"MM_DOT_PRODUCT( 7, dot07 );",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#define REDUCE(_dot)     _dot = as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 0)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 1)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 2)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 3)) +             as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 4)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 5)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 6)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 7));",    // NOLINT
"REDUCE(dot00);",    // NOLINT
"REDUCE(dot01);",    // NOLINT
"REDUCE(dot02);",    // NOLINT
"REDUCE(dot03);",    // NOLINT
"REDUCE(dot04);",    // NOLINT
"REDUCE(dot05);",    // NOLINT
"REDUCE(dot06);",    // NOLINT
"REDUCE(dot07);",    // NOLINT
"#undef REDUCE",    // NOLINT
"",    // NOLINT
"Dtype output = 0.0f;",    // NOLINT
"#define OUTPUT( _dot)     output = (local_x == 0) ? _dot.s0 : output;     output = (local_x == 1) ? _dot.s1 : output;     output = (local_x == 2) ? _dot.s2 : output;     output = (local_x == 3) ? _dot.s3 : output;     output = (local_x == 4) ? _dot.s4 : output;     output = (local_x == 5) ? _dot.s5 : output;     output = (local_x == 6) ? _dot.s6 : output;     output = (local_x == 7) ? _dot.s7 : output;     dst_write0[0] = mad(output, alpha, beta * dst_write0[0]);     dst_write0 += N;",    // NOLINT
"",    // NOLINT
"if(global_x < N && global_y * 8 < M) {",    // NOLINT
"OUTPUT(dot00);",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { OUTPUT(dot01); }",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { OUTPUT(dot02); }",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { OUTPUT(dot03); }",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { OUTPUT(dot04); }",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { OUTPUT(dot05); }",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { OUTPUT(dot06); }",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { OUTPUT(dot07); }",    // NOLINT
"}",    // NOLINT
"#undef OUTPUT",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#else",    // NOLINT
"",    // NOLINT
"__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(8)))",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_NT, Dtype)(",    // NOLINT
"const __global Dtype *src0, int off0,",    // NOLINT
"const __global Dtype *src1, int off1,",    // NOLINT
"__global Dtype *dst, int offd,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_in,",    // NOLINT
"KERNEL_ARG_DTYPE beta_in)",    // NOLINT
"{",    // NOLINT
"const Dtype alpha = (Dtype)alpha_in;",    // NOLINT
"const Dtype beta = (Dtype)beta_in;",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int local_x = get_local_id(0);",    // NOLINT
"const int local_y = get_local_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"",    // NOLINT
"Dtype8 dot00 = 0.f;",    // NOLINT
"Dtype8 dot01 = 0.f;",    // NOLINT
"Dtype8 dot02 = 0.f;",    // NOLINT
"Dtype8 dot03 = 0.f;",    // NOLINT
"Dtype8 dot04 = 0.f;",    // NOLINT
"Dtype8 dot05 = 0.f;",    // NOLINT
"Dtype8 dot06 = 0.f;",    // NOLINT
"Dtype8 dot07 = 0.f;",    // NOLINT
"",    // NOLINT
"Dtype4 brow0;",    // NOLINT
"Dtype4 brow1;",    // NOLINT
"Dtype4 brow2;",    // NOLINT
"Dtype4 brow3;",    // NOLINT
"Dtype4 brow4;",    // NOLINT
"Dtype4 brow5;",    // NOLINT
"Dtype4 brow6;",    // NOLINT
"Dtype4 brow7;",    // NOLINT
"",    // NOLINT
"__global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src0_read = src0 + local_x * (TILE_K / 8) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * K + off0;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src1_read0 = src1 + (group_x * TILE_N) * K + off1;",    // NOLINT
"",    // NOLINT
"__local Dtype slm_brow[8 * SLM_BLOCK];",    // NOLINT
"__local Dtype* slm_brow0;",    // NOLINT
"",    // NOLINT
"int local_index = mad24(local_y, 8, local_x) * 4;",    // NOLINT
"int w;",    // NOLINT
"for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(0, K, local_index)), 0, slm_brow + mad24(0, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(1, K, local_index)), 0, slm_brow + mad24(1, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(2, K, local_index)), 0, slm_brow + mad24(2, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(3, K, local_index)), 0, slm_brow + mad24(3, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(4, K, local_index)), 0, slm_brow + mad24(4, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(5, K, local_index)), 0, slm_brow + mad24(5, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(6, K, local_index)), 0, slm_brow + mad24(6, SLM_BLOCK, local_index));",    // NOLINT
"vstore4(vload4(0, src1_read0 + mad24(7, K, local_index)), 0, slm_brow + mad24(7, SLM_BLOCK, local_index));",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"",    // NOLINT
"slm_brow0 = slm_brow + local_x * (TILE_K / 8);",    // NOLINT
"w = b_tile;",    // NOLINT
"int end_w = min(b_tile + SLM_BLOCK, K);",    // NOLINT
"while( w + TILE_K <= end_w ) {",    // NOLINT
"Dtype4 arow;",    // NOLINT
"",    // NOLINT
"brow0 = vload4(0, slm_brow0 + 0 * SLM_BLOCK);",    // NOLINT
"brow1 = vload4(0, slm_brow0 + 1 * SLM_BLOCK);",    // NOLINT
"brow2 = vload4(0, slm_brow0 + 2 * SLM_BLOCK);",    // NOLINT
"brow3 = vload4(0, slm_brow0 + 3 * SLM_BLOCK);",    // NOLINT
"brow4 = vload4(0, slm_brow0 + 4 * SLM_BLOCK);",    // NOLINT
"brow5 = vload4(0, slm_brow0 + 5 * SLM_BLOCK);",    // NOLINT
"brow6 = vload4(0, slm_brow0 + 6 * SLM_BLOCK);",    // NOLINT
"brow7 = vload4(0, slm_brow0 + 7 * SLM_BLOCK);",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _row, _dot )               arow = vload4(0, src0_read + _row * K);                                       _dot = mad( (Dtype8)(arow.x), (Dtype8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot );             _dot = mad( (Dtype8)(arow.y), (Dtype8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot );             _dot = mad( (Dtype8)(arow.z), (Dtype8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot );             _dot = mad( (Dtype8)(arow.w), (Dtype8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );",    // NOLINT
"MM_DOT_PRODUCT( 0, dot00 );",    // NOLINT
"MM_DOT_PRODUCT( 1, dot01 );",    // NOLINT
"MM_DOT_PRODUCT( 2, dot02 );",    // NOLINT
"MM_DOT_PRODUCT( 3, dot03 );",    // NOLINT
"MM_DOT_PRODUCT( 4, dot04 );",    // NOLINT
"MM_DOT_PRODUCT( 5, dot05 );",    // NOLINT
"MM_DOT_PRODUCT( 6, dot06 );",    // NOLINT
"MM_DOT_PRODUCT( 7, dot07 );",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"",    // NOLINT
"src0_read += TILE_K;",    // NOLINT
"slm_brow0 += TILE_K;",    // NOLINT
"w += TILE_K;",    // NOLINT
"}",    // NOLINT
"src1_read0 += SLM_BLOCK;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(w < K) {",    // NOLINT
"Dtype4 arow;",    // NOLINT
"",    // NOLINT
"#define READ_BROW(_brow, _row)         _brow = vload4(0, slm_brow0 + _row * SLM_BLOCK);         _brow.x = (mad24(local_x, 4, w) < K) ? _brow.x : 0.0f;         _brow.y = (mad24(local_x, 4, w + 1) < K) ? _brow.y : 0.0f;         _brow.z = (mad24(local_x, 4, w + 2) < K) ? _brow.z : 0.0f;         _brow.w = (mad24(local_x, 4, w + 3) < K) ? _brow.w : 0.0f;",    // NOLINT
"READ_BROW(brow0, 0);",    // NOLINT
"READ_BROW(brow1, 1);",    // NOLINT
"READ_BROW(brow2, 2);",    // NOLINT
"READ_BROW(brow3, 3);",    // NOLINT
"READ_BROW(brow4, 4);",    // NOLINT
"READ_BROW(brow5, 5);",    // NOLINT
"READ_BROW(brow6, 6);",    // NOLINT
"READ_BROW(brow7, 7);",    // NOLINT
"",    // NOLINT
"#undef READ_BROW",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _row, _dot )           arow = vload4(0, src0_read + _row * K);                                   arow.x = (mad24(local_x, 4, w) < K) ? arow.x : 0.0f;         arow.y = (mad24(local_x, 4, w + 1) < K) ? arow.y : 0.0f;         arow.z = (mad24(local_x, 4, w + 2) < K) ? arow.z : 0.0f;         arow.w = (mad24(local_x, 4, w + 3) < K) ? arow.w : 0.0f;         _dot = mad( (Dtype8)(arow.x), (Dtype8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot );         _dot = mad( (Dtype8)(arow.y), (Dtype8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot );         _dot = mad( (Dtype8)(arow.z), (Dtype8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot );         _dot = mad( (Dtype8)(arow.w), (Dtype8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );",    // NOLINT
"MM_DOT_PRODUCT( 0, dot00 );",    // NOLINT
"MM_DOT_PRODUCT( 1, dot01 );",    // NOLINT
"MM_DOT_PRODUCT( 2, dot02 );",    // NOLINT
"MM_DOT_PRODUCT( 3, dot03 );",    // NOLINT
"MM_DOT_PRODUCT( 4, dot04 );",    // NOLINT
"MM_DOT_PRODUCT( 5, dot05 );",    // NOLINT
"MM_DOT_PRODUCT( 6, dot06 );",    // NOLINT
"MM_DOT_PRODUCT( 7, dot07 );",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#define REDUCE(_dot)     _dot = as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 0)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 1)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 2)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 3)) +             as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 4)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 5)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 6)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 7));",    // NOLINT
"REDUCE(dot00);",    // NOLINT
"REDUCE(dot01);",    // NOLINT
"REDUCE(dot02);",    // NOLINT
"REDUCE(dot03);",    // NOLINT
"REDUCE(dot04);",    // NOLINT
"REDUCE(dot05);",    // NOLINT
"REDUCE(dot06);",    // NOLINT
"REDUCE(dot07);",    // NOLINT
"#undef REDUCE",    // NOLINT
"",    // NOLINT
"Dtype output = 0.0f;",    // NOLINT
"#define OUTPUT( _dot)     output = (local_x == 0) ? _dot.s0 : output;     output = (local_x == 1) ? _dot.s1 : output;     output = (local_x == 2) ? _dot.s2 : output;     output = (local_x == 3) ? _dot.s3 : output;     output = (local_x == 4) ? _dot.s4 : output;     output = (local_x == 5) ? _dot.s5 : output;     output = (local_x == 6) ? _dot.s6 : output;     output = (local_x == 7) ? _dot.s7 : output;     dst_write0[0] = mad(output, alpha, beta * dst_write0[0]);     dst_write0 += N;",    // NOLINT
"",    // NOLINT
"if(global_x < N && global_y * 8 < M) {",    // NOLINT
"OUTPUT(dot00);",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { OUTPUT(dot01); }",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { OUTPUT(dot02); }",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { OUTPUT(dot03); }",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { OUTPUT(dot04); }",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { OUTPUT(dot05); }",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { OUTPUT(dot06); }",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { OUTPUT(dot07); }",    // NOLINT
"}",    // NOLINT
"#undef OUTPUT",    // NOLINT
"}",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#undef VEC_SIZE",    // NOLINT
"#undef LWG_HEIGHT",    // NOLINT
"#undef TILE_M",    // NOLINT
"#undef TILE_K",    // NOLINT
"#undef TILE_N",    // NOLINT
"#undef SLM_BLOCK",    // NOLINT
"",    // NOLINT
"#define SLM_SIZE 64",    // NOLINT
"void TEMPLATE(gemm_buffer_NT_M_2_edgerows,Dtype)(",    // NOLINT
"const __global Dtype* srca_read0,",    // NOLINT
"const __global Dtype* srca_read1,",    // NOLINT
"const __global Dtype* srcb_read,",    // NOLINT
"__local Dtype4* work0,",    // NOLINT
"__local Dtype4* work1,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"int x_gid,",    // NOLINT
"int lid,",    // NOLINT
"Dtype alpha,",    // NOLINT
"Dtype beta,",    // NOLINT
"__global Dtype* dstc0,",    // NOLINT
"__global Dtype* dstc1)",    // NOLINT
"{",    // NOLINT
"__local Dtype* work_each0 = (__local Dtype*)work0;",    // NOLINT
"__local Dtype* work_each1 = (__local Dtype*)work1;",    // NOLINT
"",    // NOLINT
"int rows = N - x_gid * 4;",    // NOLINT
"",    // NOLINT
"Dtype4 dot0[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot1[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"",    // NOLINT
"int i = lid;",    // NOLINT
"while( i < K / 4) {",    // NOLINT
"const Dtype4 b0 = {srca_read0[i*4], srca_read0[(i*4+1)], srca_read0[(i*4+2)], srca_read0[(i*4+3)]};",    // NOLINT
"const Dtype4 b1 = {srca_read1[i*4], srca_read1[(i*4+1)], srca_read1[(i*4+2)], srca_read1[(i*4+3)]};",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"dot0[j] += b0 * vload4(i, srcb_read + j * K);",    // NOLINT
"dot1[j] += b1 * vload4(i, srcb_read + j * K);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"i += get_local_size(0);",    // NOLINT
"}",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"work_each0[lid * 4 + j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;",    // NOLINT
"work_each1[lid * 4 + j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(i == K / 4) {",    // NOLINT
"short tail_items = K % 4;",    // NOLINT
"",    // NOLINT
"if(tail_items != 0) {",    // NOLINT
"const __global Dtype *srcb_tail = srcb_read + i * 4;",    // NOLINT
"const __global Dtype *srca_tail0 = srca_read0 + i * 4;",    // NOLINT
"const __global Dtype *srca_tail1 = srca_read1 + i * 4;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short i = 0; i < tail_items; ++i) {",    // NOLINT
"const Dtype at0 = srca_tail0[i];",    // NOLINT
"const Dtype at1 = srca_tail1[i];",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"work_each0[lid * 4 + j] += at0 * srcb_tail[i + j * K];",    // NOLINT
"work_each1[lid * 4 + j] += at1 * srcb_tail[i + j * K];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride) {",    // NOLINT
"work0[lid] += work0[lid+stride];",    // NOLINT
"work1[lid] += work1[lid+stride];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"dstc0[(x_gid * 4  + j)] = alpha * work_each0[j] + beta * dstc0[(x_gid * 4 + j)];",    // NOLINT
"dstc1[(x_gid * 4  + j)] = alpha * work_each1[j] + beta * dstc1[(x_gid * 4 + j)];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_NT_M_2,Dtype)(",    // NOLINT
"__global const Dtype * A,",    // NOLINT
"int offA,",    // NOLINT
"__global const Dtype * B,",    // NOLINT
"int offB,",    // NOLINT
"__global Dtype * C,",    // NOLINT
"int offC,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_f,",    // NOLINT
"KERNEL_ARG_DTYPE beta_f)",    // NOLINT
"{",    // NOLINT
"Dtype alpha = (Dtype)alpha_f;",    // NOLINT
"Dtype beta = (Dtype)beta_f;",    // NOLINT
"int x_gid = get_group_id(0);",    // NOLINT
"int lid = get_local_id(0);",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_read0 = A + offA;",    // NOLINT
"const __global Dtype *srca_read1 = srca_read0 + K;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srcb_read = B + x_gid * 4 * K + offB;",    // NOLINT
"",    // NOLINT
"__global Dtype4 *dstc0 = (__global Dtype4*)(C + offC);",    // NOLINT
"__global Dtype4 *dstc1 = (__global Dtype4*)((__global Dtype*)(dstc0) + N);",    // NOLINT
"",    // NOLINT
"__local Dtype4 work0[SLM_SIZE];",    // NOLINT
"__local Dtype4 work1[SLM_SIZE];",    // NOLINT
"__local Dtype* work_each0 = (__local Dtype*)work0;",    // NOLINT
"__local Dtype* work_each1 = (__local Dtype*)work1;",    // NOLINT
"",    // NOLINT
"if(x_gid == N / 4) {",    // NOLINT
"TEMPLATE(gemm_buffer_NT_M_2_edgerows,Dtype)          (srca_read0, srca_read1, srcb_read, work0, work1, N, K, x_gid, lid, alpha, beta, (__global Dtype*)dstc0, (__global Dtype*)dstc1);",    // NOLINT
"} else {",    // NOLINT
"Dtype4 dot0[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot1[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"int i = lid;",    // NOLINT
"while( i < K / 4) {",    // NOLINT
"const Dtype4 b0 = vload4(i, srca_read0);",    // NOLINT
"const Dtype4 b1 = vload4(i, srca_read1);",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"Dtype4 a = vload4(i, srcb_read + j * K);",    // NOLINT
"dot0[j] += b0 * a;",    // NOLINT
"dot1[j] += b1 * a;",    // NOLINT
"}",    // NOLINT
"i += get_local_size(0);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"work_each0[lid * 4 + j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;",    // NOLINT
"work_each1[lid * 4 + j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(i == K / 4) {",    // NOLINT
"short tail_items = K % 4;",    // NOLINT
"if(tail_items != 0) {",    // NOLINT
"const __global Dtype *srcb_tail = srcb_read + i * 4;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_tail0 = srca_read0 + i * 4;",    // NOLINT
"const __global Dtype *srca_tail1 = srca_read1 + i * 4;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short i = 0; i < tail_items; ++i) {",    // NOLINT
"const Dtype at0 = srca_tail0[i];",    // NOLINT
"const Dtype at1 = srca_tail1[i];",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"work_each0[lid * 4 + j] += at0 * srcb_tail[i + j * K];",    // NOLINT
"work_each1[lid * 4 + j] += at1 * srcb_tail[i + j * K];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride) {",    // NOLINT
"work0[lid] += work0[lid+stride];",    // NOLINT
"work1[lid] += work1[lid+stride];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"dstc0[x_gid] = alpha * work0[0] + beta * dstc0[x_gid];",    // NOLINT
"dstc1[x_gid] = alpha * work1[0] + beta * dstc1[x_gid];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#undef SLM_SIZE",    // NOLINT
"",    // NOLINT
"#define SLM_SIZE 32",    // NOLINT
"void TEMPLATE(gemm_buffer_NT_M_4_edgerows,Dtype)(",    // NOLINT
"const __global Dtype* srca_read0,",    // NOLINT
"const __global Dtype* srca_read1,",    // NOLINT
"const __global Dtype* srca_read2,",    // NOLINT
"const __global Dtype* srca_read3,",    // NOLINT
"const __global Dtype* srcb_read,",    // NOLINT
"__local Dtype4* work0,",    // NOLINT
"__local Dtype4* work1,",    // NOLINT
"__local Dtype4* work2,",    // NOLINT
"__local Dtype4* work3,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"int x_gid,",    // NOLINT
"int lid,",    // NOLINT
"Dtype alpha,",    // NOLINT
"Dtype beta,",    // NOLINT
"__global Dtype* dstc0,",    // NOLINT
"__global Dtype* dstc1,",    // NOLINT
"__global Dtype* dstc2,",    // NOLINT
"__global Dtype* dstc3)",    // NOLINT
"{",    // NOLINT
"__local Dtype* work_each0 = (__local Dtype*)(work0 + lid);",    // NOLINT
"__local Dtype* work_each1 = (__local Dtype*)(work1 + lid);",    // NOLINT
"__local Dtype* work_each2 = (__local Dtype*)(work2 + lid);",    // NOLINT
"__local Dtype* work_each3 = (__local Dtype*)(work3 + lid);",    // NOLINT
"",    // NOLINT
"int rows = N - x_gid * 4;",    // NOLINT
"",    // NOLINT
"Dtype4 dot0[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot1[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot2[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot3[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"",    // NOLINT
"int i = lid;",    // NOLINT
"while( i < K / 4) {",    // NOLINT
"const Dtype4 a0 = {srca_read0[i*4], srca_read0[(i*4+1)], srca_read0[(i*4+2)], srca_read0[(i*4+3)]};",    // NOLINT
"const Dtype4 a1 = {srca_read1[i*4], srca_read1[(i*4+1)], srca_read1[(i*4+2)], srca_read1[(i*4+3)]};",    // NOLINT
"const Dtype4 a2 = {srca_read2[i*4], srca_read2[(i*4+1)], srca_read2[(i*4+2)], srca_read2[(i*4+3)]};",    // NOLINT
"const Dtype4 a3 = {srca_read3[i*4], srca_read3[(i*4+1)], srca_read3[(i*4+2)], srca_read3[(i*4+3)]};",    // NOLINT
"#pragma unrol",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"dot0[j] += a0 * vload4(i, srcb_read + j * K);",    // NOLINT
"dot1[j] += a1 * vload4(i, srcb_read + j * K);",    // NOLINT
"dot2[j] += a2 * vload4(i, srcb_read + j * K);",    // NOLINT
"dot3[j] += a3 * vload4(i, srcb_read + j * K);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"i += get_local_size(0);",    // NOLINT
"}",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"work_each0[j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;",    // NOLINT
"work_each1[j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;",    // NOLINT
"work_each2[j] = dot2[j].x + dot2[j].y + dot2[j].z + dot2[j].w;",    // NOLINT
"work_each3[j] = dot3[j].x + dot3[j].y + dot3[j].z + dot3[j].w;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(i == K / 4) {",    // NOLINT
"short tail_items = K % 4;",    // NOLINT
"",    // NOLINT
"if(tail_items != 0) {",    // NOLINT
"const __global Dtype *srcb_tail = srcb_read + i * 4;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_tail0 = srca_read0 + i * 4;",    // NOLINT
"const __global Dtype *srca_tail1 = srca_read1 + i * 4;",    // NOLINT
"const __global Dtype *srca_tail2 = srca_read2 + i * 4;",    // NOLINT
"const __global Dtype *srca_tail3 = srca_read3 + i * 4;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short i = 0; i < tail_items; ++i) {",    // NOLINT
"const Dtype at0 = srca_tail0[i];",    // NOLINT
"const Dtype at1 = srca_tail1[i];",    // NOLINT
"const Dtype at2 = srca_tail2[i];",    // NOLINT
"const Dtype at3 = srca_tail3[i];",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"work_each0[j] += at0 * srcb_tail[i + j * K];",    // NOLINT
"work_each1[j] += at1 * srcb_tail[i + j * K];",    // NOLINT
"work_each2[j] += at2 * srcb_tail[i + j * K];",    // NOLINT
"work_each3[j] += at3 * srcb_tail[i + j * K];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride) {",    // NOLINT
"work0[lid] += work0[lid+stride];",    // NOLINT
"work1[lid] += work1[lid+stride];",    // NOLINT
"work2[lid] += work2[lid+stride];",    // NOLINT
"work3[lid] += work3[lid+stride];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"dstc0[(x_gid * 4  + j)] = alpha * work_each0[j] + beta * dstc0[(x_gid * 4 + j)];",    // NOLINT
"dstc1[(x_gid * 4  + j)] = alpha * work_each1[j] + beta * dstc1[(x_gid * 4 + j)];",    // NOLINT
"dstc2[(x_gid * 4  + j)] = alpha * work_each2[j] + beta * dstc2[(x_gid * 4 + j)];",    // NOLINT
"dstc3[(x_gid * 4  + j)] = alpha * work_each3[j] + beta * dstc3[(x_gid * 4 + j)];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_NT_M_4,Dtype)(",    // NOLINT
"__global const Dtype * A,",    // NOLINT
"int offA,",    // NOLINT
"__global const Dtype * B,",    // NOLINT
"int offB,",    // NOLINT
"__global Dtype * C,",    // NOLINT
"int offC,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_f,",    // NOLINT
"KERNEL_ARG_DTYPE beta_f)",    // NOLINT
"{",    // NOLINT
"Dtype alpha = (Dtype)alpha_f;",    // NOLINT
"Dtype beta = (Dtype)beta_f;",    // NOLINT
"int x_gid = get_group_id(0);",    // NOLINT
"int lid = get_local_id(0);",    // NOLINT
"int lsize = get_local_size(0);",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_read0 = A + offA;",    // NOLINT
"const __global Dtype *srca_read1 = srca_read0 + K;",    // NOLINT
"const __global Dtype *srca_read2 = srca_read1 + K;",    // NOLINT
"const __global Dtype *srca_read3 = srca_read2 + K;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srcb_read = B + x_gid * 4 * K + offB;",    // NOLINT
"",    // NOLINT
"__global Dtype4 *dstc0 = (__global Dtype4*)(C + offC);",    // NOLINT
"__global Dtype4 *dstc1 = (__global Dtype4*)((__global Dtype*)(dstc0) + N);",    // NOLINT
"__global Dtype4 *dstc2 = (__global Dtype4*)((__global Dtype*)(dstc1) + N);",    // NOLINT
"__global Dtype4 *dstc3 = (__global Dtype4*)((__global Dtype*)(dstc2) + N);",    // NOLINT
"",    // NOLINT
"__local Dtype4 work0[SLM_SIZE];",    // NOLINT
"__local Dtype4 work1[SLM_SIZE];",    // NOLINT
"__local Dtype4 work2[SLM_SIZE];",    // NOLINT
"__local Dtype4 work3[SLM_SIZE];",    // NOLINT
"__local Dtype* work_each0 = (__local Dtype*)(work0 + lid);",    // NOLINT
"__local Dtype* work_each1 = (__local Dtype*)(work1 + lid);",    // NOLINT
"__local Dtype* work_each2 = (__local Dtype*)(work2 + lid);",    // NOLINT
"__local Dtype* work_each3 = (__local Dtype*)(work3 + lid);",    // NOLINT
"",    // NOLINT
"if(x_gid == N / 4) {",    // NOLINT
"TEMPLATE(gemm_buffer_NT_M_4_edgerows,Dtype)          (srca_read0, srca_read1, srca_read2, srca_read3, srcb_read,          work0, work1, work2, work3, N, K, x_gid, lid, alpha, beta,          (__global Dtype*)dstc0, (__global Dtype*)dstc1, (__global Dtype*)dstc2, (__global Dtype*)dstc3);",    // NOLINT
"} else {",    // NOLINT
"Dtype4 dot0[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot1[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot2[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"Dtype4 dot3[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"",    // NOLINT
"int kid = lid;",    // NOLINT
"while( kid < K / 4) {",    // NOLINT
"const Dtype4 b0 = vload4(kid, srca_read0);",    // NOLINT
"const Dtype4 b1 = vload4(kid, srca_read1);",    // NOLINT
"const Dtype4 b2 = vload4(kid, srca_read2);",    // NOLINT
"const Dtype4 b3 = vload4(kid, srca_read3);",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"Dtype4 a = vload4(kid, srcb_read + j * K);",    // NOLINT
"dot0[j] += b0 * a;",    // NOLINT
"dot1[j] += b1 * a;",    // NOLINT
"dot2[j] += b2 * a;",    // NOLINT
"dot3[j] += b3 * a;",    // NOLINT
"}",    // NOLINT
"kid += lsize;",    // NOLINT
"}",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"work_each0[j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;",    // NOLINT
"work_each1[j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;",    // NOLINT
"work_each2[j] = dot2[j].x + dot2[j].y + dot2[j].z + dot2[j].w;",    // NOLINT
"work_each3[j] = dot3[j].x + dot3[j].y + dot3[j].z + dot3[j].w;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(kid == (K >> 2)) {",    // NOLINT
"short tail_items = K % 4;",    // NOLINT
"if(tail_items != 0) {",    // NOLINT
"int offset = kid << 2;",    // NOLINT
"const __global Dtype *srcb_tail = srcb_read + offset;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_tail0 = srca_read0 + offset;",    // NOLINT
"const __global Dtype *srca_tail1 = srca_read1 + offset;",    // NOLINT
"const __global Dtype *srca_tail2 = srca_read2 + offset;",    // NOLINT
"const __global Dtype *srca_tail3 = srca_read3 + offset;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short i = 0; i < tail_items; ++i) {",    // NOLINT
"const Dtype at0 = srca_tail0[i];",    // NOLINT
"const Dtype at1 = srca_tail1[i];",    // NOLINT
"const Dtype at2 = srca_tail2[i];",    // NOLINT
"const Dtype at3 = srca_tail3[i];",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"work_each0[j] += at0 * srcb_tail[i + j * K];",    // NOLINT
"work_each1[j] += at1 * srcb_tail[i + j * K];",    // NOLINT
"work_each2[j] += at2 * srcb_tail[i + j * K];",    // NOLINT
"work_each3[j] += at3 * srcb_tail[i + j * K];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride) {",    // NOLINT
"work0[lid] += work0[lid+stride];",    // NOLINT
"work1[lid] += work1[lid+stride];",    // NOLINT
"work2[lid] += work2[lid+stride];",    // NOLINT
"work3[lid] += work3[lid+stride];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"dstc0[x_gid] = alpha * work0[0] + beta * dstc0[x_gid];",    // NOLINT
"dstc1[x_gid] = alpha * work1[0] + beta * dstc1[x_gid];",    // NOLINT
"dstc2[x_gid] = alpha * work2[0] + beta * dstc2[x_gid];",    // NOLINT
"dstc3[x_gid] = alpha * work3[0] + beta * dstc3[x_gid];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#undef SLM_SIZE",    // NOLINT
"",    // NOLINT
"#define SLM_SIZE 16",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_NT_M_8,Dtype)(",    // NOLINT
"__global const Dtype * A,",    // NOLINT
"int offA,",    // NOLINT
"__global const Dtype * B,",    // NOLINT
"int offB,",    // NOLINT
"__global Dtype * C,",    // NOLINT
"int offC,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_f,",    // NOLINT
"KERNEL_ARG_DTYPE beta_f)",    // NOLINT
"{",    // NOLINT
"Dtype alpha = (Dtype)alpha_f;",    // NOLINT
"Dtype beta = (Dtype)beta_f;",    // NOLINT
"int x_gid = get_group_id(0);",    // NOLINT
"int lid = get_local_id(0);",    // NOLINT
"int lsize = get_local_size(0);",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_read0 = A + offA;",    // NOLINT
"const __global Dtype *srca_read1 = srca_read0 + K;",    // NOLINT
"const __global Dtype *srca_read2 = srca_read1 + K;",    // NOLINT
"const __global Dtype *srca_read3 = srca_read2 + K;",    // NOLINT
"const __global Dtype *srca_read4 = srca_read3 + K;",    // NOLINT
"const __global Dtype *srca_read5 = srca_read4 + K;",    // NOLINT
"const __global Dtype *srca_read6 = srca_read5 + K;",    // NOLINT
"const __global Dtype *srca_read7 = srca_read6 + K;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srcb_read = B + x_gid * K + offB;",    // NOLINT
"",    // NOLINT
"__global Dtype *dstc0 = C + offC;",    // NOLINT
"__global Dtype *dstc1 = dstc0 + N;",    // NOLINT
"__global Dtype *dstc2 = dstc1 + N;",    // NOLINT
"__global Dtype *dstc3 = dstc2 + N;",    // NOLINT
"__global Dtype *dstc4 = dstc3 + N;",    // NOLINT
"__global Dtype *dstc5 = dstc4 + N;",    // NOLINT
"__global Dtype *dstc6 = dstc5 + N;",    // NOLINT
"__global Dtype *dstc7 = dstc6 + N;",    // NOLINT
"",    // NOLINT
"__local Dtype work0[SLM_SIZE];",    // NOLINT
"__local Dtype work1[SLM_SIZE];",    // NOLINT
"__local Dtype work2[SLM_SIZE];",    // NOLINT
"__local Dtype work3[SLM_SIZE];",    // NOLINT
"__local Dtype work4[SLM_SIZE];",    // NOLINT
"__local Dtype work5[SLM_SIZE];",    // NOLINT
"__local Dtype work6[SLM_SIZE];",    // NOLINT
"__local Dtype work7[SLM_SIZE];",    // NOLINT
"",    // NOLINT
"Dtype4 dot0 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot1 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot2 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot3 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot4 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot5 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot6 = (Dtype4)(0.);",    // NOLINT
"Dtype4 dot7 = (Dtype4)(0.);",    // NOLINT
"",    // NOLINT
"int kid = lid;",    // NOLINT
"while( kid < K / 4) {",    // NOLINT
"const Dtype4 a0 = vload4(kid, srca_read0);",    // NOLINT
"const Dtype4 a1 = vload4(kid, srca_read1);",    // NOLINT
"const Dtype4 a2 = vload4(kid, srca_read2);",    // NOLINT
"const Dtype4 a3 = vload4(kid, srca_read3);",    // NOLINT
"const Dtype4 a4 = vload4(kid, srca_read4);",    // NOLINT
"const Dtype4 a5 = vload4(kid, srca_read5);",    // NOLINT
"const Dtype4 a6 = vload4(kid, srca_read6);",    // NOLINT
"const Dtype4 a7 = vload4(kid, srca_read7);",    // NOLINT
"Dtype4 b = vload4(kid, srcb_read);",    // NOLINT
"dot0 += a0 * b;",    // NOLINT
"dot1 += a1 * b;",    // NOLINT
"dot2 += a2 * b;",    // NOLINT
"dot3 += a3 * b;",    // NOLINT
"dot4 += a4 * b;",    // NOLINT
"dot5 += a5 * b;",    // NOLINT
"dot6 += a6 * b;",    // NOLINT
"dot7 += a7 * b;",    // NOLINT
"",    // NOLINT
"kid += lsize;",    // NOLINT
"}",    // NOLINT
"work0[lid] = dot0.x + dot0.y + dot0.z + dot0.w;",    // NOLINT
"work1[lid] = dot1.x + dot1.y + dot1.z + dot1.w;",    // NOLINT
"work2[lid] = dot2.x + dot2.y + dot2.z + dot2.w;",    // NOLINT
"work3[lid] = dot3.x + dot3.y + dot3.z + dot3.w;",    // NOLINT
"work4[lid] = dot4.x + dot4.y + dot4.z + dot4.w;",    // NOLINT
"work5[lid] = dot5.x + dot5.y + dot5.z + dot5.w;",    // NOLINT
"work6[lid] = dot6.x + dot6.y + dot6.z + dot6.w;",    // NOLINT
"work7[lid] = dot7.x + dot7.y + dot7.z + dot7.w;",    // NOLINT
"",    // NOLINT
"if(kid == (K >> 2)) {",    // NOLINT
"short tail_items = K % 4;",    // NOLINT
"if(tail_items != 0) {",    // NOLINT
"int offset = kid << 2;",    // NOLINT
"const __global Dtype *srcb_tail = srcb_read + offset;",    // NOLINT
"",    // NOLINT
"const __global Dtype *srca_tail0 = srca_read0 + offset;",    // NOLINT
"const __global Dtype *srca_tail1 = srca_read1 + offset;",    // NOLINT
"const __global Dtype *srca_tail2 = srca_read2 + offset;",    // NOLINT
"const __global Dtype *srca_tail3 = srca_read3 + offset;",    // NOLINT
"const __global Dtype *srca_tail4 = srca_read4 + offset;",    // NOLINT
"const __global Dtype *srca_tail5 = srca_read5 + offset;",    // NOLINT
"const __global Dtype *srca_tail6 = srca_read6 + offset;",    // NOLINT
"const __global Dtype *srca_tail7 = srca_read7 + offset;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short item = 0; item < tail_items; ++item) {",    // NOLINT
"work0[lid] += srca_tail0[item] * srcb_tail[item];",    // NOLINT
"work1[lid] += srca_tail1[item] * srcb_tail[item];",    // NOLINT
"work2[lid] += srca_tail2[item] * srcb_tail[item];",    // NOLINT
"work3[lid] += srca_tail3[item] * srcb_tail[item];",    // NOLINT
"work4[lid] += srca_tail4[item] * srcb_tail[item];",    // NOLINT
"work5[lid] += srca_tail5[item] * srcb_tail[item];",    // NOLINT
"work6[lid] += srca_tail6[item] * srcb_tail[item];",    // NOLINT
"work7[lid] += srca_tail7[item] * srcb_tail[item];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride) {",    // NOLINT
"work0[lid] += work0[lid+stride];",    // NOLINT
"work1[lid] += work1[lid+stride];",    // NOLINT
"work2[lid] += work2[lid+stride];",    // NOLINT
"work3[lid] += work3[lid+stride];",    // NOLINT
"work4[lid] += work4[lid+stride];",    // NOLINT
"work5[lid] += work5[lid+stride];",    // NOLINT
"work6[lid] += work6[lid+stride];",    // NOLINT
"work7[lid] += work7[lid+stride];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"dstc0[x_gid] = alpha * work0[0] + beta * dstc0[x_gid];",    // NOLINT
"dstc1[x_gid] = alpha * work1[0] + beta * dstc1[x_gid];",    // NOLINT
"dstc2[x_gid] = alpha * work2[0] + beta * dstc2[x_gid];",    // NOLINT
"dstc3[x_gid] = alpha * work3[0] + beta * dstc3[x_gid];",    // NOLINT
"dstc4[x_gid] = alpha * work4[0] + beta * dstc4[x_gid];",    // NOLINT
"dstc5[x_gid] = alpha * work5[0] + beta * dstc5[x_gid];",    // NOLINT
"dstc6[x_gid] = alpha * work6[0] + beta * dstc6[x_gid];",    // NOLINT
"dstc7[x_gid] = alpha * work7[0] + beta * dstc7[x_gid];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"#undef SLM_SIZE",    // NOLINT
"",    // NOLINT
"#define VEC_SIZE        4",    // NOLINT
"#define LWG_HEIGHT      4",    // NOLINT
"#define TILE_M          8",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"#define TILE_K          32",    // NOLINT
"#define TILE_N          64",    // NOLINT
"#else",    // NOLINT
"#define TILE_K          16",    // NOLINT
"#define TILE_N          32",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, LWG_HEIGHT, 1)))",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM)))",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_TN, Dtype)(",    // NOLINT
"const __global Dtype *src0, int off0,",    // NOLINT
"const __global Dtype *src1, int off1,",    // NOLINT
"__global Dtype *dst, int offd,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_in,",    // NOLINT
"KERNEL_ARG_DTYPE beta_in,",    // NOLINT
"int start_index)",    // NOLINT
"",    // NOLINT
"{",    // NOLINT
"const Dtype alpha = (Dtype)alpha_in;",    // NOLINT
"const Dtype beta = (Dtype)beta_in;",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int local_x = get_local_id(0);",    // NOLINT
"const int local_y = get_local_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"",    // NOLINT
"Dtype4 brow;",    // NOLINT
"",    // NOLINT
"__global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src0_read = src0 + (local_x * (TILE_K / SIMD_SIZE_GEMM) + start_index) * M + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src1_read0 = src1 + local_x * VEC_SIZE + (group_x * TILE_N) + start_index * N + off1;",    // NOLINT
"",    // NOLINT
"Dtype4 dot00 = (start_index != 0) ? vload4(0, dst_write0) : beta * vload4(0, dst_write0);",    // NOLINT
"Dtype4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + N) : beta * vload4(0, dst_write0 + N);",    // NOLINT
"Dtype4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * N) : beta * vload4(0, dst_write0 + 2 * N);",    // NOLINT
"Dtype4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * N) : beta * vload4(0, dst_write0 + 3 * N);",    // NOLINT
"Dtype4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * N) : beta * vload4(0, dst_write0 + 4 * N);",    // NOLINT
"Dtype4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * N) : beta * vload4(0, dst_write0 + 5 * N);",    // NOLINT
"Dtype4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * N) : beta * vload4(0, dst_write0 + 6 * N);",    // NOLINT
"Dtype4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * N) : beta * vload4(0, dst_write0 + 7 * N);",    // NOLINT
"",    // NOLINT
"int end_index = min(start_index + 256, K);",    // NOLINT
"while( start_index + TILE_K <= end_index ) {",    // NOLINT
"Dtype8 arow0 = alpha * vload8(0, src0_read);",    // NOLINT
"Dtype8 arow1 = alpha * vload8(0, src0_read + M);",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _arow )         brow = vload4(0, src1_read0);  src1_read0 += N;         dot00 = mad( (Dtype4)(_arow.s0), brow, dot00 );         dot01 = mad( (Dtype4)(_arow.s1), brow, dot01 );         dot02 = mad( (Dtype4)(_arow.s2), brow, dot02 );         dot03 = mad( (Dtype4)(_arow.s3), brow, dot03 );         dot04 = mad( (Dtype4)(_arow.s4), brow, dot04 );         dot05 = mad( (Dtype4)(_arow.s5), brow, dot05 );         dot06 = mad( (Dtype4)(_arow.s6), brow, dot06 );         dot07 = mad( (Dtype4)(_arow.s7), brow, dot07 );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 0 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 0 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 1 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 1 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 2 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 2 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 3 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 3 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 4 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 4 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 5 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 5 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 6 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 6 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 7 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 7 )) );",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 8 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 8 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 9 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 9 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 10 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 10 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 11 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 11 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 12 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 12 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 13 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 13 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 14 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 14 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 15 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 15 )) );",    // NOLINT
"#endif",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"",    // NOLINT
"src0_read += TILE_K * M;",    // NOLINT
"start_index += TILE_K;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(start_index < end_index) {",    // NOLINT
"Dtype8 arow0 = ((start_index + local_x * 2) < K) ? alpha * vload8(0, src0_read) : (Dtype8)0.0f;",    // NOLINT
"Dtype8 arow1 = ((start_index + local_x * 2 + 1) < K) ? alpha * vload8(0, src0_read + M) : (Dtype8)0.0f;",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _arow )         brow = (start_index < K) ? vload4(0, src1_read0) : (Dtype4)0.0f;  src1_read0 += N; start_index++;         dot00 = mad( (Dtype4)(_arow.s0), brow, dot00 );         dot01 = mad( (Dtype4)(_arow.s1), brow, dot01 );         dot02 = mad( (Dtype4)(_arow.s2), brow, dot02 );         dot03 = mad( (Dtype4)(_arow.s3), brow, dot03 );         dot04 = mad( (Dtype4)(_arow.s4), brow, dot04 );         dot05 = mad( (Dtype4)(_arow.s5), brow, dot05 );         dot06 = mad( (Dtype4)(_arow.s6), brow, dot06 );         dot07 = mad( (Dtype4)(_arow.s7), brow, dot07 );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 0 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 0 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 1 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 1 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 2 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 2 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 3 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 3 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 4 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 4 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 5 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 5 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 6 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 6 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 7 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 7 )) );",    // NOLINT
"#if TYPE == TYPE_HALF",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 8 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 8 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 9 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 9 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 10 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 10 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 11 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 11 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 12 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 12 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 13 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 13 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 14 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 14 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 15 )) );",    // NOLINT
"MM_DOT_PRODUCT( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 15 )) );",    // NOLINT
"#endif",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(global_x * 4 < N && global_y * 8 < M) {",    // NOLINT
"if(mad24(global_x, 4, 3) < N) {",    // NOLINT
"vstore4(dot00, 0, dst_write0); dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }",    // NOLINT
"} else if(mad24(global_x, 4, 2) < N) {",    // NOLINT
"vstore2(dot00.xy, 0, dst_write0); dst_write0[2] = dot00.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) {",    // NOLINT
"vstore2(dot01.xy, 0, dst_write0); dst_write0[2] = dot01.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) {",    // NOLINT
"vstore2(dot02.xy, 0, dst_write0); dst_write0[2] = dot02.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) {",    // NOLINT
"vstore2(dot03.xy, 0, dst_write0); dst_write0[2] = dot03.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) {",    // NOLINT
"vstore2(dot04.xy, 0, dst_write0); dst_write0[2] = dot04.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) {",    // NOLINT
"vstore2(dot05.xy, 0, dst_write0); dst_write0[2] = dot05.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) {",    // NOLINT
"vstore2(dot06.xy, 0, dst_write0); dst_write0[2] = dot06.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) {",    // NOLINT
"vstore2(dot07.xy, 0, dst_write0); dst_write0[2] = dot07.z;",    // NOLINT
"}",    // NOLINT
"} else if(mad24(global_x, 4, 1) < N) {",    // NOLINT
"vstore2(dot00.xy, 0, dst_write0); dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }",    // NOLINT
"} else {",    // NOLINT
"dst_write0[0] = dot00.x; dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; }",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#undef VEC_SIZE",    // NOLINT
"#undef LWG_HEIGHT",    // NOLINT
"#undef TILE_M",    // NOLINT
"#undef TILE_K",    // NOLINT
"#undef TILE_N",    // NOLINT
"",    // NOLINT
"#define VEC_SIZE        4",    // NOLINT
"#define LWG_HEIGHT      4",    // NOLINT
"#define TILE_M          8",    // NOLINT
"#define TILE_K          16",    // NOLINT
"#define TILE_N          32",    // NOLINT
"",    // NOLINT
"__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(8)))",    // NOLINT
"__kernel void TEMPLATE(gemm_buffer_TT, Dtype)(",    // NOLINT
"const __global Dtype *src0, int off0,",    // NOLINT
"const __global Dtype *src1, int off1,",    // NOLINT
"__global Dtype *dst, int offd,",    // NOLINT
"int M,",    // NOLINT
"int N,",    // NOLINT
"int K,",    // NOLINT
"KERNEL_ARG_DTYPE alpha_in,",    // NOLINT
"KERNEL_ARG_DTYPE beta_in,",    // NOLINT
"int start_index)",    // NOLINT
"",    // NOLINT
"{",    // NOLINT
"const Dtype alpha = (Dtype)alpha_in;",    // NOLINT
"const Dtype beta = (Dtype)beta_in;",    // NOLINT
"const int group_x = get_group_id(0);",    // NOLINT
"const int group_y = get_group_id(1);",    // NOLINT
"const int local_x = get_local_id(0);",    // NOLINT
"const int local_y = get_local_id(1);",    // NOLINT
"const int global_x = get_global_id(0);",    // NOLINT
"const int global_y = get_global_id(1);",    // NOLINT
"",    // NOLINT
"Dtype8 dot0 = 0.f;",    // NOLINT
"Dtype8 dot1 = 0.f;",    // NOLINT
"Dtype8 dot2 = 0.f;",    // NOLINT
"Dtype8 dot3 = 0.f;",    // NOLINT
"",    // NOLINT
"Dtype16 brow0;",    // NOLINT
"Dtype16 brow1;",    // NOLINT
"Dtype16 brow2;",    // NOLINT
"Dtype16 brow3;",    // NOLINT
"",    // NOLINT
"__global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src0_read = src0 + (local_x * (TILE_K / 8) + start_index) * M + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0;",    // NOLINT
"",    // NOLINT
"const __global Dtype *src1_read0 = src1 + (local_x * VEC_SIZE + (group_x * TILE_N)) * K + start_index + off1;",    // NOLINT
"",    // NOLINT
"Dtype4 dot00 = (start_index != 0) ? vload4(0, dst_write0) : beta * vload4(0, dst_write0);",    // NOLINT
"Dtype4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + N) : beta * vload4(0, dst_write0 + N);",    // NOLINT
"Dtype4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * N) : beta * vload4(0, dst_write0 + 2 * N);",    // NOLINT
"Dtype4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * N) : beta * vload4(0, dst_write0 + 3 * N);",    // NOLINT
"Dtype4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * N) : beta * vload4(0, dst_write0 + 4 * N);",    // NOLINT
"Dtype4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * N) : beta * vload4(0, dst_write0 + 5 * N);",    // NOLINT
"Dtype4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * N) : beta * vload4(0, dst_write0 + 6 * N);",    // NOLINT
"Dtype4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * N) : beta * vload4(0, dst_write0 + 7 * N);",    // NOLINT
"",    // NOLINT
"int end_index = min(start_index + 256, K);",    // NOLINT
"while( start_index + TILE_K <= end_index ) {",    // NOLINT
"brow0 = vload16(0, src1_read0);",    // NOLINT
"brow1 = vload16(0, src1_read0 + K);",    // NOLINT
"brow2 = vload16(0, src1_read0 + 2 * K);",    // NOLINT
"brow3 = vload16(0, src1_read0 + 3 * K);",    // NOLINT
"",    // NOLINT
"Dtype8 arow0 = alpha * vload8(0, src0_read);",    // NOLINT
"Dtype8 arow1 = alpha * vload8(0, src0_read + M);",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _brow, _dot)         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 0 )), (Dtype8)_brow.s0, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 0 )), (Dtype8)_brow.s1, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 1 )), (Dtype8)_brow.s2, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 1 )), (Dtype8)_brow.s3, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 2 )), (Dtype8)_brow.s4, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 2 )), (Dtype8)_brow.s5, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 3 )), (Dtype8)_brow.s6, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 3 )), (Dtype8)_brow.s7, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 4 )), (Dtype8)_brow.s8, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 4 )), (Dtype8)_brow.s9, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 5 )), (Dtype8)_brow.sa, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 5 )), (Dtype8)_brow.sb, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 6 )), (Dtype8)_brow.sc, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 6 )), (Dtype8)_brow.sd, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 7 )), (Dtype8)_brow.se, _dot );         _dot = mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 7 )), (Dtype8)_brow.sf, _dot );",    // NOLINT
"MM_DOT_PRODUCT( brow0, dot0 );",    // NOLINT
"MM_DOT_PRODUCT( brow1, dot1 );",    // NOLINT
"MM_DOT_PRODUCT( brow2, dot2 );",    // NOLINT
"MM_DOT_PRODUCT( brow3, dot3 );",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"",    // NOLINT
"src1_read0 += TILE_K;",    // NOLINT
"src0_read += TILE_K * M;",    // NOLINT
"start_index += TILE_K;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(start_index < end_index) {",    // NOLINT
"brow0 = vload16(0, src1_read0);  src1_read0 += K;",    // NOLINT
"brow1 = vload16(0, src1_read0);  src1_read0 += K;",    // NOLINT
"brow2 = vload16(0, src1_read0);  src1_read0 += K;",    // NOLINT
"brow3 = vload16(0, src1_read0);",    // NOLINT
"",    // NOLINT
"Dtype8 arow0 = alpha * vload8(0, src0_read);",    // NOLINT
"Dtype8 arow1 = alpha * vload8(0, src0_read + M);",    // NOLINT
"",    // NOLINT
"#define MM_DOT_PRODUCT( _brow, _dot)         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 0 )), (Dtype8)_brow.s0, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 0 )), (Dtype8)_brow.s1, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 1 )), (Dtype8)_brow.s2, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 1 )), (Dtype8)_brow.s3, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 2 )), (Dtype8)_brow.s4, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 2 )), (Dtype8)_brow.s5, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 3 )), (Dtype8)_brow.s6, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 3 )), (Dtype8)_brow.s7, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 4 )), (Dtype8)_brow.s8, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 4 )), (Dtype8)_brow.s9, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 5 )), (Dtype8)_brow.sa, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 5 )), (Dtype8)_brow.sb, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 6 )), (Dtype8)_brow.sc, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 6 )), (Dtype8)_brow.sd, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow0), 7 )), (Dtype8)_brow.se, _dot ) : _dot;         _dot = (w++ < K) ? mad( as_Dtype8(intel_sub_group_shuffle( SHUFFLE_TYPE8(arow1), 7 )), (Dtype8)_brow.sf, _dot ) : _dot;",    // NOLINT
"int w = start_index;",    // NOLINT
"MM_DOT_PRODUCT( brow0, dot0 );",    // NOLINT
"w = start_index;",    // NOLINT
"MM_DOT_PRODUCT( brow1, dot1 );",    // NOLINT
"w = start_index;",    // NOLINT
"MM_DOT_PRODUCT( brow2, dot2 );",    // NOLINT
"w = start_index;",    // NOLINT
"MM_DOT_PRODUCT( brow3, dot3 );",    // NOLINT
"#undef MM_DOT_PRODUCT",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"dot00 += (Dtype4)(dot0.s0, dot1.s0, dot2.s0, dot3.s0);",    // NOLINT
"dot01 += (Dtype4)(dot0.s1, dot1.s1, dot2.s1, dot3.s1);",    // NOLINT
"dot02 += (Dtype4)(dot0.s2, dot1.s2, dot2.s2, dot3.s2);",    // NOLINT
"dot03 += (Dtype4)(dot0.s3, dot1.s3, dot2.s3, dot3.s3);",    // NOLINT
"dot04 += (Dtype4)(dot0.s4, dot1.s4, dot2.s4, dot3.s4);",    // NOLINT
"dot05 += (Dtype4)(dot0.s5, dot1.s5, dot2.s5, dot3.s5);",    // NOLINT
"dot06 += (Dtype4)(dot0.s6, dot1.s6, dot2.s6, dot3.s6);",    // NOLINT
"dot07 += (Dtype4)(dot0.s7, dot1.s7, dot2.s7, dot3.s7);",    // NOLINT
"",    // NOLINT
"if(global_x * 4 < N && global_y * 8 < M) {",    // NOLINT
"if(mad24(global_x, 4, 3) < N) {",    // NOLINT
"vstore4(dot00, 0, dst_write0); dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }",    // NOLINT
"} else if(mad24(global_x, 4, 2) < N) {",    // NOLINT
"vstore2(dot00.xy, 0, dst_write0); dst_write0[2] = dot00.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) {",    // NOLINT
"vstore2(dot01.xy, 0, dst_write0); dst_write0[2] = dot01.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) {",    // NOLINT
"vstore2(dot02.xy, 0, dst_write0); dst_write0[2] = dot02.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) {",    // NOLINT
"vstore2(dot03.xy, 0, dst_write0); dst_write0[2] = dot03.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) {",    // NOLINT
"vstore2(dot04.xy, 0, dst_write0); dst_write0[2] = dot04.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) {",    // NOLINT
"vstore2(dot05.xy, 0, dst_write0); dst_write0[2] = dot05.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) {",    // NOLINT
"vstore2(dot06.xy, 0, dst_write0); dst_write0[2] = dot06.z;",    // NOLINT
"dst_write0 += N;",    // NOLINT
"} else",    // NOLINT
"return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) {",    // NOLINT
"vstore2(dot07.xy, 0, dst_write0); dst_write0[2] = dot07.z;",    // NOLINT
"}",    // NOLINT
"} else if(mad24(global_x, 4, 1) < N) {",    // NOLINT
"vstore2(dot00.xy, 0, dst_write0); dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }",    // NOLINT
"} else {",    // NOLINT
"dst_write0[0] = dot00.x; dst_write0 += N;",    // NOLINT
"if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += N; }",    // NOLINT
"else return;",    // NOLINT
"if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; }",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#undef VEC_SIZE",    // NOLINT
"#undef LWG_HEIGHT",    // NOLINT
"#undef TILE_M",    // NOLINT
"#undef TILE_K",    // NOLINT
"#undef TILE_N",    // NOLINT
"#undef SIMD_SIZE_GEMM",    // NOLINT
"#undef SHUFFLE_TYPE2",    // NOLINT
"#undef SHUFFLE_TYPE8",    // NOLINT
"",    // NOLINT
"#endif",    // NOLINT
"#endif",    // NOLINT
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
"data_im[index] = 0;",    // NOLINT
"done = true;",    // NOLINT
"break;  // for (int_tp i = 0; i < num_axes; ++i)",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"if (!done) {",    // NOLINT
"// Loop over the col to compute the output val.",    // NOLINT
"Dtype val = 0;",    // NOLINT
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
"const KERNEL_ARG_DTYPE negative_beta,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < nthreads;",    // NOLINT
"index += get_global_size(0)) {",    // NOLINT
"out[index] = in[index] * pow(scale[index], (Dtype)negative_beta);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,",    // NOLINT
"const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp size,",    // NOLINT
"const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,",    // NOLINT
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
"const KERNEL_ARG_DTYPE negative_beta,",    // NOLINT
"const KERNEL_ARG_DTYPE cache_ratio,",    // NOLINT
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
"* step] * pow(scale_off[(head - post_pad) * step], (Dtype)negative_beta)",    // NOLINT
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
"* step] * pow(scale_off[(head - post_pad) * step], (Dtype)negative_beta)",    // NOLINT
"- cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#if defined(cl_intel_subgroups)",    // NOLINT
"#pragma OPENCL EXTENSION  cl_intel_subgroups : enable",    // NOLINT
"",    // NOLINT
"#define SIMD_WIDTH 16",    // NOLINT
"#define TILE_W SIMD_WIDTH",    // NOLINT
"#define TILE_H 8",    // NOLINT
"",    // NOLINT
"#ifndef BEIGNET",    // NOLINT
"__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))",    // NOLINT
"#endif",    // NOLINT
"// Fuse pooling max layer into LRN across channel layer.",    // NOLINT
"// Currently, only support non-padding, non-dilation mode and pool_w/h == pool_stride_w + 1.",    // NOLINT
"// This kernel only get better performance on those Intel platforms with edram.",    // NOLINT
"__kernel void TEMPLATE(lrn_fuse_pool_max,Dtype)(",    // NOLINT
"__global const Dtype* in,",    // NOLINT
"const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width,",    // NOLINT
"const int_tp tiled_height, int_tp tiled_width,",    // NOLINT
"const int_tp size,",    // NOLINT
"const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,",    // NOLINT
"__global Dtype* const out,",    // NOLINT
"const KERNEL_ARG_DTYPE negative_beta,",    // NOLINT
"const int_tp pool_h, const int_tp pool_w, const int_tp pool_stride_h, int_tp pool_stride_w,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width,",    // NOLINT
"const int_tp tile_pooled_block_h, const int_tp tile_pooled_block_w) {",    // NOLINT
"// find out the local offset",    // NOLINT
"const int_tp block_x = get_global_id(0) % tiled_width;",    // NOLINT
"const int_tp block_y = (get_global_id(0) / tiled_width) % tiled_height;",    // NOLINT
"const int_tp n = get_global_id(0) / (tiled_width * tiled_height);",    // NOLINT
"",    // NOLINT
"const int_tp w = block_x * tile_pooled_block_w * pool_stride_w;",    // NOLINT
"const int_tp h = block_y * tile_pooled_block_h * pool_stride_h;",    // NOLINT
"const int_tp offset = (n * channels * height + h) * width + w;",    // NOLINT
"const int_tp out_h = block_y * tile_pooled_block_h;",    // NOLINT
"const int_tp out_w = block_x * tile_pooled_block_w;",    // NOLINT
"const int_tp out_offset = (n * channels * pooled_height + out_h) * pooled_width + out_w + get_local_id(1);",    // NOLINT
"const int_tp step = height * width;",    // NOLINT
"const int_tp out_step = pooled_height * pooled_width;",    // NOLINT
"__global const Dtype* in_off = in + offset + get_local_id(1);",    // NOLINT
"__global Dtype* out_off = out + out_offset;",    // NOLINT
"Dtype scale_val;",    // NOLINT
"int_tp head = 0;",    // NOLINT
"const int_tp pre_pad = (size - 1) / 2;",    // NOLINT
"const int_tp post_pad = size - pre_pad - 1;",    // NOLINT
"Dtype accum_scale[TILE_H] = {0};",    // NOLINT
"if (w + get_local_id(1) >= width)",    // NOLINT
"return;",    // NOLINT
"",    // NOLINT
"while ( head < channels + post_pad ) {",    // NOLINT
"int ph = 0;",    // NOLINT
"int cur_out_h = 0;",    // NOLINT
"Dtype output_val = -DTYPE_MAX;",    // NOLINT
"// fill the scale at [n, :, h, w]",    // NOLINT
"// accumulate values",    // NOLINT
"for( int lrn_out_h = 0; lrn_out_h < TILE_H && (lrn_out_h + h) < height; lrn_out_h++) {",    // NOLINT
"Dtype prev_val = accum_scale[lrn_out_h];",    // NOLINT
"// add",    // NOLINT
"if (head < channels) {",    // NOLINT
"prev_val += in_off[head * step + width * lrn_out_h] * in_off[head * step + width * lrn_out_h];",    // NOLINT
"}",    // NOLINT
"// subtract",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"prev_val -= in_off[(head - size) * step + width * lrn_out_h] * in_off[(head - size) * step + width * lrn_out_h];",    // NOLINT
"}",    // NOLINT
"// compute output.",    // NOLINT
"if (head >= post_pad) {",    // NOLINT
"scale_val = k + prev_val * alpha_over_size;",    // NOLINT
"Dtype tmp = -DTYPE_MAX;",    // NOLINT
"//if (w + get_local_id(1) < width)",    // NOLINT
"tmp = in_off[(head - post_pad) * step + width * lrn_out_h] * native_powr(scale_val, (Dtype)negative_beta);",    // NOLINT
"",    // NOLINT
"Dtype h_max_val = -DTYPE_MAX;",    // NOLINT
"int index = (get_local_id(1) * pool_stride_w) % SIMD_WIDTH;",    // NOLINT
"for(int i = 0; i < pool_w; i++) {",    // NOLINT
"Dtype val = intel_sub_group_shuffle(tmp, index);",    // NOLINT
"if (h_max_val < val && (index + w < width))",    // NOLINT
"h_max_val = val;",    // NOLINT
"",    // NOLINT
"index = (index + 1) % SIMD_WIDTH;",    // NOLINT
"}",    // NOLINT
"// update output value.",    // NOLINT
"output_val = (output_val > h_max_val) ?",    // NOLINT
"output_val : h_max_val;",    // NOLINT
"// time to write previous output and move to next value",    // NOLINT
"if (lrn_out_h - cur_out_h + 1 == pool_h) {",    // NOLINT
"if (get_local_id(1) < tile_pooled_block_w && (out_w + get_local_id(1)) < pooled_width) {",    // NOLINT
"out_off[(head - post_pad) * out_step + ph * pooled_width] = output_val;",    // NOLINT
"",    // NOLINT
"output_val = h_max_val;",    // NOLINT
"}",    // NOLINT
"++ph;",    // NOLINT
"cur_out_h += pool_stride_h;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"accum_scale[lrn_out_h] = prev_val;",    // NOLINT
"}",    // NOLINT
"// Handle the incomplete pool box",    // NOLINT
"// an incomplete tiling box and we are not hitting the end of the pooled output.",    // NOLINT
"if (head >= post_pad &&",    // NOLINT
"ph < tile_pooled_block_h &&",    // NOLINT
"ph + out_h < pooled_height &&",    // NOLINT
"get_local_id(1) < tile_pooled_block_w &&",    // NOLINT
"(out_w + get_local_id(1)) < pooled_width) {",    // NOLINT
"out_off[(head - post_pad) * out_step + ph * pooled_width] = output_val;",    // NOLINT
"}",    // NOLINT
"head++;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"#undef TILE_W",    // NOLINT
"#undef TILE_H",    // NOLINT
"#undef SIMD_WIDTH",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_full_no_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,",    // NOLINT
"const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp size,",    // NOLINT
"const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,",    // NOLINT
"__global Dtype* const out,",    // NOLINT
"const KERNEL_ARG_DTYPE negative_beta) {",    // NOLINT
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
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"// subtract only",    // NOLINT
"while (head < channels + post_pad) {",    // NOLINT
"if (head - size >= 0) {",    // NOLINT
"accum_scale -= in_off[(head - size) * step]",    // NOLINT
"* in_off[(head - size) * step];",    // NOLINT
"}",    // NOLINT
"scale_val = k + accum_scale * alpha_over_size;",    // NOLINT
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);",    // NOLINT
"++head;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(lrn_full,Dtype)(const int_tp nthreads, __global const Dtype* in,",    // NOLINT
"const int_tp num, const int_tp channels,",    // NOLINT
"const int_tp height, const int_tp width, const int_tp size,",    // NOLINT
"const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,",    // NOLINT
"__global Dtype* const scale,",    // NOLINT
"__global Dtype* const out,",    // NOLINT
"const KERNEL_ARG_DTYPE negative_beta) {",    // NOLINT
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
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);",    // NOLINT
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
"out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);",    // NOLINT
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
"__kernel void TEMPLATE(add_scalar,Dtype)(const int_tp N, const KERNEL_ARG_DTYPE alpha,",    // NOLINT
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
"__kernel void TEMPLATE(powx,Dtype)(const int_tp n, __global const Dtype* a,",    // NOLINT
"const int_tp offa, KERNEL_ARG_DTYPE alpha,",    // NOLINT
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
    {"void TEMPLATE(matvec_mul_trail_rows,Dtype)(unsigned int M,",    // NOLINT
"unsigned int N,",    // NOLINT
"int row_gid,",    // NOLINT
"int lid,",    // NOLINT
"const __global Dtype* src0_read,",    // NOLINT
"int lda,",    // NOLINT
"const __global Dtype* src1_read,",    // NOLINT
"int incv,",    // NOLINT
"__local Dtype4* work,",    // NOLINT
"Dtype alpha,",    // NOLINT
"Dtype beta,",    // NOLINT
"__global Dtype* result,",    // NOLINT
"int incr)",    // NOLINT
"{",    // NOLINT
"__local Dtype* work_each = (__local Dtype*)work;",    // NOLINT
"",    // NOLINT
"int rows = M - row_gid * 4;",    // NOLINT
"",    // NOLINT
"Dtype4 dot[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};",    // NOLINT
"",    // NOLINT
"int i = lid;",    // NOLINT
"while( i < N / 4) {",    // NOLINT
"const Dtype4 b0 = {src1_read[i*4*incv], src1_read[(i*4+1)*incv], src1_read[(i*4+2)*incv], src1_read[(i*4+3)*incv]};",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"dot[j] += b0 * vload4(i, src0_read + j * lda);",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"i += get_local_size(0);",    // NOLINT
"}",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"work_each[lid * 4 + j] = dot[j].x + dot[j].y + dot[j].z + dot[j].w;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(i == N / 4) {",    // NOLINT
"short trail_item = N % 4;",    // NOLINT
"",    // NOLINT
"if(trail_item != 0) {",    // NOLINT
"const __global Dtype *src0_trail = src0_read + i * 4;",    // NOLINT
"const __global Dtype *src1_trail = src1_read + i * 4 * incv;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short i = 0; i < trail_item; ++i) {",    // NOLINT
"const Dtype bt = src1_trail[i*incv];",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"work_each[lid * 4 + j] += bt * src0_trail[i + j * lda];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride)",    // NOLINT
"work[lid] += work[lid+stride];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < rows; ++j) {",    // NOLINT
"result[(row_gid * 4  + j) * incr] = alpha * work_each[j] + beta * result[(row_gid * 4 + j) * incr];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(matvec_mul,Dtype)(",    // NOLINT
"unsigned int M,",    // NOLINT
"unsigned int N,",    // NOLINT
"__global const Dtype * A,",    // NOLINT
"int offA,",    // NOLINT
"int lda,",    // NOLINT
"__global const Dtype * v,",    // NOLINT
"int offv,",    // NOLINT
"int incv,",    // NOLINT
"KERNEL_ARG_DTYPE alpha,",    // NOLINT
"KERNEL_ARG_DTYPE beta,",    // NOLINT
"__global Dtype * result,",    // NOLINT
"int offr,",    // NOLINT
"int incr)",    // NOLINT
"{",    // NOLINT
"int row_gid = get_group_id(0);",    // NOLINT
"int lid = get_local_id(0);",    // NOLINT
"const __global Dtype *src0_read = A + row_gid * 4 * lda + offA;",    // NOLINT
"const __global Dtype *src1_read = v + offv;",    // NOLINT
"result = result + offr;",    // NOLINT
"",    // NOLINT
"src1_read += incv > 0 ? 0 : (1 - N) * incv;",    // NOLINT
"result += incr > 0 ? 0 : (1 - M) * incr;",    // NOLINT
"__local Dtype4 work[128];",    // NOLINT
"__local Dtype* work_each = (__local Dtype*)work;",    // NOLINT
"",    // NOLINT
"if(row_gid == M / 4)",    // NOLINT
"TEMPLATE(matvec_mul_trail_rows,Dtype)(M, N, row_gid, lid, src0_read, lda, src1_read, incv, work, alpha, beta, result, incr);",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"Dtype4 dot[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.f), (Dtype4)(0.f)};",    // NOLINT
"int i = lid;",    // NOLINT
"while( i < N / 4) {",    // NOLINT
"const Dtype4 b0 = {src1_read[i*4*incv], src1_read[(i*4+1)*incv], src1_read[(i*4+2)*incv], src1_read[(i*4+3)*incv]};",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"dot[j] += b0 * vload4(i, src0_read + j * lda);",    // NOLINT
"}",    // NOLINT
"i += get_local_size(0);",    // NOLINT
"}",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"work_each[lid * 4 + j] = dot[j].x + dot[j].y + dot[j].z + dot[j].w;",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(i == N / 4) {",    // NOLINT
"short trail_item = N % 4;",    // NOLINT
"if(trail_item != 0) {",    // NOLINT
"const __global Dtype *src0_trail = src0_read + i * 4;",    // NOLINT
"const __global Dtype *src1_trail = src1_read + i * 4 * incv;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(short i = 0; i < trail_item; ++i) {",    // NOLINT
"const Dtype bt = src1_trail[i * incv];",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int j = 0; j < 4; ++j) {",    // NOLINT
"work_each[lid * 4 + j] += bt * src0_trail[i + j * lda];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {",    // NOLINT
"barrier(CLK_LOCAL_MEM_FENCE);",    // NOLINT
"if(lid < stride)",    // NOLINT
"work[lid] += work[lid+stride];",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"if(lid == 0) {",    // NOLINT
"// vstore4(alpha * work[0] + beta * vload4(row_gid, result), row_gid, result);",    // NOLINT
"result[row_gid*4*incr] = alpha * work[0].s0 + beta * result[row_gid*4*incr];",    // NOLINT
"result[(row_gid*4+1)*incr] = alpha * work[0].s1 + beta * result[(row_gid*4+1)*incr];",    // NOLINT
"result[(row_gid*4+2)*incr] = alpha * work[0].s2 + beta * result[(row_gid*4+2)*incr];",    // NOLINT
"result[(row_gid*4+3)*incr] = alpha * work[0].s3 + beta * result[(row_gid*4+3)*incr];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(trans_matvec_mul,Dtype)(",    // NOLINT
"unsigned int M,",    // NOLINT
"unsigned int N,",    // NOLINT
"__global const Dtype * A,",    // NOLINT
"int offA,",    // NOLINT
"int lda,",    // NOLINT
"__global const Dtype * v,",    // NOLINT
"int offv,",    // NOLINT
"int incv,",    // NOLINT
"KERNEL_ARG_DTYPE alpha,",    // NOLINT
"KERNEL_ARG_DTYPE beta,",    // NOLINT
"__global Dtype * result,",    // NOLINT
"int offr,",    // NOLINT
"int incr)",    // NOLINT
"{",    // NOLINT
"int col_gid = get_global_id(0);",    // NOLINT
"A += offA + col_gid;",    // NOLINT
"v += offv;",    // NOLINT
"result += offr;",    // NOLINT
"",    // NOLINT
"v += incv > 0 ? 0 : (1 - M) * incv;",    // NOLINT
"result += incr > 0 ? 0 : (1 - N) * incr;",    // NOLINT
"",    // NOLINT
"Dtype dot_prod = 0;",    // NOLINT
"int row_id = 0;",    // NOLINT
"#pragma unroll",    // NOLINT
"for(int row = 0; row < M; ++row) {",    // NOLINT
"dot_prod += A[row_id] * v[row * incv];",    // NOLINT
"row_id += lda;",    // NOLINT
"}",    // NOLINT
"result[col_gid * incr] = beta * result[col_gid * incr];",    // NOLINT
"result[col_gid * incr] += alpha * dot_prod;",    // NOLINT
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
"__kernel void TEMPLATE(DivBsx, Dtype)(const int nthreads,",    // NOLINT
"__global const Dtype* A, const int A_off, __global const Dtype* v, const int v_off, const int rows, const int cols,",    // NOLINT
"__global Dtype* B, const int B_off) {",    // NOLINT
"",    // NOLINT
"for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {",    // NOLINT
"int c = index % cols;",    // NOLINT
"B[index+B_off] = A[index+A_off] / v[c+v_off];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(MulBsx, Dtype)(const int nthreads, __global Dtype* A, const int A_off,",    // NOLINT
"__global Dtype* v, const int rows, const int cols, int trans,",    // NOLINT
"__global Dtype* B, const int B_off) {",    // NOLINT
"for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {",    // NOLINT
"int c = index % cols;",    // NOLINT
"int r = (index / cols) % rows;",    // NOLINT
"if (trans == 0) {",    // NOLINT
"B[index+B_off] = A[index+A_off] * v[c];",    // NOLINT
"} else {",    // NOLINT
"B[index+B_off] = A[index+A_off] * v[r];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"#define OCL_KERNEL_LOOP(i, n)  for (int i = get_global_id(0); i < (n); i += get_global_size(0))",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(PermuteKernel, Dtype)(const int nthreads,",    // NOLINT
"__global Dtype* bottom_data, const int forward, global int* permute_order,",    // NOLINT
"global int* old_steps, global int* new_steps, const int num_axes,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"OCL_KERNEL_LOOP(index, nthreads) {",    // NOLINT
"int temp_idx = index;",    // NOLINT
"int old_idx = 0;",    // NOLINT
"for (int i = 0; i < num_axes; ++i) {",    // NOLINT
"int order = permute_order[i];",    // NOLINT
"old_idx += (temp_idx / new_steps[i]) * old_steps[order];",    // NOLINT
"temp_idx %= new_steps[i];",    // NOLINT
"}",    // NOLINT
"if (forward != 0) {",    // NOLINT
"top_data[index] = bottom_data[old_idx];",    // NOLINT
"} else {",    // NOLINT
"bottom_data[old_idx] = top_data[index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"",    // NOLINT
"",    // NOLINT
""},   // NOLINT
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"void TEMPLATE(max_pool_forward_impl, Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* top_data,",    // NOLINT
"const int use_mask, __global int_tp* mask, __global Dtype* top_mask, bool no_mask) {",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"if (!no_mask) {",    // NOLINT
"if (use_mask == 1) {",    // NOLINT
"mask[index] = maxidx;",    // NOLINT
"} else {",    // NOLINT
"top_mask[index] = maxidx;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_forward_no_mask, Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* top_data) {",    // NOLINT
"",    // NOLINT
"TEMPLATE(max_pool_forward_impl, Dtype)(",    // NOLINT
"nthreads, bottom_data, num, channels, height, width,",    // NOLINT
"pooled_height, pooled_width, kernel_h,",    // NOLINT
"kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, 0, NULL, NULL, true",    // NOLINT
");",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(max_pool_forward, Dtype)(",    // NOLINT
"const int_tp nthreads, __global const Dtype* bottom_data, const int_tp num,",    // NOLINT
"const int_tp channels, const int_tp height, const int_tp width,",    // NOLINT
"const int_tp pooled_height, const int_tp pooled_width, const int_tp kernel_h,",    // NOLINT
"const int_tp kernel_w, const int_tp stride_h, const int_tp stride_w, const int_tp pad_h,",    // NOLINT
"const int_tp pad_w,",    // NOLINT
"__global Dtype* top_data,",    // NOLINT
"const int use_mask, __global int_tp* mask, __global Dtype* top_mask) {",    // NOLINT
"",    // NOLINT
"TEMPLATE(max_pool_forward_impl, Dtype)(",    // NOLINT
"nthreads, bottom_data, num, channels, height, width,",    // NOLINT
"pooled_height, pooled_width, kernel_h,",    // NOLINT
"kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, use_mask, mask, top_mask, false",    // NOLINT
");",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ave_pool_forward, Dtype)(",    // NOLINT
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
"const Dtype thres = rand_idx[index] * cumsum;",    // NOLINT
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
"Dtype cumsum = DTYPE_MIN;",    // NOLINT
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
"* (Dtype)(index == (int_tp) (rand_idx_slice[ph * pooled_width + pw])?1.0:0.0);",    // NOLINT
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
"top_data[index] = -DTYPE_MAX;",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"Dtype thres = rand_idx[index] * cumsum;",    // NOLINT
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
"Dtype cumsum = DTYPE_MIN;",    // NOLINT
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
"__kernel void TEMPLATE(reorg, Dtype)(const int_tp n,__global const Dtype* x,",    // NOLINT
"int_tp w, int_tp h, int_tp c,",    // NOLINT
"int_tp batch, int_tp stride, int_tp forward,",    // NOLINT
"__global Dtype* out) {",    // NOLINT
"int_tp size = batch*c*h*w;",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0))",    // NOLINT
"{",    // NOLINT
"int_tp i;",    // NOLINT
"i = index;",    // NOLINT
"if(i >= size) return;",    // NOLINT
"int_tp in_index = i;",    // NOLINT
"int_tp in_w = i%w;",    // NOLINT
"i = i/w;",    // NOLINT
"int_tp in_h = i%h;",    // NOLINT
"i = i/h;",    // NOLINT
"int_tp in_c = i%c;",    // NOLINT
"i = i/c;",    // NOLINT
"int_tp b = i%batch;",    // NOLINT
"",    // NOLINT
"int_tp out_c = c/(stride*stride);",    // NOLINT
"",    // NOLINT
"int_tp c2 = in_c % out_c;",    // NOLINT
"int_tp offset = in_c / out_c;",    // NOLINT
"int_tp w2 = in_w*stride + (offset % stride);",    // NOLINT
"int_tp h2 = in_h*stride + offset / stride;",    // NOLINT
"int_tp out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));",    // NOLINT
"",    // NOLINT
"if(forward)",    // NOLINT
"{",    // NOLINT
"out[out_index] = x[in_index];",    // NOLINT
"}",    // NOLINT
"else",    // NOLINT
"{",    // NOLINT
"out[in_index] = x[out_index];",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"Dtype maxval = -DTYPE_MAX;",    // NOLINT
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
"(Dtype) DTYPE_MIN)));",    // NOLINT
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
"KERNEL_ARG_DTYPE momentum,",    // NOLINT
"KERNEL_ARG_DTYPE delta,",    // NOLINT
"KERNEL_ARG_DTYPE local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype gi = g[i];",    // NOLINT
"Dtype hi = h[i] = momentum * h[i] + ((Dtype)1.0 - momentum) * gi * gi;",    // NOLINT
"gi = gi * sqrt((h2[i] + delta) / (hi + delta));",    // NOLINT
"h2[i] = momentum * h2[i] + ((Dtype)1.0 - momentum) * gi * gi;",    // NOLINT
"g[i] = local_rate * gi;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(ada_grad_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"KERNEL_ARG_DTYPE delta,",    // NOLINT
"KERNEL_ARG_DTYPE local_rate) {",    // NOLINT
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
"KERNEL_ARG_DTYPE beta1,",    // NOLINT
"KERNEL_ARG_DTYPE beta2,",    // NOLINT
"KERNEL_ARG_DTYPE eps_hat,",    // NOLINT
"KERNEL_ARG_DTYPE corrected_local_rate) {",    // NOLINT
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
"KERNEL_ARG_DTYPE momentum,",    // NOLINT
"KERNEL_ARG_DTYPE local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype hi = h[i];",    // NOLINT
"Dtype hi_new = h[i] = momentum * hi + local_rate * g[i];",    // NOLINT
"g[i] = (1 + momentum) * hi_new - momentum * hi;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(rms_prop_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"KERNEL_ARG_DTYPE rms_decay,",    // NOLINT
"KERNEL_ARG_DTYPE delta,",    // NOLINT
"KERNEL_ARG_DTYPE local_rate) {",    // NOLINT
"for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {",    // NOLINT
"Dtype gi = g[i];",    // NOLINT
"Dtype hi = h[i] = rms_decay * h[i] + (1 - rms_decay) * gi * gi;",    // NOLINT
"g[i] = local_rate * g[i] / (sqrt(hi) + delta);",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(sgd_update,Dtype)(int_tp N, __global Dtype* g,",    // NOLINT
"__global Dtype* h,",    // NOLINT
"KERNEL_ARG_DTYPE momentum,",    // NOLINT
"KERNEL_ARG_DTYPE local_rate) {",    // NOLINT
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
    "bbox_util",   // NOLINT
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
    "gemm",   // NOLINT
    "im2col",   // NOLINT
    "im2col_nd",   // NOLINT
    "lrn",   // NOLINT
    "lstm_unit",   // NOLINT
    "math",   // NOLINT
    "matvec_mul",   // NOLINT
    "mergecrop",   // NOLINT
    "normalize_layer",   // NOLINT
    "permute_layer",   // NOLINT
    "pooling",   // NOLINT
    "pooling_nd",   // NOLINT
    "pooling_sk",   // NOLINT
    "reorg",   // NOLINT
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
  ss << "#define as_Dtype as_float" << "\n\n";  // NOLINT
  ss << "#define as_Dtype2 as_float2" << "\n\n";  // NOLINT
  ss << "#define as_Dtype4 as_float4" << "\n\n";  // NOLINT
  ss << "#define as_Dtype8 as_float8" << "\n\n";  // NOLINT
  ss << "#define as_Dtype16 as_float16" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_FLOAT" << "\n\n";  // NOLINT
  ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
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
  ss << "#undef as_Dtype" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype2" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype4" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype8" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype16" << "\n\n";  // NOLINT
  ss << "#define as_Dtype as_double" << "\n\n";  // NOLINT
  ss << "#define as_Dtype2 as_double2" << "\n\n";  // NOLINT
  ss << "#define as_Dtype4 as_double4" << "\n\n";  // NOLINT
  ss << "#define as_Dtype8 as_double8" << "\n\n";  // NOLINT
  ss << "#define as_Dtype16 as_double16" << "\n\n";  // NOLINT
  ss << "#undef TYPE" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_DOUBLE" << "\n\n";  // NOLINT
  ss << "#undef KERNEL_ARG_DTYPE" << "\n\n";  // NOLINT
  ss << "#define KERNEL_ARG_DTYPE double" << "\n\n";  // NOLINT
  ss << "#undef DTYPE_MAX" << "\n\n";  // NOLINT
  ss << "#undef DTYPE_MIN" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] != std::string("fft")) {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  ss << "#endif  // DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  ss << "#if defined(HALF_SUPPORT_AVAILABLE) && defined(HAS_HALF_SUPPORT)" << "\n\n";  // NOLINT
  ss << "#undef Dtype" << "\n\n";  // NOLINT
  ss << "#undef Dtype2" << "\n\n";  // NOLINT
  ss << "#undef Dtype4" << "\n\n";  // NOLINT
  ss << "#undef Dtype8" << "\n\n";  // NOLINT
  ss << "#undef Dtype16" << "\n\n";  // NOLINT
  ss << "#define Dtype half" << "\n\n";  // NOLINT
  ss << "#define Dtype2 half2" << "\n\n";  // NOLINT
  ss << "#define Dtype4 half4" << "\n\n";  // NOLINT
  ss << "#define Dtype8 half8" << "\n\n";  // NOLINT
  ss << "#define Dtype16 half16" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype2" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype4" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype8" << "\n\n";  // NOLINT
  ss << "#undef as_Dtype16" << "\n\n";  // NOLINT
  ss << "#define as_Dtype as_half" << "\n\n";  // NOLINT
  ss << "#define as_Dtype2 as_half2" << "\n\n";  // NOLINT
  ss << "#define as_Dtype4 as_half4" << "\n\n";  // NOLINT
  ss << "#define as_Dtype8 as_half8" << "\n\n";  // NOLINT
  ss << "#define as_Dtype16 as_half16" << "\n\n";  // NOLINT
  ss << "#undef TYPE" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_HALF" << "\n\n";  // NOLINT
  ss << "#undef KERNEL_ARG_DTYPE" << "\n\n";  // NOLINT
  ss << "#undef DTYPE_MAX" << "\n\n";  // NOLINT
  ss << "#undef DTYPE_MIN" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MAX HALF_MAX" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MIN HALF_MIN" << "\n\n";  // NOLINT
  ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] != std::string("fft")) {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  ss << "#endif  // HALF_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  std::string kernel_string = ss.str();
  const char* kernel_program = kernel_string.c_str();
  string options;
#ifdef USE_FFT
  options = " -DFFT ";
#endif
#ifdef HAS_HALF_SUPPORT
  options += " -DHAS_HALF_SUPPORT ";
#endif
  bool is_beignet = ctx->devices()[0].opencl_c_version().find("beignet")
                    != std::string::npos;
  if (!is_beignet)
    options += (" -cl-no-subgroup-ifp ");
  ctx->build_options(options);
  viennacl::ocl::program &program = ctx->add_program(kernel_program,
      "kernel_program");
  return program;
}
template<typename Dtype>
viennacl::ocl::program & submit_conv_spatial_program(
viennacl::ocl::context *ctx, string name, string options) {
  static const char* float_core_defines =
  "#define Dtype float\n"
  "#define Dtype2 float2\n"
  "#define Dtype4 float4\n"
  "#define Dtype8 float8\n"
  "#define Dtype16 float16\n"
  "#define as_Dtype as_float\n"
  "#define as_Dtype2 as_float2\n"
  "#define as_Dtype4 as_float4\n"
  "#define as_Dtype8 as_float8\n"
  "#define as_Dtype16 as_float16\n"
  "#define TYPE TYPE_FLOAT\n"
  "#define DTYPE_MAX FLT_MAX\n"
  "#define DTYPE_MIN FLT_MIN\n"
  "#define KERNEL_ARG_DTYPE float\n";
  static const char* half_core_defines =
  "#define Dtype half\n"
  "#define Dtype2 half2\n"
  "#define Dtype4 half4\n"
  "#define Dtype8 half8\n"
  "#define Dtype16 half16\n"
  "#define as_Dtype as_half\n"
  "#define as_Dtype2 as_half2\n"
  "#define as_Dtype4 as_half4\n"
  "#define as_Dtype8 as_half8\n"
  "#define as_Dtype16 as_half16\n"
  "#define TYPE TYPE_HALF\n"
  "#define DTYPE_MAX HALF_MAX\n"
  "#define DTYPE_MIN HALF_MIN\n"
  "#define KERNEL_ARG_DTYPE float\n";
  std::stringstream ss;
  if (std::is_same<Dtype, float>::value) {
    ss << float_core_defines;
  } else {
    ss << half_core_defines;
  }
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
  bool is_beignet = ctx->devices()[0].opencl_c_version().find("beignet")
                    != std::string::npos;
  if (!is_beignet)
    options += (" -cl-no-subgroup-ifp ");
  ctx->build_options(options);
  viennacl::ocl::program &program = ctx->add_program(ss.str(), name);
  return program;
}
template
viennacl::ocl::program & submit_conv_spatial_program<half>(
viennacl::ocl::context *ctx, string name, string options);
template
viennacl::ocl::program & submit_conv_spatial_program<float>(
viennacl::ocl::context *ctx, string name, string options);
template
viennacl::ocl::program & submit_conv_spatial_program<double>(
viennacl::ocl::context *ctx, string name, string options);
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
    ss << "#define as_Dtype as_float" << "\n\n";  // NOLINT
    ss << "#define as_Dtype2 as_float2" << "\n\n";  // NOLINT
    ss << "#define as_Dtype4 as_float4" << "\n\n";  // NOLINT
    ss << "#define as_Dtype8 as_float8" << "\n\n";  // NOLINT
    ss << "#define as_Dtype16 as_float16" << "\n\n";  // NOLINT
    ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  } else if (std::is_same<Dtype, double>::value) {
    ss << "#ifdef DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
    ss << "#define Dtype double" << "\n\n";  // NOLINT
    ss << "#define Dtype2 double2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 double4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 double8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 double16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_DOUBLE" << "\n\n";  // NOLINT
    ss << "#define as_Dtype as_double" << "\n\n";  // NOLINT
    ss << "#define as_Dtype2 as_double2" << "\n\n";  // NOLINT
    ss << "#define as_Dtype4 as_double4" << "\n\n";  // NOLINT
    ss << "#define as_Dtype8 as_double8" << "\n\n";  // NOLINT
    ss << "#define as_Dtype16 as_double16" << "\n\n";  // NOLINT
    ss << "#define KERNEL_ARG_DTYPE double" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  } else {
    ss << "#if defined(HALF_SUPPORT_AVAILABLE) && defined(HAS_HALF_SUPPORT)" << "\n\n";  // NOLINT
    ss << "#define Dtype half" << "\n\n";  // NOLINT
    ss << "#define Dtype2 half2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 half4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 half8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 half16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_HALF" << "\n\n";  // NOLINT
    ss << "#define as_Dtype as_half" << "\n\n";  // NOLINT
    ss << "#define as_Dtype2 as_half2" << "\n\n";  // NOLINT
    ss << "#define as_Dtype4 as_half4" << "\n\n";  // NOLINT
    ss << "#define as_Dtype8 as_half8" << "\n\n";  // NOLINT
    ss << "#define as_Dtype16 as_half16" << "\n\n";  // NOLINT
    ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MAX HALF_MAX" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MIN HALF_MIN" << "\n\n";  // NOLINT
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
template std::string getKernelBundleSource<half>(int index);
template std::string getKernelBundleSource<float>(int index);
template std::string getKernelBundleSource<double>(int index);
std::string getKernelBundleName(int index) {
  return cl_kernel_names[index];
}
}  // namespace caffe
#endif
