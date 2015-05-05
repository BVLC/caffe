#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __constant
#define __local
#define get_global_id(x) 0
#define get_global_size(x) 0
#define get_local_id(x) 0
#define get_local_size(x) 0
#define FLT_MAX 0
#define FLT_MIN 0
#define cl_khr_fp64
#define cl_amd_fp64
#define DOUBLE_SUPPORT_AVAILABLE
#define CLK_LOCAL_MEM_FENCE
#define Dtype float
#define barrier(x)
#endif

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)


#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#endif
