#! /bin/bash
# This script converts all OpenCL Kernels to C++ char strings and defines the helper function to
# load the kernels to ViennaCL/OpenCL contexts.
# Outputs (overwrites): cl_kernels.hpp and cl_kernels.cpp

declare -a CL_HEADERS_32=("src/caffe/greentea/cl_headers/header.cl" "src/caffe/greentea/cl_headers/definitions_32.cl")
declare -a CL_HEADERS_64=("src/caffe/greentea/cl_headers/header.cl" "src/caffe/greentea/cl_headers/definitions_64.cl")
CL_KERNELDIR="src/caffe/greentea/cl_kernels/*.cl"
HEADER='include/caffe/greentea/cl_kernels.hpp'
INCHEADER='caffe/greentea/cl_kernels.hpp'
SOURCE='src/caffe/greentea/cl_kernels.cpp'

echo "// AUTOMATICALLY GENERATED FILE, DO NOT EDIT" > $HEADER
echo "// AUTOMATICALLY GENERATED FILE, DO NOT EDIT" > $SOURCE
echo "#include <string>" >> $HEADER
echo "#include \"caffe/common.hpp\"" >> $HEADER
echo "#ifdef USE_GREENTEA" >> $HEADER
echo "#include \"caffe/common.hpp\"" >> $SOURCE
echo "#ifdef USE_GREENTEA" >> $SOURCE

echo "#ifndef GREENTEA_CL_KERNELS_HPP_" >> $HEADER
echo "#define GREENTEA_CL_KERNELS_HPP_" >> $HEADER
echo "#include \"caffe/greentea/greentea.hpp\"" >> $HEADER
echo "#include \"viennacl/backend/opencl.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/backend.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/context.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/device.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/platform.hpp\"" >> $HEADER
echo "namespace caffe {" >> $HEADER
echo "#include \"$INCHEADER\"" >> $SOURCE
echo "#include <sstream>" >> $SOURCE
echo "#include <string>" >> $SOURCE
echo "#include <type_traits>" >> $SOURCE
echo "#include <vector>" >> $SOURCE

echo "#ifdef DISABLE_DOUBLE_SUPPORT" >> $SOURCE
echo "  #define DOUBLE_SUPPORT \"#define DISABLE_DOUBLE_SUPPORT\n\"" >> $SOURCE
echo "#else" >> $SOURCE
echo "  #define DOUBLE_SUPPORT \"#define ENABLE_DOUBLE_SUPPORT\n\"" >> $SOURCE
echo "#endif  // DISABLE_DOUBLE_SUPPORT" >> $SOURCE

echo "namespace caffe {" >> $SOURCE

echo "viennacl::ocl::program & RegisterCommonKernels(viennacl::ocl::context *ctx);" >> $HEADER
echo "template <typename Dtype>" >> $HEADER
echo "viennacl::ocl::program & RegisterKernels(viennacl::ocl::context *ctx);" >> $HEADER
echo "template <typename Dtype>" >> $HEADER
echo "viennacl::ocl::program & submit_conv_spatial_program(" >> $HEADER
echo "viennacl::ocl::context *ctx, string name, string options);" >> $HEADER
echo "std::string getKernelBundleName(int index);" >> $HEADER
echo "int getKernelBundleCount();" >> $HEADER
echo "template<typename Dtype>" >> $HEADER
echo "std::string getKernelBundleSource(int index);" >> $HEADER
echo "}  // namespace caffe" >> $HEADER
echo "#endif" >> $HEADER

echo "#ifdef USE_INDEX_64" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in "${CL_HEADERS_64[@]}"
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
    echo -n "static std::string $CL_KERNEL_NAME = DOUBLE_SUPPORT \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e 's/\\$/\\\\/g' | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";  // NOLINT" >> $SOURCE
done
echo "#else" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in "${CL_HEADERS_32[@]}"
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "static std::string $CL_KERNEL_NAME = DOUBLE_SUPPORT \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e 's/\\$/\\\\/g' | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";  // NOLINT" >> $SOURCE
done
echo "#endif" >> $SOURCE

TOTALCOUNTER=0
for CL_KERNEL in $CL_KERNELDIR
do
    TOTALCOUNTER=$((TOTALCOUNTER + 1))
done

COUNTER=0
echo "static std::vector<std::vector<std::string>> cl_kernels{" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in $CL_KERNELDIR
do
    COUNTER=$((COUNTER + 1))
    echo -n "    {" >> $SOURCE
    while read i; do
        echo -n "\"" >> $SOURCE
	    echo -n "$i" | sed -e 's/\\$/\\\\/g'| sed -e 's/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
        echo -e "\",    // NOLINT" >> $SOURCE
    done < ${CL_KERNEL}

    if (($COUNTER == $TOTALCOUNTER)) ; then
        echo "\"\"}   // NOLINT" >> $SOURCE
    else
        echo "\"\"},   // NOLINT" >> $SOURCE
    fi
done
echo "};" >> $SOURCE

COUNTER=0
echo "static std::string cl_kernel_names[] = {" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in $CL_KERNELDIR
do
    COUNTER=$((COUNTER + 1))
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"

	echo -n "    \"$CL_KERNEL_NAME\"" >> $SOURCE

    if (($COUNTER == $TOTALCOUNTER)) ; then
        echo "   // NOLINT" >> $SOURCE
    else
	    echo ",   // NOLINT" >> $SOURCE
    fi
done
echo "};" >> $SOURCE

echo "viennacl::ocl::program & RegisterCommonKernels(viennacl::ocl::context *ctx) {" >> $SOURCE
echo "  std::stringstream ss;" >> $SOURCE
echo "  for (int i = 0; i < cl_kernels.size(); ++i) {" >> $SOURCE
echo "    if (cl_kernel_names[i] == std::string(\"common\")) {" >> $SOURCE
echo "      for (int j = 0; j < cl_kernels[i].size(); ++j) {" >> $SOURCE
echo "        ss << cl_kernels[i][j] << \"\n\n\";" >> $SOURCE
echo "      }" >> $SOURCE
echo "    }" >> $SOURCE
echo "  }" >> $SOURCE
echo "  std::string kernel_string = ss.str();" >> $SOURCE
echo "  const char* kernel_program = kernel_string.c_str();" >> $SOURCE
echo "  string options;" >> $SOURCE
echo "  ctx->build_options(options);" >> $SOURCE
echo "  viennacl::ocl::program &program = ctx->add_program(kernel_program," >> $SOURCE
echo "      \"common_kernel_program\");" >> $SOURCE
echo "  return program;" >> $SOURCE
echo "}" >> $SOURCE

echo "template <typename Dtype>" >> $SOURCE
echo "viennacl::ocl::program & RegisterKernels(viennacl::ocl::context *ctx) {" >> $SOURCE
echo "  std::stringstream ss;" >> $SOURCE
echo "  std::stringstream int64_base_atomics;" >> $SOURCE
echo "  int64_base_atomics << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  int64_base_atomics << \"#if defined(cl_khr_int64_base_atomics)\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  int64_base_atomics << \"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  int64_base_atomics << \"#define ATOMICS_64_AVAILABLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  int64_base_atomics << \"#endif\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  if(ctx->devices()[0].extensions().find(\"cl_khr_int64_base_atomics\")!= std::string::npos) {" >> $SOURCE
echo "    header += int64_base_atomics.str();" >> $SOURCE
echo "  }" >> $SOURCE

echo "#ifdef USE_INDEX_64" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in "${CL_HEADERS_64[@]}"
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << $CL_KERNEL_NAME << \"\\n\\n\";  // NOLINT" >> $SOURCE
done
echo "#else" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in "${CL_HEADERS_32[@]}"
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << $CL_KERNEL_NAME << \"\\n\\n\";  // NOLINT" >> $SOURCE
done
echo "#endif" >> $SOURCE

shopt -s nullglob
echo "  if (std::is_same<Dtype, float>::value) { // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype float\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype2 float2\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype4 float4\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype8 float8\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype16 float16\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype as_float\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype2 as_float2\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype4 as_float4\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype8 as_float8\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype16 as_float16\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define TYPE TYPE_FLOAT\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define KERNEL_ARG_DTYPE float\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define DTYPE_MAX FLT_MAX\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define DTYPE_MIN FLT_MIN\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  for (int i = 0; i < cl_kernels.size(); ++i) {" >> $SOURCE
echo "    for (int j = 0; j < cl_kernels[i].size(); ++j) {" >> $SOURCE
echo "      ss << cl_kernels[i][j] << \"\n\n\";" >> $SOURCE
echo "    }" >> $SOURCE
echo "  }" >> $SOURCE
echo "  }" >> $SOURCE

echo "  if (std::is_same<Dtype, double>::value) { // NOLINT" >> $SOURCE
echo "  ss << \"#ifdef DOUBLE_SUPPORT_AVAILABLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype double\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype2 double2\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype4 double4\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype8 double8\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype16 double16\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype as_double\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype2 as_double2\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype4 as_double4\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype8 as_double8\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype16 as_double16\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define TYPE TYPE_DOUBLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define KERNEL_ARG_DTYPE double\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define DTYPE_MAX FLT_MAX\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define DTYPE_MIN FLT_MIN\" << \"\\n\\n\";  // NOLINT" >> $SOURCE

shopt -s nullglob
echo "  for (int i = 0; i < cl_kernels.size(); ++i) {" >> $SOURCE
echo "    if (cl_kernel_names[i] != std::string(\"fft\")) {" >> $SOURCE
echo "      for (int j = 0; j < cl_kernels[i].size(); ++j) {" >> $SOURCE
echo "        ss << cl_kernels[i][j] << \"\n\n\";" >> $SOURCE
echo "      }" >> $SOURCE
echo "    }" >> $SOURCE
echo "  }" >> $SOURCE
echo "  ss << \"#endif  // DOUBLE_SUPPORT_AVAILABLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  }" >> $SOURCE

echo "  if (std::is_same<Dtype, half_float::half>::value) { // NOLINT" >> $SOURCE
echo "  ss << \"#if defined(HALF_SUPPORT_AVAILABLE) && defined(HAS_HALF_SUPPORT)\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  if (ctx->devices()[0].extensions().find(\"cl_intel_subgroups\")!= std::string::npos   // NOLINT" >> $SOURCE
echo "      && ctx->devices()[0].extensions().find(\"cl_intel_subgroups_short\")== std::string::npos) { // NOLINT" >> $SOURCE
echo "    std::cerr << \"Fatal error: Intel iGPU device found but doesn\'t support cl_intel_subgroups_short.\" << std::endl; // NOLINT" >> $SOURCE
echo "    std::cerr << \"Please upgrade the GPU driver and OpenCL SDK.\" << std::endl; // NOLINT" >> $SOURCE
echo "    std::cerr << \"For iGPU platforms before Gen9, fp16 is not supported.\" << std::endl; // NOLINT" >> $SOURCE
echo "    exit(-1);" >> $SOURCE
echo "  }" >> $SOURCE
echo "  ss << \"#define Dtype half\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype2 half2\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype4 half4\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype8 half8\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype16 half16\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype as_half\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype2 as_half2\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype4 as_half4\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype8 as_half8\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define as_Dtype16 as_half16\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define TYPE TYPE_HALF\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define DTYPE_MAX HALF_MAX\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define DTYPE_MIN HALF_MIN\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define KERNEL_ARG_DTYPE float\" << \"\\n\\n\";  // NOLINT" >> $SOURCE

shopt -s nullglob
echo "  for (int i = 0; i < cl_kernels.size(); ++i) {" >> $SOURCE
echo "    if (cl_kernel_names[i] != std::string(\"fft\")) {" >> $SOURCE
echo "      for (int j = 0; j < cl_kernels[i].size(); ++j) {" >> $SOURCE
echo "        ss << cl_kernels[i][j] << \"\n\n\";" >> $SOURCE
echo "      }" >> $SOURCE
echo "    }" >> $SOURCE
echo "  }" >> $SOURCE
echo "  ss << \"#endif  // HALF_SUPPORT_AVAILABLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  }" >> $SOURCE

echo "  std::string kernel_string = ss.str();" >> $SOURCE
echo "  const char* kernel_program = kernel_string.c_str();" >> $SOURCE
echo "  string options;" >> $SOURCE
echo "#ifdef USE_FFT" >> $SOURCE
echo "  options = \" -DFFT \";" >> $SOURCE
echo "#endif" >> $SOURCE
echo "#ifdef HAS_HALF_SUPPORT" >> $SOURCE
echo "  options += \" -DHAS_HALF_SUPPORT \";" >> $SOURCE
echo "#endif" >> $SOURCE
echo "  if(ctx->devices()[0].extensions().find(\"cl_intel_subgroups\")!= std::string::npos) {" >> $SOURCE
echo "    options += \" -DHAS_INTEL_SUBGROUPS \";" >> $SOURCE
echo "  }" >> $SOURCE
echo "  bool is_beignet = ctx->devices()[0].opencl_c_version().find(\"beignet\")" >> $SOURCE
echo "                    != std::string::npos;" >> $SOURCE
echo "  if (!is_beignet)" >> $SOURCE
echo "    options += (\" -cl-no-subgroup-ifp \");" >> $SOURCE
echo "  ctx->build_options(options);" >> $SOURCE
echo "  viennacl::ocl::program &program = ctx->add_program(kernel_program," >> $SOURCE
echo "      \"kernel_program\");" >> $SOURCE
echo "  return program;" >> $SOURCE
echo "}" >> $SOURCE
echo "#ifdef HAS_HALF_SUPPORT" >> $SOURCE
echo "template" >> $SOURCE
echo "viennacl::ocl::program & RegisterKernels<half>(viennacl::ocl::context *ctx);" >> $SOURCE
echo "#endif" >> $SOURCE
echo "template" >> $SOURCE
echo "viennacl::ocl::program & RegisterKernels<float>(viennacl::ocl::context *ctx);" >> $SOURCE
echo "template" >> $SOURCE
echo "viennacl::ocl::program & RegisterKernels<double>(viennacl::ocl::context *ctx);" >> $SOURCE
echo "template<typename Dtype>" >> $SOURCE
echo "viennacl::ocl::program & submit_conv_spatial_program(" >> $SOURCE
echo "viennacl::ocl::context *ctx, string name, string options) {" >> $SOURCE
echo "  if (ctx->has_program(name)) {" >> $SOURCE
echo "    viennacl::ocl::program &p = ctx->get_program(name);" >> $SOURCE
echo "    return p;" >> $SOURCE
echo "  }" >> $SOURCE
echo "  static const char* float_core_defines =" >> $SOURCE
echo "  \"#define Dtype float\n\"" >> $SOURCE
echo "  \"#define Dtype2 float2\n\"" >> $SOURCE
echo "  \"#define Dtype4 float4\n\"" >> $SOURCE
echo "  \"#define Dtype8 float8\n\"" >> $SOURCE
echo "  \"#define Dtype16 float16\n\"" >> $SOURCE
echo "  \"#define as_Dtype as_float\n\"" >> $SOURCE
echo "  \"#define as_Dtype2 as_float2\n\"" >> $SOURCE
echo "  \"#define as_Dtype4 as_float4\n\"" >> $SOURCE
echo "  \"#define as_Dtype8 as_float8\n\"" >> $SOURCE
echo "  \"#define as_Dtype16 as_float16\n\"" >> $SOURCE
echo "  \"#define TYPE TYPE_FLOAT\n\"" >> $SOURCE
echo "  \"#define DTYPE_MAX FLT_MAX\n\"" >> $SOURCE
echo "  \"#define DTYPE_MIN FLT_MIN\n\"" >> $SOURCE
echo "  \"#define KERNEL_ARG_DTYPE float\n\";" >> $SOURCE

echo "  static const char* half_core_defines =" >> $SOURCE
echo "  \"#define Dtype half\n\"" >> $SOURCE
echo "  \"#define Dtype2 half2\n\"" >> $SOURCE
echo "  \"#define Dtype4 half4\n\"" >> $SOURCE
echo "  \"#define Dtype8 half8\n\"" >> $SOURCE
echo "  \"#define Dtype16 half16\n\"" >> $SOURCE
echo "  \"#define as_Dtype as_half\n\"" >> $SOURCE
echo "  \"#define as_Dtype2 as_half2\n\"" >> $SOURCE
echo "  \"#define as_Dtype4 as_half4\n\"" >> $SOURCE
echo "  \"#define as_Dtype8 as_half8\n\"" >> $SOURCE
echo "  \"#define as_Dtype16 as_half16\n\"" >> $SOURCE
echo "  \"#define TYPE TYPE_HALF\n\"" >> $SOURCE
echo "  \"#define DTYPE_MAX HALF_MAX\n\"" >> $SOURCE
echo "  \"#define DTYPE_MIN HALF_MIN\n\"" >> $SOURCE
echo "  \"#define KERNEL_ARG_DTYPE float\n\";" >> $SOURCE

echo "  std::stringstream ss;" >> $SOURCE
echo "  if (std::is_same<Dtype, float>::value) {" >> $SOURCE
echo "    ss << float_core_defines;" >> $SOURCE
echo "  } else {" >> $SOURCE
echo "    ss << half_core_defines;" >> $SOURCE
echo "  }" >> $SOURCE
echo "#ifdef USE_INDEX_64" >> $SOURCE
echo "  ss << header + \"\n\";" >> $SOURCE
echo "  ss << definitions_64 + \"\n\";" >> $SOURCE
echo "#else" >> $SOURCE
echo "  ss << header + \"\n\";" >> $SOURCE
echo "  ss << definitions_32 + \"\n\";" >> $SOURCE
echo "#endif" >> $SOURCE
echo "  for (int i = 0; i < cl_kernels.size(); ++i) {" >> $SOURCE
echo "    if (cl_kernel_names[i] == \"conv_layer_spatial\") {" >> $SOURCE
echo "      for (int j = 0; j < cl_kernels[i].size(); ++j) {" >> $SOURCE
echo "        ss << cl_kernels[i][j] << \"\n\n\";" >> $SOURCE
echo "      }" >> $SOURCE
echo "    }" >> $SOURCE
echo "  }" >> $SOURCE
echo "  bool is_beignet = ctx->devices()[0].opencl_c_version().find(\"beignet\")" >> $SOURCE
echo "                    != std::string::npos;" >> $SOURCE
echo "  if (!is_beignet)" >> $SOURCE
echo "    options += (\" -cl-no-subgroup-ifp \");" >> $SOURCE
echo "  if(ctx->devices()[0].extensions().find(\"cl_intel_subgroups\")!= std::string::npos) {" >> $SOURCE
echo "    options += \" -DHAS_INTEL_SUBGROUPS \";" >> $SOURCE
echo "  }" >> $SOURCE
echo "  ctx->build_options(options);" >> $SOURCE
echo "  viennacl::ocl::program &program = ctx->add_program(ss.str(), name);" >> $SOURCE
echo "  return program;" >> $SOURCE
echo "}" >> $SOURCE

echo "template" >> $SOURCE
echo "viennacl::ocl::program & submit_conv_spatial_program<half>(" >> $SOURCE
echo "viennacl::ocl::context *ctx, string name, string options);" >> $SOURCE
echo "template" >> $SOURCE
echo "viennacl::ocl::program & submit_conv_spatial_program<float>(" >> $SOURCE
echo "viennacl::ocl::context *ctx, string name, string options);" >> $SOURCE
echo "template" >> $SOURCE
echo "viennacl::ocl::program & submit_conv_spatial_program<double>(" >> $SOURCE
echo "viennacl::ocl::context *ctx, string name, string options);" >> $SOURCE
echo "int getKernelBundleCount() {" >> $SOURCE
echo "  return cl_kernels.size();" >> $SOURCE
echo "}" >> $SOURCE
echo "template<typename Dtype>" >> $SOURCE
echo "std::string getKernelBundleSource(int index) {" >> $SOURCE
echo "  std::stringstream ss;" >> $SOURCE
echo "#ifdef USE_INDEX_64" >> $SOURCE
echo "  ss << header << \"\n\n\";  // NOLINT" >> $SOURCE
echo "  ss << definitions_64 << \"\n\n\";  // NOLINT" >> $SOURCE
echo "#else" >> $SOURCE
echo "  ss << header << \"\n\n\";  // NOLINT" >> $SOURCE
echo "  ss << definitions_32 << \"\n\n\";  // NOLINT" >> $SOURCE
echo "#endif" >> $SOURCE
echo "  if (std::is_same<Dtype, float>::value) {" >> $SOURCE
echo "    ss << \"#define Dtype float\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype2 float2\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype4 float4\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype8 float8\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype16 float16\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define TYPE TYPE_FLOAT\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype as_float\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype2 as_float2\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype4 as_float4\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype8 as_float8\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype16 as_float16\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define KERNEL_ARG_DTYPE float\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define DTYPE_MAX FLT_MAX\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define DTYPE_MIN FLT_MIN\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "  } else if (std::is_same<Dtype, double>::value) {" >> $SOURCE
echo "    ss << \"#ifdef DOUBLE_SUPPORT_AVAILABLE\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype double\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype2 double2\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype4 double4\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype8 double8\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype16 double16\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define TYPE TYPE_DOUBLE\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype as_double\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype2 as_double2\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype4 as_double4\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype8 as_double8\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype16 as_double16\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define KERNEL_ARG_DTYPE double\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define DTYPE_MAX FLT_MAX\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define DTYPE_MIN FLT_MIN\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "  } else {" >> $SOURCE
echo "    ss << \"#if defined(HALF_SUPPORT_AVAILABLE) && defined(HAS_HALF_SUPPORT)\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype half\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype2 half2\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype4 half4\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype8 half8\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define Dtype16 half16\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define TYPE TYPE_HALF\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype as_half\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype2 as_half2\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype4 as_half4\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype8 as_half8\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define as_Dtype16 as_half16\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define KERNEL_ARG_DTYPE float\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define DTYPE_MAX HALF_MAX\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "    ss << \"#define DTYPE_MIN HALF_MIN\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "  }" >> $SOURCE
echo "  for (int j = 0; j < cl_kernels[index].size(); ++j) {" >> $SOURCE
echo "    ss << cl_kernels[index][j] << \"\n\n\";" >> $SOURCE
echo "  }" >> $SOURCE
echo "  if (std::is_same<Dtype, float>::value) {" >> $SOURCE
echo "  } else {" >> $SOURCE
echo "    ss << \"#endif\" << \"\n\n\";  // NOLINT" >> $SOURCE
echo "  }" >> $SOURCE
echo "  return ss.str();" >> $SOURCE
echo "}" >> $SOURCE
echo "template std::string getKernelBundleSource<half>(int index);" >> $SOURCE
echo "template std::string getKernelBundleSource<float>(int index);" >> $SOURCE
echo "template std::string getKernelBundleSource<double>(int index);" >> $SOURCE
echo "std::string getKernelBundleName(int index) {" >> $SOURCE
echo "  return cl_kernel_names[index];" >> $SOURCE
echo "}" >> $SOURCE

echo "}  // namespace caffe" >> $SOURCE

echo "#endif" >> $HEADER
echo "#endif" >> $SOURCE
