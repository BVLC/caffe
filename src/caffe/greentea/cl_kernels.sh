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
echo "namespace caffe {" >> $SOURCE

echo "viennacl::ocl::program & RegisterKernels(viennacl::ocl::context *ctx);" >> $HEADER
echo "}" >> $HEADER
echo "#endif" >> $HEADER

echo "#ifdef USE_INDEX_64" >> $SOURCE
shopt -s nullglob
for CL_KERNEL in "${CL_HEADERS_64[@]}"
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "static std::string $CL_KERNEL_NAME = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
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
	echo -n "static std::string $CL_KERNEL_NAME = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";  // NOLINT" >> $SOURCE
done
echo "#endif" >> $SOURCE

shopt -s nullglob
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "static std::string ${CL_KERNEL_NAME}_float = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";  // NOLINT" >> $SOURCE
done

shopt -s nullglob
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "static std::string ${CL_KERNEL_NAME}_double = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";  // NOLINT" >> $SOURCE
done

echo "viennacl::ocl::program & RegisterKernels(viennacl::ocl::context *ctx) {" >> $SOURCE
echo "  std::stringstream ss;" >> $SOURCE

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
echo "  ss << \"#define Dtype float\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define TYPE TYPE_FLOAT\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << ${CL_KERNEL_NAME}_float << \"\\n\\n\";  // NOLINT" >> $SOURCE
done

shopt -s nullglob
echo "  ss << \"#ifdef DOUBLE_SUPPORT_AVAILABLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#undef Dtype\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define Dtype double\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#undef TYPE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
echo "  ss << \"#define TYPE TYPE_DOUBLE\" << \"\\n\\n\";  // NOLINT" >> $SOURCE
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << ${CL_KERNEL_NAME}_double << \"\\n\\n\";  // NOLINT" >> $SOURCE
done
echo "  ss << \"#endif\" << \"\\n\\n\";" >> $SOURCE

echo "  std::string kernel_string = ss.str();" >> $SOURCE
echo "  const char* kernel_program = kernel_string.c_str();" >> $SOURCE
echo "  // ctx->build_options(\"-cl-fast-relaxed-math -cl-mad-enable\");" >> $SOURCE
echo "  viennacl::ocl::program &program = ctx->add_program(kernel_program," >> $SOURCE
echo "      \"kernel_program\");" >> $SOURCE
echo "  return program;" >> $SOURCE
echo "}" >> $SOURCE
echo "}  // namespace caffe" >> $SOURCE

echo "#endif" >> $HEADER
echo "#endif" >> $SOURCE
