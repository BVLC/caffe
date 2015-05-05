#! /bin/bash
# This script converts all OpenCL Kernels to C++ char strings and defines the helper function to
# load the kernels to ViennaCL/OpenCL contexts.
# Outputs (overwrites): cl_kernels.hpp and cl_kernels.cpp

CL_HEADERDIR="src/caffe/greentea/cl_headers/*.cl"
CL_KERNELDIR="src/caffe/greentea/cl_kernels/*.cl"
HEADER='include/caffe/greentea/cl_kernels.hpp'
INCHEADER='caffe/greentea/cl_kernels.hpp'
SOURCE='src/caffe/greentea/cl_kernels.cpp'

echo "// AUTOMATICALLY GENERATED FILE, DO NOT EDIT" > $HEADER
echo "// AUTOMATICALLY GENERATED FILE, DO NOT EDIT" > $SOURCE
echo "#ifdef USE_GREENTEA" >> $HEADER
echo "#ifdef USE_GREENTEA" >> $SOURCE

echo "#ifndef GREENTEA_CL_KERNELS_HPP_" >> $HEADER
echo "#define GREENTEA_CL_KERNELS_HPP_" >> $HEADER
echo "#include \"caffe/greentea/greentea.hpp\"" >> $HEADER
echo "#include \"viennacl/backend/opencl.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/context.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/device.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/platform.hpp\"" >> $HEADER
echo "#include \"viennacl/ocl/backend.hpp\"" >> $HEADER
echo "namespace caffe {" >> $HEADER
echo "#include \"$INCHEADER\"" >> $SOURCE
echo "#include <sstream>" >> $SOURCE
echo "#include <string>" >> $SOURCE
echo "namespace caffe {" >> $SOURCE

echo "viennacl::ocl::program & RegisterKernels(viennacl::ocl::context &ctx);" >> $HEADER
echo "}" >> $HEADER
echo "#endif" >> $HEADER

shopt -s nullglob
for CL_KERNEL in $CL_HEADERDIR
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "std::string $CL_KERNEL_NAME = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";" >> $SOURCE
done

shopt -s nullglob
for CL_KERNEL in $CL_HEADERDIR $CL_KERNELDIR
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "std::string ${CL_KERNEL_NAME}_float = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";" >> $SOURCE
done

shopt -s nullglob
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_STR=`cat $CL_KERNEL`
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo -n "std::string ${CL_KERNEL_NAME}_double = \"" >> $SOURCE
	echo -n "$CL_KERNEL_STR" | sed -e ':a;N;$!ba;s/\n/\\n/g' | sed -e 's/\"/\\"/g' >> $SOURCE
	echo "\";" >> $SOURCE
done



echo "viennacl::ocl::program & RegisterKernels(viennacl::ocl::context &ctx) {" >> $SOURCE
echo "  std::stringstream ss;" >> $SOURCE

shopt -s nullglob
for CL_KERNEL in $CL_HEADERDIR
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << $CL_KERNEL_NAME << \"\\n\\n\";" >> $SOURCE
done

shopt -s nullglob
echo "  ss << \"#define Dtype float\" << \"\\n\\n\";" >> $SOURCE
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << ${CL_KERNEL_NAME}_float << \"\\n\\n\";" >> $SOURCE
done

shopt -s nullglob
echo "  ss << \"#ifdef DOUBLE_SUPPORT_AVAILABLE\" << \"\\n\\n\";" >> $SOURCE
echo "  ss << \"#undef Dtype\" << \"\\n\\n\";" >> $SOURCE
echo "  ss << \"#define Dtype double\" << \"\\n\\n\";" >> $SOURCE
for CL_KERNEL in $CL_KERNELDIR
do
	CL_KERNEL_NAME=`echo $CL_KERNEL`
	CL_KERNEL_NAME="${CL_KERNEL_NAME##*/}"
	CL_KERNEL_NAME="${CL_KERNEL_NAME%.cl}"
	echo "  ss << ${CL_KERNEL_NAME}_double << \"\\n\\n\";" >> $SOURCE
done
echo "  ss << \"#endif\" << \"\\n\\n\";" >> $SOURCE

echo "  std::string kernel_string = ss.str();" >> $SOURCE
echo "  const char* kernel_program = kernel_string.c_str();" >> $SOURCE
echo "  viennacl::ocl::program &program = ctx.add_program(kernel_program,\"kernel_program\");" >> $SOURCE
echo "  return program;" >> $SOURCE
echo "}" >> $SOURCE
echo "}" >> $SOURCE

echo "#endif" >> $HEADER
echo "#endif" >> $SOURCE
