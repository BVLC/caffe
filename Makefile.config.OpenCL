## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# OpenCL switch (uncomment to build with OpenCL support).
USE_OPENCL := 1
CPU_ONLY   := 1
OPENCL_DIR := /opt/AMDAPPSDK
CAFFE_OPENCL_VERSION := 0.1
CLBLAS_DIR := /usr/local

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0
