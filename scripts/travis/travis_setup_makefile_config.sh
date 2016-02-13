#!/bin/bash

set -e

mv Makefile.config.example Makefile.config

if $WITH_CUDA; then
  # Only generate compute_50.
  GENCODE="-gencode arch=compute_50,code=sm_50"
  GENCODE="$GENCODE -gencode arch=compute_50,code=compute_50"
  echo "CUDA_ARCH := $GENCODE" >> Makefile.config
fi

# Remove IO library settings from Makefile.config
# to avoid conflicts with CI configuration
sed -i -e '/USE_LMDB/d' Makefile.config
sed -i -e '/USE_LEVELDB/d' Makefile.config
sed -i -e '/USE_OPENCV/d' Makefile.config

cat << 'EOF' >> Makefile.config
# Travis' nvcc doesn't like newer boost versions
NVCCFLAGS := -Xcudafe --diag_suppress=cc_clobber_ignored -Xcudafe --diag_suppress=useless_using_declaration -Xcudafe --diag_suppress=set_but_not_used
ANACONDA_HOME := $(CONDA_DIR)
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		$(ANACONDA_HOME)/include/python2.7 \
		$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include
PYTHON_LIB := $(ANACONDA_HOME)/lib
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
WITH_PYTHON_LAYER := 1
USE_CUDNN := 0
EOF
