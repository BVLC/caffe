#!/bin/bash

set -e

mv Makefile.config.example Makefile.config

if $WITH_CUDA; then
<<<<<<< HEAD
<<<<<<< HEAD
  # Only generate compute_50.
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  # Only generate compute_50.
=======
=======
>>>>>>> pod/caffe-merge
  # Remove default gencode set; only generate compute_50.
  sed -i 's/-gencode arch=.*\\//' Makefile.config
  sed -i 's/CUDA_ARCH :=//' Makefile.config
>>>>>>> origin/BVLC/parallel
=======
  # Only generate compute_50.
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
  GENCODE="-gencode arch=compute_50,code=sm_50"
  GENCODE="$GENCODE -gencode arch=compute_50,code=compute_50"
  echo "CUDA_ARCH := $GENCODE" >> Makefile.config
fi
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge

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
EOF
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
