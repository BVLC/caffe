#!/bin/bash

set -e

mv Makefile.config.example Makefile.config

if $WITH_CUDA; then
  # Remove default gencode set; only generate compute_50.
  sed -i 's/-gencode arch=.*\\//' Makefile.config
  sed -i 's/CUDA_ARCH :=//' Makefile.config
  GENCODE="-gencode arch=compute_50,code=sm_50"
  GENCODE="$GENCODE -gencode arch=compute_50,code=compute_50"
  echo "CUDA_ARCH := $GENCODE" >> Makefile.config
fi

TURBO_JPEG_INCLUDE=/opt/libjpeg-turbo/include
TURBO_JPEG_LIBRARY=/opt/libjpeg-turbo/lib64
echo "INCLUDE_DIRS += $TURBO_JPEG_INCLUDE" >> Makefile.config
echo "LIBRARY_DIRS += $TURBO_JPEG_LIBRARY" >> Makefile.config
echo "COMMON_FLAGS += -DWORKAROUND_PROTOBUF_FIND_EXTENSION" >> Makefile.config
