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
