#!/bin/bash

set -e

mv Makefile.config.example Makefile.config

if $WITH_CUDA; then
  # Only generate compute_50.
  GENCODE="-gencode arch=compute_50,code=sm_50"
  GENCODE="$GENCODE -gencode arch=compute_50,code=compute_50"
  echo "CUDA_ARCH := $GENCODE" >> Makefile.config
fi
