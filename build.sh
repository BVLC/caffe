#! /bin/bash

set -e

mkdir -p build && cd build && \
  cmake -DUSE_OPENCV=ON -DUSE_LEVELDB=ON -DCUDA_ARCH_NAME=Pascal .. && \
  make -j"$(nproc)" || exit 1

echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
