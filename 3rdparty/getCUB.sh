#!/usr/bin/env sh
VERSION="$1"
GITHUBURL="https://raw.githubusercontent.com/NVlabs/cub"
TARGETDIR=3rdparty
mkdir -p $TARGETDIR/cub
mkdir -p $TARGETDIR/cub/host
for CUH in \
  cub/util_allocator.cuh \
  cub/util_arch.cuh \
  cub/util_namespace.cuh \
  cub/util_debug.cuh \
  cub/host/mutex.cuh
do
  wget -q -N -O $TARGETDIR/$CUH $GITHUBURL/$VERSION/$CUH || exit 1
done
exit 0

