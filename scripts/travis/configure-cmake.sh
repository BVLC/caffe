# CMake configuration

mkdir -p build
cd build

ARGS="-DCMAKE_BUILD_TYPE=Release -DBLAS=Open"

if $WITH_PYTHON3 ; then
  ARGS="$ARGS -Dpython_version=3"
fi

if $WITH_IO ; then
  ARGS="$ARGS -DUSE_OPENCV=On -DUSE_LMDB=On -DUSE_LEVELDB=On"
else
  ARGS="$ARGS -DUSE_OPENCV=Off -DUSE_LMDB=Off -DUSE_LEVELDB=Off"
fi

if $WITH_CUDA ; then
  # Only build SM50
  ARGS="$ARGS -DCPU_ONLY=Off -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=\"50\" -DCUDA_ARCH_PTX=\"\""
else
  ARGS="$ARGS -DCPU_ONLY=On"
fi

if $WITH_CUDNN ; then
  ARGS="$ARGS -DUSE_CUDNN=On"
else
  ARGS="$ARGS -DUSE_CUDNN=Off"
fi

cmake .. $ARGS

