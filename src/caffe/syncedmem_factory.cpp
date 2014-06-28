// Copyright 2014 BVLC and contributors.

#include "caffe/syncedmem_factory.hpp"

namespace caffe {

AbstractSyncedMemory* GetSyncedMemory(const size_t size) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
  case Caffe::GPU:
    return new SyncedMemory(size);
#ifdef USE_OPENCL
  case Caffe::OPENCL_CPU:
  case Caffe::OPENCL_GPU:
    return new OpenCLSyncedMemory(size);
#endif
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<AbstractSyncedMemory*>(NULL);
  }
}

}  // namespace caffe

