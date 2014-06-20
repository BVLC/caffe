// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_SYNCEDMEM_FACTORY_HPP_
#define CAFFE_SYNCEDMEM_FACTORY_HPP_

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/opencl_syncedmem.hpp"

namespace caffe {

// The SyncedMemory factory function
AbstractSyncedMemory* GetSyncedMemory(const size_t size = 0);

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_FACTORY_HPP_
