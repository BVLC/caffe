#ifndef CAFFE_UTIL_HALIDE_H_
#define CAFFE_UTIL_HALIDE_H_

#include <HalideRuntimeCuda.h>

#include "caffe/blob.hpp"


namespace caffe {

// the data/diff param is ugly
template <typename Dtype>
buffer_t HalideWrapBlob(const Blob<Dtype>& blob, bool data = true) {
  CHECK_LE(blob.num_axes(), 4) << "Halide does not support blobs with more "
    "than four axes";
  buffer_t buf = {};
  for (int i = 0; i < blob.num_axes(); ++i) {
    // XXX reversing dimensions here -- should I?
    buf.extent[blob.num_axes() - 1 - i] = blob.shape(i);
    buf.stride[blob.num_axes() - 1 - i] = blob.count(i + 1);
  }
  buf.elem_size = sizeof(Dtype);
  shared_ptr<SyncedMemory> mem;
  if (data) {
    mem = blob.data();
  } else {
    mem = blob.diff();
  }
  // not using mutable_ accessors since we don't want to cause a copy here
  // XXX this will force unneeded GPU -> CPU -> GPU copies
  buf.host = const_cast<uint8_t*>(
      reinterpret_cast<const uint8_t*>(mem->cpu_data()));
  // XXX this will force a GPU allocation
  CHECK_EQ(halide_cuda_wrap_device_ptr(NULL, &buf,
      reinterpret_cast<uintptr_t>(mem->gpu_data())), 0);
  buf.host_dirty = mem->head() == SyncedMemory::HEAD_AT_CPU;
  buf.dev_dirty = mem->head() == SyncedMemory::HEAD_AT_GPU;
  return buf;
}

// XXX XXX
template <typename Dtype>
buffer_t HalideWrapBlobFlat(const Blob<Dtype>& blob, bool data = true) {
  buffer_t buf = {};
  buf.extent[0] = blob.count();
  buf.stride[0] = 1;
  buf.elem_size = sizeof(Dtype);
  shared_ptr<SyncedMemory> mem;
  if (data) {
    mem = blob.data();
  } else {
    mem = blob.diff();
  }
  // not using mutable_ accessors since we don't want to cause a copy here
  // XXX this will force unneeded GPU -> CPU -> GPU copies
  buf.host = const_cast<uint8_t*>(
      reinterpret_cast<const uint8_t*>(mem->cpu_data()));
  // XXX this will force a GPU allocation
  CHECK_EQ(halide_cuda_wrap_device_ptr(NULL, &buf,
      reinterpret_cast<uintptr_t>(mem->gpu_data())), 0);
  buf.host_dirty = mem->head() == SyncedMemory::HEAD_AT_CPU;
  buf.dev_dirty = mem->head() == SyncedMemory::HEAD_AT_GPU;
  return buf;
}

template <typename Dtype>
void HalideSyncBlob(const buffer_t& buf, Blob<Dtype>* blob,
    bool data = true) {
  CHECK(!(buf.host_dirty && buf.dev_dirty));
  shared_ptr<SyncedMemory> mem;
  if (data) {
    mem = blob->data();
  } else {
    mem = blob->diff();
  }
  if (buf.host_dirty) {
    mem->set_head(SyncedMemory::HEAD_AT_CPU);
  } else if (buf.dev_dirty) {
    mem->set_head(SyncedMemory::HEAD_AT_GPU);
  } else {
    mem->set_head(SyncedMemory::SYNCED);
  }
}

}  // namespace caffe

#endif

