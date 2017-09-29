/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include <mkl_blas.h>
  #include <mkl_service.h>
#endif

#include "boost/thread/mutex.hpp"
#include "caffe/common.hpp"

#include "caffe/multinode/mlsl.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif

#ifdef USE_MLSL
  if (mn::is_multinode()) {
    *ptr = mn::alloc(size ? size : 1, 64);
  } else {
#endif /* !USE_MLSL */

#ifdef USE_MKL
    *ptr = mkl_malloc(size ? size : 1, 64);
#else
   *ptr = malloc(size);
#endif

#ifdef USE_MLSL
  }
#endif /* USE_MLSL */

  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif

#ifdef USE_MLSL
  if (mn::is_multinode()) {
    mn::free(ptr);
  } else {
#endif /* !USE_MLSL */

#ifdef USE_MKL
    mkl_free(ptr);
#else
    free(ptr);
#endif

#ifdef USE_MLSL
  }
#endif /* USE_MLSL */

}

// Base class
struct PrvMemDescr {
  virtual ~PrvMemDescr() {}
  virtual void convert_from_prv(void* cpu_ptr) = 0;
  virtual void convert_to_prv(void* cpu_ptr) = 0;
  virtual void convert_from_other(shared_ptr<PrvMemDescr> other) = 0;
  virtual bool on_to_cpu() { return false; }
  virtual void* prv_ptr() = 0;
  // returns true for matching layouts
  virtual bool layout_compare(shared_ptr<PrvMemDescr> other) = 0;
  virtual size_t prv_count() = 0;
  virtual size_t prv_size() = 0;  // TODO: do we need both count() and size()?
  // This might help using prv_ptr_ by different accelerators/engines
  enum PrvDescrType {
    PRV_DESCR_MKL2017,
    PRV_DESCR_MKLDNN
  };
  virtual PrvDescrType get_descr_type() = 0;
};

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL),
        size_(0), head_(UNINITIALIZED), own_cpu_data_(false),
        cpu_malloc_use_cuda_(false), own_gpu_data_(false), own_prv_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL),
        size_(size), head_(UNINITIALIZED), own_cpu_data_(false),
        cpu_malloc_use_cuda_(false), own_gpu_data_(false), own_prv_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();
  void swap(shared_ptr<SyncedMemory> other);
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();

  const void* cpu_ptr() const { return cpu_ptr_; }

  shared_ptr<PrvMemDescr> prv_descriptor_;
  void set_prv_descriptor(shared_ptr<PrvMemDescr> descriptor, bool same_data);
  const void* prv_data();
  void* mutable_prv_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED,
                    HEAD_AT_PRV, SYNCED_PRV};
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  const size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  bool own_prv_data_;
  int gpu_device_;
  boost::mutex mtx;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
