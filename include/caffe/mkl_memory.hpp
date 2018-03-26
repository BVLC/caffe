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

#ifndef CAFFE_MKL_MEMORY_HPP_
#define CAFFE_MKL_MEMORY_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "mkl_dnn_cppwrapper.h"

namespace caffe {
template <typename Dtype>
struct MKLMemoryDescriptorBase : PrvMemDescr,
    boost::enable_shared_from_this<MKLMemoryDescriptorBase<Dtype> > {
  MKLMemoryDescriptorBase() : layout_usr(NULL), layout_int(NULL),
          convert_to_int(NULL), convert_from_int(NULL), convert_prv2prv(NULL),
          name("UNKNOWN"), internal_ptr(NULL) {}
  ~MKLMemoryDescriptorBase() {
    dnnLayoutDelete<Dtype>(layout_usr);
    dnnLayoutDelete<Dtype>(layout_int);

#ifdef USE_MLSL
    if (mn::is_multinode()) {
      if (internal_ptr != NULL) {
        mn::free((void*)internal_ptr);
        internal_ptr = NULL;
      }
    } else {
#endif /* !USE_MLSL */
      dnnReleaseBuffer<Dtype>(internal_ptr);
#ifdef USE_MLSL
    }
#endif /* USE_MLSL */

    dnnDelete<Dtype>(convert_to_int);
    dnnDelete<Dtype>(convert_from_int);
    dnnDelete<Dtype>(convert_prv2prv);
  }

  shared_ptr<MKLMemoryDescriptorBase<Dtype> > get_shared_ptr() {
    return this->shared_from_this();
  }

  dnnLayout_t layout_usr;
  dnnLayout_t layout_int;
  dnnPrimitive_t convert_to_int;
  dnnPrimitive_t convert_from_int;
  dnnPrimitive_t convert_prv2prv;
  shared_ptr<MKLMemoryDescriptorBase<Dtype> > descr_prv2prv_conversion;

  std::string name;  // for debugging purposes
  void allocate() {
    if (internal_ptr == NULL) {

#ifdef USE_MLSL
      if (mn::is_multinode()) {
        internal_ptr = (Dtype*)mn::alloc(prv_size(), 64);
        if (internal_ptr == NULL)
          LOG(FATAL) << "internal_ptr is NULL after MLSL::Alloc";
      } else {
#endif /* !USE_MLSL */
        int status = dnnAllocateBuffer<Dtype>(
          reinterpret_cast<void **>(&internal_ptr), layout_int);
        CHECK_EQ(status, E_SUCCESS)
          << "Failed internal_ptr memory allocation with status "
          << status << "\n";
#ifdef USE_MLSL
      }
#endif /* USE_MLSL */

      caffe_set(prv_count(), Dtype(0), internal_ptr);
    }
  }
  virtual void* prv_ptr() {
    if (internal_ptr == NULL)
        allocate();
    return internal_ptr;
  }
  inline bool conversion_needed() { return (convert_to_int != NULL);}
  void create_conversions();
  void create_internal_layout(const dnnPrimitive_t primitive,
                              dnnResourceType_t type);
  void create_user_layout(size_t dimension, const size_t size[],
                          const size_t strides[],
                          bool create_conversion_if_possible = true);
  void create_layouts(
    const dnnPrimitive_t primitive, dnnResourceType_t type,
    size_t dimension, const size_t size[], const size_t strides[]);

  void remove_internal_layout();
  void remove_user_layout();

  virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKL2017;}
  virtual size_t prv_size() {
      return dnnLayoutGetMemorySize<Dtype>(layout_int);
  }
  virtual size_t prv_count() {
      return dnnLayoutGetMemorySize<Dtype>(layout_int) / sizeof(Dtype);
  }
  virtual void convert_from_prv(void* cpu_ptr);
  virtual void convert_to_prv(void* cpu_ptr);
#ifdef CO_SIM
    virtual void create_reorder_from_prv_cosim(void* cpu_ptr);
    virtual void convert_from_prv_cosim(void* cpu_ptr);
#endif
  virtual bool layout_compare(shared_ptr<PrvMemDescr> other);
  virtual void convert_from_other(shared_ptr<PrvMemDescr> other);
 protected:
  void remove_conversions();
 protected:
  Dtype* internal_ptr;
};

template <typename Dtype, bool is_diff>
struct MKLMemoryDescriptor : MKLMemoryDescriptorBase<Dtype> {
  // The last get_converted_prv() argument is a hack for reusing
  // in backward a conversion done already in the forward direction.
  Dtype* get_converted_prv(Blob<Dtype> * blob, bool set_prv_ptr,
          MKLMemoryDescriptor<Dtype, is_diff>* converted_in_fwd = NULL);
};

template <typename Dtype>
struct MKLData : MKLMemoryDescriptor<Dtype, false>
{};

template <typename Dtype>
struct MKLDiff : MKLMemoryDescriptor<Dtype, true>
{};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKL_MEMORY_HPP_
