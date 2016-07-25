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
    dnnReleaseBuffer<Dtype>(internal_ptr);
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
      int status = dnnAllocateBuffer<Dtype>(
        reinterpret_cast<void **>(&internal_ptr), layout_int);
      CHECK_EQ(status, 0)
        << "Failed internal_ptr memory allocation with status "
        << status << "\n";

      caffe_set(prv_count(), Dtype(0), internal_ptr);
    }
  }
  Dtype* prv_ptr() {
    if (internal_ptr == NULL)
        allocate();
    return internal_ptr;
  }
  void create_conversions() {
    if (layout_int
        && !dnnLayoutCompare<Dtype>(layout_usr, layout_int)) {
      CHECK(layout_usr);
      int status = dnnConversionCreate<Dtype>(&convert_to_int, layout_usr,
              layout_int);
      CHECK_EQ(status, 0) << "Failed creation convert_to_int with status "
              << status << " for buffer: " << this->name << "\n";
      status = dnnConversionCreate<Dtype>(&convert_from_int, layout_int,
              layout_usr);
      CHECK_EQ(status, 0) << "Failed creation convert_from_int with status "
              << status << " for buffer: " << this->name << "\n";
    }
  }
  
  void create_internal_layout(const dnnPrimitive_t primitive, dnnResourceType_t type) {
      int status;
      CHECK(this->layout_int == NULL) << "Internal layout already created for "
                                      << this->name;
      status = dnnLayoutCreateFromPrimitive<Dtype>(
          &this->layout_int, primitive, type);
      CHECK_EQ(status, 0) << "Failed dnnLayoutCreateFromPrimitive with status "
          << status << " for buffer: " << this->name << "\n";

      if(this->layout_usr && !this->convert_from_int && !this->convert_to_int)
          create_conversions();
  }
  
  void create_user_layout(size_t dimension, const size_t size[], const size_t strides[]) {
      int status;
      CHECK(this->layout_usr == NULL) << "User layout already created for"
                                      << this->name;
      status = dnnLayoutCreate<Dtype>(
          &this->layout_usr, dimension, size, strides);
      CHECK_EQ(status, 0) << "Failed dnnLayoutCreate with status "
          << status << " for buffer: " << this->name << "\n";

      if(this->layout_int && !this->convert_from_int && !this->convert_to_int)
          create_conversions();
  }
  void create_layouts(const dnnPrimitive_t primitive, dnnResourceType_t type, 
                 size_t dimension, const size_t size[], const size_t strides[]) {
      this->create_internal_layout(primitive, type);
      this->create_user_layout(dimension, size, strides);
      this->create_conversions();
  }
  
  virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKL2017;}
  virtual size_t prv_size() {
      return dnnLayoutGetMemorySize<Dtype>(layout_int);
  }
  virtual size_t prv_count() {
      return dnnLayoutGetMemorySize<Dtype>(layout_int) / sizeof(Dtype);
  }
  virtual void convert_from_prv(void* prv_ptr, void* cpu_ptr);
  virtual void convert_to_prv(void* cpu_ptr, void* prv_ptr);
  virtual bool layout_compare(shared_ptr<PrvMemDescr> other);
  virtual void convert_from_other(shared_ptr<PrvMemDescr> other,
                                  void* from, void* to);
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
