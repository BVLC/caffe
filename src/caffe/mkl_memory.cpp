#ifdef MKL2017_SUPPORTED
#include "caffe/mkl_memory.hpp"

// Uncomment to see where the layout conversions are done
// #undef DLOG
#ifndef DLOG
#define DLOG LOG
#endif

namespace caffe {

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_conversions() {
  int status;
  if (this->convert_from_int) {
    DLOG(INFO) << "convert_from_int layout already created, recreating for"
           << this->name;
    status = dnnDelete<Dtype>(this->convert_from_int);
    CHECK_EQ(status, E_SUCCESS);
  }
  if (this->convert_to_int) {
    DLOG(INFO) << "convert_to_int layout already created, recreating for"
           << this->name;
    status = dnnDelete<Dtype>(this->convert_to_int);
    CHECK_EQ(status, E_SUCCESS);
  }
  if (layout_int
      && !dnnLayoutCompare<Dtype>(layout_usr, layout_int)) {
    CHECK(layout_usr);
    status = dnnConversionCreate<Dtype>(&convert_to_int, layout_usr,
            layout_int);
    CHECK_EQ(status, E_SUCCESS)
            << "Failed creation convert_to_int with status "
            << status << " for buffer: " << this->name << "\n";
    status = dnnConversionCreate<Dtype>(&convert_from_int, layout_int,
            layout_usr);
    CHECK_EQ(status, E_SUCCESS)
            << "Failed creation convert_from_int with status "
            << status << " for buffer: " << this->name << "\n";
  }
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_internal_layout(
    const dnnPrimitive_t primitive, dnnResourceType_t type) {
  int status;
  if (this->layout_int) {
    DLOG(INFO) << "Internal layout already created, recreating for"
           << this->name;
    status = dnnLayoutDelete<Dtype>(this->layout_int);
    CHECK_EQ(status, E_SUCCESS);
  }
  status = dnnLayoutCreateFromPrimitive<Dtype>(
      &this->layout_int, primitive, type);
  CHECK_EQ(status, E_SUCCESS)
      << "Failed dnnLayoutCreateFromPrimitive with status "
      << status << " for buffer: " << this->name << "\n";

  if (this->layout_usr)
    this->create_conversions();
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_user_layout(
    size_t dimension, const size_t size[], const size_t strides[]) {
  int status;
  if (this->layout_usr) {
    DLOG(INFO) << "User layout already created, recreating for"
               << this->name;
    status = dnnLayoutDelete<Dtype>(this->layout_usr);
    CHECK_EQ(status, E_SUCCESS);
  }

  status = dnnLayoutCreate<Dtype>(
      &this->layout_usr, dimension, size, strides);
  CHECK_EQ(status, E_SUCCESS) << "Failed dnnLayoutCreate with status "
      << status << " for buffer: " << this->name << "\n";

  if (this->layout_int)
    this->create_conversions();
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::create_layouts(
    const dnnPrimitive_t primitive, dnnResourceType_t type,
    size_t dimension, const size_t size[], const size_t strides[]) {
  // To avoid creating conversion among potentialiy diffrent 
  // (in terms of size) layouts we need to destroy existing layouts here

  if (this->layout_usr) {
    DLOG(INFO) << "User layout already created, recreating for"
               << this->name;
    int status = dnnLayoutDelete<Dtype>(this->layout_usr);
    CHECK_EQ(status, E_SUCCESS);
  }
  this->create_internal_layout(primitive, type);
  this->create_user_layout(dimension, size, strides);
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_from_prv(void* cpu_ptr) {
  CHECK(cpu_ptr);
  CHECK(this->convert_from_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  DLOG(INFO) << "convert priv =>           "  << this->name << " =>";

  convert_resources[dnnResourceFrom] = this->prv_ptr();
  convert_resources[dnnResourceTo]   = cpu_ptr;
  status = dnnExecute<Dtype>(this->convert_from_int, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv failed with status " << status;
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_to_prv(void* cpu_ptr) {
  CHECK(cpu_ptr);
  CHECK(this->convert_to_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  DLOG(INFO) << "convert      => priv                                => "
             << this->name;

  convert_resources[dnnResourceFrom] = cpu_ptr;
  convert_resources[dnnResourceTo]   = this->prv_ptr();
  status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv failed with status " << status;
}


template <typename Dtype>
bool MKLMemoryDescriptorBase<Dtype>::layout_compare(
  shared_ptr<PrvMemDescr> other) {
  CHECK_EQ(other->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL2017);

  shared_ptr<MKLMemoryDescriptorBase<Dtype> > other_descr =
      boost::static_pointer_cast<MKLMemoryDescriptorBase<Dtype> >
            (other);

  if (dnnLayoutCompare<Dtype>(other_descr->layout_int,
      this->layout_int))
    return true;
  else
    return false;
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_from_other(
  shared_ptr<PrvMemDescr> other) {
  shared_ptr<MKLMemoryDescriptorBase<Dtype> > other_descr =
      boost::static_pointer_cast<MKLMemoryDescriptorBase<Dtype> >
            (other);

  DLOG(INFO) << "convert other => priv     "  << other_descr->name
             << " => " << this->name;

  int status;
  dnnPrimitive_t convert;
  // TODO: cache this primitive?
  status = dnnConversionCreate<Dtype>(&convert,
    other_descr->layout_int, this->layout_int);

  void *convert_resources[dnnResourceNumber];
  convert_resources[dnnResourceFrom] = other_descr->prv_ptr();
  convert_resources[dnnResourceTo]   = this->prv_ptr();
  status = dnnExecute<Dtype>(convert, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from other failed with status "
                      << status;

  dnnDelete<Dtype>(convert);
}

template <typename Dtype, bool is_diff>
Dtype* MKLMemoryDescriptor<Dtype, is_diff>::get_converted_prv(
  Blob<Dtype>* blob, bool set_prv_ptr,
  MKLMemoryDescriptor<Dtype, is_diff>* converted_in_fwd) {
  if (this->convert_to_int) {
    int status;
    void *convert_resources[dnnResourceNumber];
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if (prv_ptr == NULL) {
      if (converted_in_fwd) {
        // hack for reusing previously done conversion
        // if(dnnLayoutCompare(converted_in_fwd->layout_int , this->layout_int))
        if (1) {
          DLOG(INFO) << "reusing fwd               "
                  << converted_in_fwd->name << " == " << this->name;
          return converted_in_fwd->internal_ptr;
        } else {
          DLOG(INFO) << "layout doesn't match      "
                  << converted_in_fwd->name << " != " << this->name;
        }
      }

      DLOG(INFO) << "convert      => priv                                => "
                 << this->name;

      this->allocate();
      convert_resources[dnnResourceFrom] =
              is_diff ?
                reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_diff()))
              : reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_data()));
      convert_resources[dnnResourceTo] =
              reinterpret_cast<void *>(this->internal_ptr);

      status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
      CHECK_EQ(status, 0) << "Conversion failed with status " << status;

      if (set_prv_ptr) {
        if (is_diff)
          blob->set_prv_diff_descriptor(this->get_shared_ptr(), true);
        else
          blob->set_prv_data_descriptor(this->get_shared_ptr(), true);
      }
      return this->internal_ptr;
    } else {
      // This section helps if padding needs to be added (or removed...)
      // TODO: consider removing when no longer needed.
      shared_ptr<PrvMemDescr> prv_mem_descriptor =
          is_diff ? (blob->get_prv_diff_descriptor()) :
            (blob->get_prv_data_descriptor());

      CHECK_EQ(prv_mem_descriptor->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKL2017);

      shared_ptr<MKLMemoryDescriptor<Dtype, is_diff> > current_descr =
        boost::static_pointer_cast<MKLMemoryDescriptor<Dtype, is_diff> >
              (prv_mem_descriptor);

      if (!dnnLayoutCompare<Dtype>(current_descr->layout_int,
              this->layout_int)) {
        if (converted_in_fwd) {
          // hack for reusing previously done conversion
          // if(dnnLayoutCompare(converted_in_fwd->layout_int,this->layout_int))
          if (1) {
            DLOG(INFO) << "reusing fwd               "
                    << converted_in_fwd->name << " == " << this->name;
            return converted_in_fwd->internal_ptr;
          } else {
            DLOG(INFO) << "layout doesn't match      "
                    << converted_in_fwd->name << " != " << this->name;
          }
        }
        DLOG(INFO) << "convert priv => priv      "
                << current_descr->name << " => " << this->name;

        if (this->convert_prv2prv) {
          CHECK_EQ(dnnLayoutCompare<Dtype>(
              this->descr_prv2prv_conversion->layout_int,
              this->layout_int), 0);
          status = 0;
        } else {
          status = dnnConversionCreate<Dtype>(&this->convert_prv2prv,
                  current_descr->layout_int , this->layout_int);
          if (status == 0)
            this->descr_prv2prv_conversion = current_descr;
        }

        if (status != 0) {
          // TODO: Very weird that we end up here for conv1. No idea why....
          DLOG(INFO) << "!!!! Failed creation convert_prv2prv with status "
                  << status << "\n";

          this->allocate();
          convert_resources[dnnResourceFrom] = is_diff ?
            reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_diff())) :
            reinterpret_cast<void *>(const_cast<Dtype*>(blob->cpu_data()));
          convert_resources[dnnResourceTo] =
            reinterpret_cast<void*>(this->internal_ptr);

          status = dnnExecute<Dtype>(this->convert_to_int, convert_resources);
          CHECK_EQ(status, 0) << "Conversion failed with status " << status;

        } else {
          this->allocate();

          convert_resources[dnnResourceFrom] = is_diff ?
            reinterpret_cast<void *>(const_cast<Dtype *>(blob->prv_diff())) :
            reinterpret_cast<void *>(const_cast<Dtype *>(blob->prv_data()));
          convert_resources[dnnResourceTo] =
                  reinterpret_cast<void *>(this->internal_ptr);
          status = dnnExecute<Dtype>(this->convert_prv2prv, convert_resources);
          CHECK_EQ(status, 0) << "Conversion failed with status " << status;
        }

        if (set_prv_ptr) {
          if (is_diff)
            blob->set_prv_diff_descriptor(this->get_shared_ptr(), true);
          else
            blob->set_prv_data_descriptor(this->get_shared_ptr(), true);
        }
        return this->internal_ptr;
      } else if (current_descr.get() != this) {
        DLOG(INFO) << "layout OK                 "
                << current_descr->name << " == " << this->name;
      }
    }

    return const_cast<Dtype *>(prv_ptr);
  }

  return (is_diff ? const_cast<Dtype *>(blob->cpu_diff()) :
                    const_cast<Dtype *>(blob->cpu_data()));
}

template class MKLMemoryDescriptor<double, true>;
template class MKLMemoryDescriptor<float, true>;
template class MKLMemoryDescriptor<float, false>;
template class MKLMemoryDescriptor<double, false>;
template class MKLMemoryDescriptorBase<float>;
template class MKLMemoryDescriptorBase<double>;
}  // namespace caffe
#endif  // #ifdef MKL2017_SUPPORTED
