#ifdef MKL2017_SUPPORTED
#include "caffe/mkl_memory.hpp"

// Uncomment to see where the layout conversions are done
// #undef DLOG
#ifndef DLOG
#define DLOG LOG
#endif

static int getMKLBuildDate() {
  static int build = 0;
  if (build == 0) {
    MKLVersion v;
    mkl_get_version(&v);
    build = atoi(v.Build);
  }
  return build;
}

namespace caffe {

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_from_prv(void* prv_ptr,
        void* cpu_ptr) {
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(this->convert_from_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  DLOG(INFO) << "convert priv =>           "  << this->name << " =>";

  convert_resources[dnnResourceFrom] = prv_ptr;
  convert_resources[dnnResourceTo]   = cpu_ptr;
  status = dnnExecute<Dtype>(this->convert_from_int, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv failed with status " << status;
}

template <typename Dtype>
void MKLMemoryDescriptorBase<Dtype>::convert_to_prv(void* cpu_ptr,
        void* prv_ptr) {
  CHECK(prv_ptr);
  CHECK(cpu_ptr);
  CHECK(this->convert_to_int);
  int status;
  void *convert_resources[dnnResourceNumber];

  DLOG(INFO) << "convert      => priv                                => "
             << this->name << " =>";

  convert_resources[dnnResourceFrom] = cpu_ptr;
  convert_resources[dnnResourceTo]   = prv_ptr;
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
  shared_ptr<PrvMemDescr> other, void* from, void* to) {
  // TODO: cache this primitive

  shared_ptr<MKLMemoryDescriptorBase<Dtype> > other_descr =
      boost::static_pointer_cast<MKLMemoryDescriptorBase<Dtype> >
            (other);

  DLOG(INFO) << "convert other => priv     "  << other_descr->name
             << " => " << this->name;

  int status;
  dnnPrimitive_t convert;
  status = dnnConversionCreate<Dtype>(&convert,
    other_descr->layout_int, this->layout_int);

  void *convert_resources[dnnResourceNumber];
  convert_resources[dnnResourceFrom] = from;
  convert_resources[dnnResourceTo]   = to;
  status = dnnExecute<Dtype>(convert, convert_resources);
  CHECK_EQ(status, 0) << "Conversion from prv to other failed with status "
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
          blob->set_prv_diff(this->internal_ptr, this->get_shared_ptr(), true);
        else
          blob->set_prv_data(this->internal_ptr, this->get_shared_ptr(), true);
      }
      return this->internal_ptr;
    } else {
      // This section helps if padding needs to be added (or removed...)
      // TODO: consider removing when no longer needed.
      shared_ptr<PrvMemDescr> prv_mem_descriptor =
          is_diff ? (blob->get_prv_descriptor_diff()) :
            (blob->get_prv_descriptor_data());

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
            blob->set_prv_diff(this->internal_ptr, this->get_shared_ptr(), true);
          else
            blob->set_prv_data(this->internal_ptr, this->get_shared_ptr(), true);
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

}  // namespace caffe
#endif  // #ifdef MKL2017_SUPPORTED
