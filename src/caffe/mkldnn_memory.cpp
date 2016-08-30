#ifdef MKLDNN_SUPPORTED
#include "caffe/mkldnn_memory.hpp"

namespace caffe {

template <typename Dtype>
MKLDNNMemoryDescriptorBase<Dtype>::MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                                            , shared_ptr<memory::primitive_desc> prv_memory_pd )
                                    : _usr_memory_pd(NULL), _prv_memory_pd(NULL)
                                    ,_reorder_usr2prv_pd(NULL), _reorder_prv2usr_pd(NULL)
                                    ,_prv_memory(NULL), _internal_ptr(NULL)
                                    , name("MKLDNNMemoryDescriptorBase")
{
    set_usr_memory_pd(usr_memory_pd);
    set_prv_memory_pd(prv_memory_pd);
}

template <typename Dtype>
size_t MKLDNNMemoryDescriptorBase<Dtype>::prv_count()
{
    mkldnn::c_api::mkldnn_dims_t* pdims = &(_usr_memory_pd->data.memory_desc.tensor_desc.dims);
    uint32_t ndims = _usr_memory_pd->data.memory_desc.tensor_desc.ndims;
    uint32_t count = 1;
    for (uint32_t i = 0; i < ndims; ++i) count *= (*pdims)[i];
    return count;
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::check_usr_with_prv_descriptors()
{
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    uint32_t ndims = _usr_memory_pd->data.memory_desc.tensor_desc.ndims;
    CHECK_EQ(ndims, _prv_memory_pd->data.memory_desc.tensor_desc.ndims)
            << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions number";
    for (uint32_t dim = 0; dim < ndims; ++dim) {
        CHECK_EQ(_usr_memory_pd->data.memory_desc.tensor_desc.dims[dim]
                , _prv_memory_pd->data.memory_desc.tensor_desc.dims[dim])
                << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions";
    }
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::create_reorders()
{
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    if ( *_usr_memory_pd != *_prv_memory_pd) {
        _reorder_usr2prv_pd = shared_ptr<reorder::primitive_desc>(
                new reorder::primitive_desc(*_usr_memory_pd, *_prv_memory_pd));
        _reorder_prv2usr_pd = shared_ptr<reorder::primitive_desc>(
                new reorder::primitive_desc(*_prv_memory_pd, *_usr_memory_pd));
    }
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::convert_from_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    CHECK(_reorder_prv2usr_pd);
    memory usr_memory(*_usr_memory_pd, cpu_ptr);
    reorder reorder_prv2usr(*_reorder_prv2usr_pd, *this->get_prv_memory(), usr_memory );
    VLOG(1) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_from_prv --- " << this->name;
    stream().submit({reorder_prv2usr}).wait();
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::convert_to_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    CHECK(_reorder_usr2prv_pd);
    memory usr_memory(*_usr_memory_pd, cpu_ptr);
    reorder reorder_usr2prv(*_reorder_usr2prv_pd, usr_memory, *this->get_prv_memory() );
    VLOG(1) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_to_prv --- " << this->name;
    stream().submit({reorder_usr2prv}).wait();
}


template <typename Dtype>
bool MKLDNNMemoryDescriptorBase<Dtype>::layout_compare(shared_ptr<PrvMemDescr> other)
{
    CHECK_EQ(other->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKLDNN);

    shared_ptr<MKLDNNMemoryDescriptorBase<Dtype> > other_descr =
        boost::static_pointer_cast<MKLDNNMemoryDescriptorBase<Dtype> >(other);

    return (*other_descr->prv_memory_pd() == *this->prv_memory_pd());
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::convert_from_other(shared_ptr<PrvMemDescr> other)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype, bool is_diff>
Dtype* MKLDNNMemoryDescriptor<Dtype, is_diff>::get_blob_data_ptr(Blob<Dtype>* blob
                                        ,bool set_prv_ptr
                                        ,MKLDNNMemoryDescriptor<Dtype,is_diff>* converted_in_fwd)
{
    if (!this->conversion_needed()) {
        return (is_diff ? const_cast<Dtype *>(blob->cpu_diff()) :
                    const_cast<Dtype *>(blob->cpu_data()));
    }

    // Conversion is needed
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if (prv_ptr == NULL) {
        if (converted_in_fwd) {
            // TODO: use previously done conversion on forward - needed for training
            NOT_IMPLEMENTED;
        }

        this->convert_to_prv(const_cast<Dtype*>(is_diff ? blob->cpu_diff() : blob->cpu_data()));

        if (set_prv_ptr) {
            if (is_diff)
                blob->set_prv_diff_descriptor(this->get_shared_ptr(), true);
            else
                blob->set_prv_data_descriptor(this->get_shared_ptr(), true);
        }
        return static_cast<Dtype *>(this->prv_ptr());
    } else {
        shared_ptr<PrvMemDescr> prv_mem_descriptor = is_diff ?
                (blob->get_prv_diff_descriptor()) : (blob->get_prv_data_descriptor());

        CHECK_EQ(prv_mem_descriptor->get_descr_type(), PrvMemDescr::PRV_DESCR_MKLDNN);

        shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > current_descr =
                boost::static_pointer_cast<MKLDNNMemoryDescriptor<Dtype, is_diff> >(prv_mem_descriptor);

        if (*current_descr->prv_memory_pd() !=  *this->prv_memory_pd()) {
            // TODO: prv in blob and in this descrptor may have different layouts
            NOT_IMPLEMENTED;
        } else if (current_descr.get() != this) {
            VLOG(1) << "layout OK " << current_descr->name << " == " << this->name;
        }
        return const_cast<Dtype *>(prv_ptr);
    }
    NOT_IMPLEMENTED;
    return NULL;
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::sync_blob_prv_data(Blob<Dtype>* blob)
{
    get_blob_data_ptr(blob, false);
}

template <typename Dtype, bool is_diff>
shared_ptr<memory> MKLDNNMemoryDescriptor<Dtype, is_diff>::create_input_memory(Blob<Dtype> * blob)
{
    shared_ptr<memory> imem;
    if (this->conversion_needed())
        imem.reset(new memory(*this->prv_memory_pd(), this->get_blob_data_ptr(blob, false)));
    else
        imem.reset(new memory(*this->usr_memory_pd(), const_cast<Dtype*>(is_diff ?  blob->cpu_diff() : blob->cpu_data())));
    return imem;
}

template <typename Dtype, bool is_diff>
shared_ptr<memory> MKLDNNMemoryDescriptor<Dtype, is_diff>::create_output_memory(Blob<Dtype> * blob)
{
    shared_ptr<memory> omem;
    if (this->conversion_needed()) {
        if(blob->get_prv_data_descriptor() != NULL) {
            shared_ptr<PrvMemDescr> prv_mem_descriptor = is_diff ?
                (blob->get_prv_diff_descriptor()) : (blob->get_prv_data_descriptor());
            CHECK_EQ(prv_mem_descriptor->get_descr_type(), PrvMemDescr::PRV_DESCR_MKLDNN);
            shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > current_descr =
                boost::static_pointer_cast<MKLDNNMemoryDescriptor<Dtype, is_diff> >(prv_mem_descriptor);
            omem = current_descr->get_prv_memory();
            this->set_prv_memory(omem);
        } else {
            omem = this->get_prv_memory();
        }
    } else {
        omem.reset(new memory(*this->usr_memory_pd(), is_diff ? blob->mutable_cpu_diff() : blob->mutable_cpu_data()));
    }
    return omem;
}

template class MKLDNNMemoryDescriptor<double, true>;
template class MKLDNNMemoryDescriptor<float, true>;
template class MKLDNNMemoryDescriptor<float, false>;
template class MKLDNNMemoryDescriptor<double, false>;
template class MKLDNNMemoryDescriptorBase<float>;
template class MKLDNNMemoryDescriptorBase<double>;
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
