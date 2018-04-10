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

#ifdef MKLDNN_SUPPORTED
#include "caffe/mkldnn_memory.hpp"
#include "caffe/util/performance.hpp"

namespace caffe {


template <typename Dtype>
MKLDNNMemoryDescriptorBase<Dtype>::MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                                            , shared_ptr<memory::primitive_desc> prv_memory_pd
                                                            , Blob<Dtype>* blob
                                                            , MKLDNNLayer<Dtype>* mkldnn_layer
                                                            , std::vector<float> scale
                                                            , int mask
                                                            , bool is_sum)
                                    : name("MKLDNNMemoryDescriptorBase")
                                    , _reorder_usr2prv_pd(), _reorder_prv2usr_pd(), _reorder_extprv2prv_pd()
                                    ,_prv_memory(), _internal_ptr(NULL), _usr_memory(), _cpu_ptr(NULL)
                                    , _mkldnn_layer(NULL)
{
    set_usr_memory_pd(usr_memory_pd, scale);
    set_prv_memory_pd(prv_memory_pd, scale, mask);
    set_mkldnn_layer(mkldnn_layer);
    this->set_scale(scale);
    this->set_sum(is_sum);
    this->_blob = blob;
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::check_usr_with_prv_descriptors()
{
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    int32_t ndims = _usr_memory_pd->desc().data.ndims;
    CHECK_EQ(ndims, _prv_memory_pd->desc().data.ndims)
            << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions number";
    for (int32_t dim = 0; dim < ndims; ++dim) {
        CHECK_EQ(_usr_memory_pd->desc().data.dims[dim]
                , _prv_memory_pd->desc().data.dims[dim])
                << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions";
    }
}

template <typename Dtype>
void MKLDNNMemoryDescriptorBase<Dtype>::create_reorder_descriptors(std::vector<float> scale, int mask, std::vector<float> scale_ext, bool is_sum)
{
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);

    primitive_attr attri;
    int count = scale.size();
    if ( *_usr_memory_pd != *_prv_memory_pd) {
        std::vector<float> scales_u2p(count);
        #pragma omp parallel for if (count > 1)
        for(int i=0; i < count; i++){
            scales_u2p[i] = scale[i];
        }
        attri.set_output_scales(mask, scales_u2p);
        attri.set_int_output_round_mode(round_nearest);
        _reorder_usr2prv_pd = shared_ptr<reorder::primitive_desc>(
                new reorder::primitive_desc(*_usr_memory_pd, *_prv_memory_pd, attri));

        std::vector<float> scales_p2u(count);
        #pragma omp parallel for if (count > 1)
        for(int i=0; i < count; i++){
            scales_p2u[i] = (1. / scale[i]);
        }
        attri.set_output_scales(mask, scales_p2u); 
        attri.set_int_output_round_mode(round_nearest);
        _reorder_prv2usr_pd = shared_ptr<reorder::primitive_desc>(
                new reorder::primitive_desc(*_prv_memory_pd, *_usr_memory_pd, attri));
    }
    if ( _extprv_memory_pd && (*_prv_memory_pd != *_extprv_memory_pd || scale != scale_ext)) {
        if(is_sum == true && scale == scale_ext && _extprv_memory_pd->desc().data.data_type == memory::data_type::s8 && _prv_memory_pd->desc().data.data_type == memory::data_type::u8){
#ifdef DEBUG
            LOG(INFO) << "skip s8 to u8 reorder....";
#endif
            _reorder_extprv2prv_pd = NULL;
        }else{
            std::vector<float> scales_e2p(count);
            float shift_scale;
            #pragma omp parallel for if (count > 1)
            for(int i=0; i < count; i++){
                shift_scale = scale[i] / scale_ext[i]; //fp32->int8 blob_prv_mkldnn_mem_descr->get_scale() will always be 0 ?
                scales_e2p[i] = shift_scale;
            }
            attri.set_output_scales(mask, scales_e2p);
            attri.set_int_output_round_mode(round_nearest);
            _reorder_extprv2prv_pd = shared_ptr<reorder::primitive_desc>(new reorder::primitive_desc(*_extprv_memory_pd, *_prv_memory_pd, attri));
            
        }
    }
}

template <typename Dtype, bool is_diff>
 MKLDNNMemoryDescriptor<Dtype, is_diff>::MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                        , std::vector<float> scale
                        , int mask
                        , bool is_sum)
        : MKLDNNMemoryDescriptorBase<Dtype>(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer, scale, mask, is_sum)
{
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();

    if (prv_ptr != NULL) {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor<Dtype, is_diff>(blob);
#ifdef DEBUG        
        LOG(INFO) << "Format of blob-prv-memory-pd: " << blob_prv_mkldnn_mem_descr->prv_memory_pd()->desc().data.format;
        LOG(INFO) << "Format of this-prv-memory-pd: " << this->prv_memory_pd()->desc().data.format;
#endif
        if (*blob_prv_mkldnn_mem_descr->prv_memory_pd() !=  *this->prv_memory_pd() || blob_prv_mkldnn_mem_descr->get_scale() != this->get_scale()) {
#ifdef DEBUG
            LOG(INFO) << "Formats of blob-prv-memory-pd and this-prv-memory-pd are not equal !";
#endif
            this->set_extprv_memory_pd(blob_prv_mkldnn_mem_descr->prv_memory_pd(), scale, blob_prv_mkldnn_mem_descr->get_scale(), blob_prv_mkldnn_mem_descr->get_sum());
        }
    }
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::create_reorder_to_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_usr2prv_pd);

    if(this->_usr_memory == NULL || this->_cpu_ptr != cpu_ptr)
        this->_usr_memory.reset(new memory(*this->_usr_memory_pd, cpu_ptr));
    if(this->_reorder_usr2prv.aprimitive == NULL || this->_cpu_ptr != cpu_ptr)
        this->_reorder_usr2prv.reset(new reorder(*this->_reorder_usr2prv_pd, *this->_usr_memory, *this->get_prv_memory()));

    this->_cpu_ptr = cpu_ptr;
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::convert_to_prv(void* cpu_ptr)
{
#ifdef DEBUG
    LOG(INFO) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_to_prv --- " << this->name;
#endif
    create_reorder_to_prv(cpu_ptr);
    VLOG(1) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_to_prv --- " << this->name;
#ifdef DEBUG
    LOG(INFO) << "Reorder: from usr to prv.";
    LOG(INFO) << "Format of _usr_memory_pd: " << this->_usr_memory_pd->desc().data.format << "   Data_type of _usr_memory_pd: " << this->_usr_memory_pd->desc().data.data_type;
    LOG(INFO) << "Format of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.format << "   Data_type of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.data_type;
#endif
    PERFORMANCE_MEASUREMENT_BEGIN();
    this->_reorder_usr2prv.submit();
    PERFORMANCE_MEASUREMENT_END_STATIC("mkldnn_conversion");
}
#ifdef CO_SIM
template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::convert_from_prv_cosim(void* cpu_ptr_cosim)
{
    if(this->_reorder_prv2usr_pd == NULL)
        return;
    create_reorder_from_prv_cosim(cpu_ptr_cosim);
    this->_reorder_prv2usr_cosim.submit();

}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::create_reorder_from_prv_cosim(void* cpu_ptr_cosim)
{
    CHECK(cpu_ptr_cosim);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_prv2usr_pd);

    this->_usr_memory_cosim = this->_usr_memory;
    this->_reorder_prv2usr_cosim.aprimitive = this->_reorder_prv2usr.aprimitive;

    // Used to save data copy from prv.
    this->_prv_memory_cosim.reset(new memory(*this->_prv_memory_pd));

    // Copy prv data to _prv_memory_cosim.
    memcpy(this->_prv_memory_cosim->get_data_handle(), this->get_prv_ptr(), this->prv_size());

    // Wrap _prv_memory_cosim.
    this->at_prv_cosim.reset(new primitive::at(*this->_prv_memory_cosim));

    if(this->_usr_memory == NULL || this->_cpu_ptr != cpu_ptr_cosim)
        this->_usr_memory_cosim.reset(new memory(*this->_usr_memory_pd, cpu_ptr_cosim));
    if(this->_reorder_prv2usr.aprimitive == NULL || this->_cpu_ptr != cpu_ptr_cosim){

        // Create primitive for reorder.
        this->_reorder_prv2usr_cosim.aprimitive.reset(new reorder(*this->_reorder_prv2usr_pd, *this->at_prv_cosim, *this->_usr_memory_cosim));
    }
}
#endif
template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::create_reorder_from_prv(void* cpu_ptr)
{
    CHECK(cpu_ptr);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_prv2usr_pd);
    if(this->_usr_memory == NULL || this->_cpu_ptr != cpu_ptr)
        this->_usr_memory.reset(new memory(*this->_usr_memory_pd, cpu_ptr));
    if(this->_reorder_prv2usr.aprimitive == NULL || this->_cpu_ptr != cpu_ptr) {
        CHECK(this->aprimitive());
        this->_reorder_prv2usr.aprimitive.reset(new reorder(*this->_reorder_prv2usr_pd, *this->aprimitive(), *this->_usr_memory));
    }
    this->_cpu_ptr = cpu_ptr;
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::convert_from_prv(void* cpu_ptr)
{
#ifdef DEBUG
    LOG(INFO) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_from_prv --- " << this->name;
#endif
    CHECK(cpu_ptr);
    if(this->_reorder_prv2usr_pd == NULL)
        return;
    create_reorder_from_prv(cpu_ptr);
    VLOG(1) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_from_prv --- " << this->name;
#ifdef DEBUG
    LOG(INFO) << "Reorder: from prv to usr.";
    LOG(INFO) << "Format of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.format;
    LOG(INFO) << "Format of _usr_memory_pd: " << this->_usr_memory_pd->desc().data.format;
    LOG(INFO) << "Format of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.format << "   Data_type of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.data_type;
    LOG(INFO) << "Format of _usr_memory_pd: " << this->_usr_memory_pd->desc().data.format << "   Data_type of _usr_memory_pd: " << this->_usr_memory_pd->desc().data.data_type;
#endif
    PERFORMANCE_MEASUREMENT_BEGIN();
    this->_reorder_prv2usr.submit();
    PERFORMANCE_MEASUREMENT_END_STATIC("mkldnn_conversion");
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::create_reorder_from_extprv(shared_ptr<primitive> aprimitive)
{
    CHECK(aprimitive);
    CHECK(this->_extprv_memory_pd);
    CHECK(this->_prv_memory_pd);
    CHECK(this->_reorder_extprv2prv_pd);
    if(this->_reorder_extprv2prv.aprimitive == NULL)
        this->_reorder_extprv2prv.reset(new reorder(*this->_reorder_extprv2prv_pd, *aprimitive, *this->get_prv_memory()));
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::convert_from_extprv(shared_ptr<primitive> aprimitive)
{
#ifdef DEBUG
    LOG(INFO) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_from_extprv --- " << this->name;
#endif
    CHECK(aprimitive);
    if(this->_reorder_extprv2prv_pd == NULL)
        return;
//    if (*this->_extprv_memory_pd == *this->_prv_memory_pd)
//    {
//#ifdef DEBUG
//        LOG(INFO) << "The format and data_type of _extprv_memory_pd and _prv_memory_pd is same, no need do conversion.";
//#endif
//        return;
//    }
    create_reorder_from_extprv(aprimitive);
    VLOG(1) << "--- MKLDNNMemoryDescriptorBase<Dtype>::convert_from_extprv --- " << this->name;
#ifdef DEBUG
    LOG(INFO) << "Reorder: from extprv to prv.";
    LOG(INFO) << "Format of _extprv_memory_pd: " << this->_extprv_memory_pd->desc().data.format << "   Data_type of _extprv_memory_pd: " << this->_extprv_memory_pd->desc().data.data_type;
    LOG(INFO) << "Format of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.format<< "   Data_type of _prv_memory_pd: " << this->_prv_memory_pd->desc().data.data_type;
#endif
    PERFORMANCE_MEASUREMENT_BEGIN();
    this->_reorder_extprv2prv.submit();
    PERFORMANCE_MEASUREMENT_END_STATIC("mkldnn_conversion");
}


template <typename Dtype, bool is_diff>
bool MKLDNNMemoryDescriptor<Dtype, is_diff>::on_to_cpu()
{
    CHECK(this->mkldnn_layer());
    if (StreamHolder::Instance().current_stream() != NULL && StreamHolder::Instance().current_stream()->ready()) {
        VLOG(1) << "- MKLDNNMemoryDescriptorBase<Dtype>::" << __FUNCTION__ << ": stream.wait() - " << this->name;
        StreamHolder::Instance().current_stream()->wait();
    }
    return true;
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
shared_ptr<primitive> MKLDNNMemoryDescriptor<Dtype, is_diff>::get_blob_prv_primitive(Blob<Dtype>* blob
                                            ,bool set_prv_ptr, bool convert
                                            ,MKLDNNMemoryDescriptor<Dtype,is_diff>* converted_in_fwd)
{
#ifdef DEBUG        
    LOG(INFO) << "GET_BLOB_PRV_PRIMITIVE";
#endif

    if (!this->conversion_needed()) {
        return shared_ptr<primitive>(); // TODO: may be CHECK ?
    }

    // Conversion is needed
    const Dtype* prv_ptr = is_diff ?  blob->prv_diff() : blob->prv_data();
    if (prv_ptr == NULL) {
        if (converted_in_fwd) {
            // TODO: use previously done conversion on forward - needed for training
            NOT_IMPLEMENTED;
        }
        if(convert) {
            this->convert_to_prv(const_cast<Dtype*>(is_diff ? blob->cpu_diff() : blob->cpu_data()));
        }
        else {
            this->create_reorder_to_prv(const_cast<Dtype*>(is_diff ? blob->cpu_diff() : blob->cpu_data()));
        }
        if (set_prv_ptr) {
            if (is_diff) {
                blob->set_prv_diff_descriptor(this->get_shared_ptr(), false);
                // below line designated to set correspondent SyncedMemory->_head to HEAD_AT_CPU
                // TODO: need to optimize
                blob->set_prv_diff_descriptor(NULL);
            } else {
                blob->set_prv_data_descriptor(this->get_shared_ptr(), false);
                // below line designated to set correspondent SyncedMemory->_head to HEAD_AT_CPU
                // TODO: need to optimize
                blob->set_prv_data_descriptor(NULL);
            }
        }
        return this->reorder_usr2prv();
    } else {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor<Dtype, is_diff>(blob);
        if ((*blob_prv_mkldnn_mem_descr->prv_memory_pd() !=  *this->prv_memory_pd() || blob_prv_mkldnn_mem_descr->get_scale() != this->get_scale()) && this->_reorder_extprv2prv_pd != NULL) {
            // prv in blob and in this descrptor may have different layouts
            if(convert) {
                LOG(INFO) << "BAD CONVERT";
                this->convert_from_extprv(blob_prv_mkldnn_mem_descr->aprimitive());
            }
            else {
                this->create_reorder_from_extprv(blob_prv_mkldnn_mem_descr->aprimitive());
            }
            return this->reorder_extprv2prv();
        } else if (blob_prv_mkldnn_mem_descr.get() != this) {
            VLOG(1) << "layout OK " << blob_prv_mkldnn_mem_descr->name << " == " << this->name;
        }
        return blob_prv_mkldnn_mem_descr->aprimitive();
    }
    NOT_IMPLEMENTED;
    return shared_ptr<mkldnn::primitive>();
}

// TODO: explain what is happenning here!!!
template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::sync_before_read()
{
#ifdef DEBUG        
    LOG(INFO) << "SYNC_BEFORE_READ";
#endif

    // TODO: need to optimize code
    if (!this->conversion_needed()) {
        return;
    }

    // Conversion is needed
    const Dtype* prv_ptr = is_diff ?  this->_blob->prv_diff() : this->_blob->prv_data();
    if (prv_ptr == NULL) {
        this->convert_to_prv(const_cast<Dtype*>(is_diff ? this->_blob->cpu_diff() : this->_blob->cpu_data()));
        // if blob has not prv descriptor then set it to avoid conversions on next iterations
        if (is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), false);
            // Original:
            // below line designated to set correspondent SyncedMemory->_head to HEAD_AT_CPU
            // TODO: need to optimize
            //this->_blob->set_prv_diff_descriptor(NULL);
            // It will lead the performance drop in two aspects:
            // 1. FWD Conv: Reorder of weights from oihw to OIhw16i16o is executed for every iteration. This should be happening only once per convolution layer including all iterations.
            // 2. BWD Conv: Reorder of weights is happening from oihw to OIhw16o16i format, where as expected, the reorder should happen from OIhw16i16o to OIhw16o16i for better performance.
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), true);     //Change from false to true, suggested by Czaja, Jacek
            // Original:
            // below line designated to set correspondent SyncedMemory->_head to HEAD_AT_CPU
            // TODO: need to optimize
            //this->_blob->set_prv_data_descriptor(NULL);
            // It will lead the performance drop in two aspects:
            // 1. FWD Conv: Reorder of weights from oihw to OIhw16i16o is executed for every iteration. This should be happening only once per convolution layer including all iterations.
            // 2. BWD Conv: Reorder of weights is happening from oihw to OIhw16o16i format, where as expected, the reorder should happen from OIhw16i16o to OIhw16o16i for better performance.
        }
    } else {
        shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > blob_prv_mkldnn_mem_descr = get_mkldnn_prv_descriptor<Dtype, is_diff>(this->_blob);

        if (*blob_prv_mkldnn_mem_descr->prv_memory_pd() !=  *this->prv_memory_pd() || blob_prv_mkldnn_mem_descr->get_scale() != this->get_scale()) {
            // prv in blob and in this descrptor may have different layouts
#ifdef D
        LOG(INFO) << "Convert from extprv";
#endif
            this->convert_from_extprv(blob_prv_mkldnn_mem_descr->aprimitive());
        } else {
            if (is_diff) {
                this->_blob->mutable_prv_diff();
            } else {
                this->_blob->mutable_prv_data();
            }
        }
    }
}

template <typename Dtype, bool is_diff>
void MKLDNNMemoryDescriptor<Dtype, is_diff>::sync_before_write(bool inplace)
{
    // TODO: need to optimize code
    if(!inplace) {
        if(is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), this->conversion_needed() ? false : true);
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), this->conversion_needed() ? false : true);
        }
    }
    //Fix me: this->conversion_needed() == false means diff/data is in the CPU, no need to set the prv_diff/data_descriptor
    /*
    if ((!inplace) && (this->conversion_needed())) {
        if (is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), false);
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), false);
        }
    }
    */
}

template <typename Dtype, bool is_diff>
shared_ptr<primitive> MKLDNNMemoryDescriptor<Dtype, is_diff>::create_input(Blob<Dtype> * blob, bool set_prv_ptr)
{
    shared_ptr<mkldnn::primitive> pres;
    if (this->conversion_needed()) {
        pres = this->get_blob_prv_primitive(blob, set_prv_ptr, false);
    } else {
        pres.reset(new memory(*this->usr_memory_pd(), const_cast<Dtype*>(is_diff ?  blob->cpu_diff() : blob->cpu_data())));
    }
    return pres;
}

template <typename Dtype, bool is_diff>
shared_ptr<memory> MKLDNNMemoryDescriptor<Dtype, is_diff>::create_output_memory(Blob<Dtype> * blob, bool inplace)
{
    shared_ptr<memory> omem;
    if (this->conversion_needed()) {
        shared_ptr<PrvMemDescr> blob_prv_mem_descriptor = is_diff ?
            (blob->get_prv_diff_descriptor()) : (blob->get_prv_data_descriptor());

        if(blob_prv_mem_descriptor != NULL) {
            shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > current_descr = get_mkldnn_prv_descriptor<Dtype, is_diff>(blob);

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

template <typename Dtype, bool is_diff>
shared_ptr<primitive> MKLDNNMemoryDescriptor<Dtype, is_diff>::create_input(bool set_prv_ptr)
{
    // TODO: need to iptimize code
    return create_input(this->_blob, set_prv_ptr);
}

template <typename Dtype, bool is_diff>
shared_ptr<memory> MKLDNNMemoryDescriptor<Dtype, is_diff>::create_output_memory(bool inplace)
{
    // TODO: need to optimize code
    shared_ptr<memory> omem = create_output_memory(this->_blob);
    if(!inplace) {
        if(is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), this->conversion_needed() ? false : true);
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), this->conversion_needed() ? false : true);
        }
    }
    /*
    //Fix me: this->conversion_needed() == false means diff/data is in the CPU, no need to set the prv_diff/data_descriptor
    if ((!inplace) && (this->conversion_needed())) {
        if (is_diff) {
            this->_blob->set_prv_diff_descriptor(this->get_shared_ptr(), false);
        } else {
            this->_blob->set_prv_data_descriptor(this->get_shared_ptr(), false);
        }
    }
    */
    return omem;
}

template <typename Dtype, bool is_diff>
Dtype* MKLDNNMemoryDescriptor<Dtype, is_diff>::get_memory_ptr(long offset) {
    if (this->conversion_needed()) {
      // TODO: support DFP16 offset
      if (this->prv_ptr() != NULL) return (Dtype*)this->prv_ptr() + offset;
      // when _internal_ptr is null, having same private layout as _blob
      else return is_diff ?
             (Dtype*)this->_blob->prv_diff() + offset :
             (Dtype*)this->_blob->prv_data() + offset;
    } else {
      return const_cast<Dtype*>(
        is_diff ? this->_blob->cpu_diff() + offset : this->_blob->cpu_data() + offset);
    }
}

template <typename Dtype, bool is_diff>
shared_ptr<memory::desc> MKLDNNMemoryDescriptor<Dtype, is_diff>::get_memory_desc() {
    shared_ptr<memory::desc> desc;
    if (this->conversion_needed()) {
        desc.reset(new memory::desc(this->prv_memory_pd()->desc()));
    } else {
        desc.reset(new memory::desc(this->usr_memory_pd()->desc()));
    }
    return desc;
}

template <typename Dtype, bool is_diff>
size_t MKLDNNMemoryDescriptor<Dtype, is_diff>::get_memory_count() {
  if (this->conversion_needed()) {
    return this->prv_count();
  } else {
    return this->_blob->count();
  }
}

template <typename Dtype, bool is_diff>
shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > get_mkldnn_prv_descriptor(Blob<Dtype>* blob)
{
    shared_ptr<PrvMemDescr> blob_prv_mem_descriptor = is_diff ?
            (blob->get_prv_diff_descriptor()) : (blob->get_prv_data_descriptor());

    CHECK_EQ(blob_prv_mem_descriptor->get_descr_type(), PrvMemDescr::PRV_DESCR_MKLDNN);

    shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > blob_prv_mkldnn_mem_descr =
            boost::static_pointer_cast<MKLDNNMemoryDescriptor<Dtype, is_diff> >(blob_prv_mem_descriptor);
    CHECK(blob_prv_mkldnn_mem_descr != NULL);
    return blob_prv_mkldnn_mem_descr;
}

template class MKLDNNMemoryDescriptor<double, true>;
template class MKLDNNMemoryDescriptor<float, true>;
template class MKLDNNMemoryDescriptor<float, false>;
template class MKLDNNMemoryDescriptor<double, false>;
template class MKLDNNMemoryDescriptorBase<float>;
template class MKLDNNMemoryDescriptorBase<double>;

template
shared_ptr<MKLDNNMemoryDescriptor<double, true> > get_mkldnn_prv_descriptor<double, true>(Blob<double>* blob);
template
shared_ptr<MKLDNNMemoryDescriptor<float, true> > get_mkldnn_prv_descriptor<float, true>(Blob<float>* blob);
template
shared_ptr<MKLDNNMemoryDescriptor<double, false> > get_mkldnn_prv_descriptor<double, false>(Blob<double>* blob);
template
shared_ptr<MKLDNNMemoryDescriptor<float, false> > get_mkldnn_prv_descriptor<float, false>(Blob<float>* blob);
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
