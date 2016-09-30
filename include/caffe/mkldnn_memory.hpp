#ifndef CAFFE_MKLDNN_MEMORY_HPP_
#define CAFFE_MKLDNN_MEMORY_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "mkldnn.hpp"
#include "mkldnn_base.hpp"

using namespace mkldnn;

namespace caffe {

// =====  MKLDNNMemoryDescriptorBase =======================================
template <typename Dtype>
class MKLDNNMemoryDescriptorBase : public PrvMemDescr
        , public boost::enable_shared_from_this<MKLDNNMemoryDescriptorBase<Dtype> >
{
public:
    MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                , shared_ptr<memory::primitive_desc> prv_memory_pd
                                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer);

    ~MKLDNNMemoryDescriptorBase() {}
    // ---- PrvMemDescr virtual functions -----
    virtual void convert_from_other(shared_ptr<PrvMemDescr> other);
    virtual bool layout_compare(shared_ptr<PrvMemDescr> other);
    virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKLDNN;}

    // TODO: assuming size/sizeof = count may be not correct
    virtual size_t prv_count() { return prv_size()/sizeof(Dtype); }

    virtual size_t prv_size() { return _prv_memory_pd->get_size(); }
    // ---------------------------------------
    shared_ptr<MKLDNNMemoryDescriptorBase<Dtype> > get_shared_ptr() {
        return this->shared_from_this();
    }
    shared_ptr<memory::primitive_desc>  prv_memory_pd() const {
        return _prv_memory_pd;
    }
    shared_ptr<memory::primitive_desc>  usr_memory_pd() const {
        return _usr_memory_pd;
    }
    inline bool conversion_needed() const { return (_reorder_usr2prv_pd != NULL || _reorder_extprv2prv_pd != NULL); }
    virtual void* prv_ptr() { return _internal_ptr;  }

    shared_ptr<memory>  get_prv_memory()
    {
        if (_prv_memory == NULL) allocate();
        return _prv_memory;
    }
    Dtype* get_prv_ptr() {
        if (_prv_memory == NULL) allocate();
        return _internal_ptr;
    }
    shared_ptr<primitive>  reorder_usr2prv() { return _reorder_usr2prv.aprimitive; }
    shared_ptr<primitive>  reorder_prv2usr() { return _reorder_prv2usr.aprimitive; }
    shared_ptr<primitive>  reorder_extprv2prv() { return _reorder_extprv2prv.aprimitive; }

    void set_mkldnn_layer(MKLDNNLayer<Dtype>* layer) { _mkldnn_layer = layer;  }
    MKLDNNLayer<Dtype>*  mkldnn_layer() const { return _mkldnn_layer;  }

    std::string name;  // for debugging purposes
protected:
    void check_usr_with_prv_descriptors();
    void set_prv_memory(shared_ptr<memory> memory)
    {
        _prv_memory = memory;
        _internal_ptr = (Dtype *)(_prv_memory->get_data_handle());
    }

    void allocate() {
        if (_prv_memory == NULL) {
            _prv_memory = shared_ptr<memory>(new memory(*_prv_memory_pd));
            _internal_ptr = (Dtype *)(_prv_memory->get_data_handle());
            // TODO: may need initialize memory by 0
        }
    }
    void set_prv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd)  {
        _prv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors();
        }
    }
    void set_extprv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd)  {
        _extprv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors();
        }
    }
    void set_usr_memory_pd(shared_ptr<memory::primitive_desc> memory_pd) {
        _usr_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors();
        }
    }
    void create_reorder_descriptors();

    shared_ptr<memory::primitive_desc> _usr_memory_pd;
    shared_ptr<memory::primitive_desc> _prv_memory_pd;
    shared_ptr<memory::primitive_desc> _extprv_memory_pd;
    shared_ptr<reorder::primitive_desc> _reorder_usr2prv_pd;
    shared_ptr<reorder::primitive_desc> _reorder_prv2usr_pd;
    shared_ptr<reorder::primitive_desc> _reorder_extprv2prv_pd;
    MKLDNNPrimitive<Dtype> _reorder_usr2prv;
    MKLDNNPrimitive<Dtype> _reorder_prv2usr;
    MKLDNNPrimitive<Dtype> _reorder_extprv2prv;
    shared_ptr<memory> _prv_memory;
    Dtype* _internal_ptr;
    shared_ptr<memory> _usr_memory;
    shared_ptr<memory> _extprv_memory;
    void* _cpu_ptr; // TODO: ?? 
    void* _extprv_ptr;

    MKLDNNLayer<Dtype>* _mkldnn_layer;
    Blob<Dtype>* _blob;
};

template <typename Dtype, bool is_diff>
class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase<Dtype> {
public:
    MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer);

    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);
    virtual void convert_from_extprv(void* cpu_ptr);
    virtual bool on_to_cpu();

    virtual void create_reorder_from_prv(void* cpu_ptr);
    virtual void create_reorder_to_prv(void* cpu_ptr);
    virtual void create_reorder_from_extprv(void* cpu_ptr);

    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.
    shared_ptr<primitive> get_blob_prv_primitive(Blob<Dtype> * blob, bool set_prv_ptr, bool convert = true,
            MKLDNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd = NULL);
    void sync_blob_prv_data(Blob<Dtype> * blob, bool set_prv_ptr);

    void sync_before_read(bool set_prv_ptr);
    void sync_before_write();

    shared_ptr<primitive> create_input(Blob<Dtype> * blob, bool set_prv_ptr);
    shared_ptr<memory> create_output_memory(Blob<Dtype> * blob);
    shared_ptr<primitive> create_input(bool set_prv_ptr);
    shared_ptr<memory> create_output_memory();

    void set_mkldnn_primitive(MKLDNNPrimitive<Dtype>& mprimitive) { CHECK(mprimitive.aprimitive); _mkldnn_primitive = mprimitive;  }
    MKLDNNPrimitive<Dtype>&  mkldnn_primitive() { return _mkldnn_primitive; }
    shared_ptr<primitive> aprimitive() const { return _mkldnn_primitive.aprimitive; }
private:
    MKLDNNPrimitive<Dtype> _mkldnn_primitive;
};

template <typename Dtype>
class MKLDNNData : public MKLDNNMemoryDescriptor<Dtype, false>
{
public:
    MKLDNNData(shared_ptr<memory::primitive_desc> usr_memory_pd
                , shared_ptr<memory::primitive_desc> prv_memory_pd
                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer)
        : MKLDNNMemoryDescriptor<Dtype, false>(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer) {}
};

template <typename Dtype>
class MKLDNNDiff : public MKLDNNMemoryDescriptor<Dtype, true>
{
public:
    MKLDNNDiff(shared_ptr<memory::primitive_desc> usr_memory_pd
                , shared_ptr<memory::primitive_desc> prv_memory_pd
                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer)
        : MKLDNNMemoryDescriptor<Dtype, true>(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer ) {}
};

template <typename Dtype, bool is_diff>
shared_ptr<MKLDNNMemoryDescriptor<Dtype, is_diff> > get_mkldnn_prv_descriptor(Blob<Dtype>* blob);

}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_MEMORY_HPP_
