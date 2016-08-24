#ifndef CAFFE_MKLDNN_MEMORY_HPP_
#define CAFFE_MKLDNN_MEMORY_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "mkldnn.hpp"



using namespace mkldnn;

namespace caffe {

template <typename Dtype>
class MKLDNNMemoryDescriptorBase : PrvMemDescr
        , boost::enable_shared_from_this<MKLDNNMemoryDescriptorBase<Dtype> >
{
public:
    MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                ,shared_ptr<memory::primitive_desc> prv_memory_pd
                                ,shared_ptr<primitive> mkldnn_primitive = NULL);
    ~MKLDNNMemoryDescriptorBase() {}
    // ---- PrvMemDescr virtual functions -----
    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);
    virtual void convert_from_other(shared_ptr<PrvMemDescr> other);
    virtual bool layout_compare(shared_ptr<PrvMemDescr> other);
    virtual PrvDescrType get_descr_type() {return PRV_DESCR_MKLDNN;}
    virtual size_t prv_count();
    virtual size_t prv_size() { return prv_size() * sizeof(Dtype); }
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
    inline bool conversion_needed() const { return (_reorder_usr2prv_pd != NULL); }
    virtual void* prv_ptr() { CHECK(_prv_memory); return _internal_ptr;  }

    shared_ptr<memory>  get_prv_memory()
    {
        if (_prv_memory == NULL) allocate();
        return _prv_memory;
    }
    Dtype* get_prv_ptr() {
        if (_prv_memory == NULL) allocate();
        return _internal_ptr;
    }
    std::string name;  // for debugging purposes
private:
    void check_usr_with_prv_descriptors();
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
            this->create_reorders();
        }
    }
    void set_usr_memory_pd(shared_ptr<memory::primitive_desc> memory_pd) {
        _usr_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorders();
        }
    }
    void create_reorders();

    shared_ptr<memory::primitive_desc> _usr_memory_pd;
    shared_ptr<memory::primitive_desc> _prv_memory_pd;
    shared_ptr<reorder::primitive_desc> _reorder_usr2prv_pd;
    shared_ptr<reorder::primitive_desc> _reorder_prv2usr_pd;
    shared_ptr<memory> _prv_memory;
    Dtype* _internal_ptr;
    shared_ptr<primitive> _mkldnn_primitive;
};

template <typename Dtype, bool is_diff>
class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase<Dtype> {
public:
    MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , shared_ptr<primitive> mkldnn_primitive = NULL )
        : MKLDNNMemoryDescriptorBase<Dtype>(usr_memory_pd, prv_memory_pd, mkldnn_primitive ) {}
    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.
    Dtype* get_blob_data_ptr(Blob<Dtype> * blob, bool set_prv_ptr,
          MKLDNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd = NULL);
    void sync_blob_prv_data(Blob<Dtype> * blob);
    shared_ptr<primitive> create_input(Blob<Dtype> * blob);
    shared_ptr<memory> create_output_memory(Blob<Dtype> * blob);
};

template <typename Dtype>
class MKLDNNData : public MKLDNNMemoryDescriptor<Dtype, false>
{
public:
    MKLDNNData(shared_ptr<memory::primitive_desc> usr_memory_pd
                ,shared_ptr<memory::primitive_desc> prv_memory_pd
                , shared_ptr<primitive> mkldnn_primitive = NULL )
        : MKLDNNMemoryDescriptor<Dtype, false>(usr_memory_pd, prv_memory_pd, mkldnn_primitive ) {}
};

template <typename Dtype>
class MKLDNNDiff : public MKLDNNMemoryDescriptor<Dtype, true>
{
public:
    MKLDNNDiff(shared_ptr<memory::primitive_desc> usr_memory_pd
                , shared_ptr<memory::primitive_desc> prv_memory_pd
                , shared_ptr<primitive> mkldnn_primitive = NULL )
        : MKLDNNMemoryDescriptor<Dtype, true>(usr_memory_pd, prv_memory_pd, mkldnn_primitive ) {}
};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_MEMORY_HPP_
