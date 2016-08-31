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


// =====  MKLDNNStream =======================================
class MKLDNNStream {
public:
    explicit MKLDNNStream():_ready(false) { prepare(); }
    virtual ~MKLDNNStream() {}
    MKLDNNStream  &submit(std::vector<primitive> primitives) { _stream->submit(primitives); return *this; }
    bool wait(bool block = true) {
        _ready = false;
        return _stream->wait(block);
    }
    bool ready() { return _ready; }
    void prepare() {
        if(_ready == false) {
            // stream just created or already executed
            // !! TODO: change below if stream will have method to reset its state
            _stream.reset(new stream());
        }
        _ready = true;
    }
protected:
private:
    bool _ready;
    shared_ptr<stream> _stream;
};

// =====  MKLDNNLayer =======================================
template <typename Dtype>
class MKLDNNLayer {
public:
    explicit MKLDNNLayer():_mkldnn_stream(NULL), _previous_mkldnn_layer(NULL) {}
    virtual ~MKLDNNLayer() {}
    shared_ptr<MKLDNNStream> mkldnn_stream() { return _mkldnn_stream; }
    shared_ptr<MKLDNNStream> get_mkldnn_stream() {
        if(_mkldnn_stream == NULL || !_mkldnn_stream->ready()) {
            _mkldnn_stream.reset(new MKLDNNStream());
        }
        return _mkldnn_stream;
    }
    void set_mkldnn_stream(shared_ptr<MKLDNNStream> mkldnn_stream) { _mkldnn_stream = mkldnn_stream; }
    MKLDNNLayer<Dtype>* get_mkldnn_layer(Blob<Dtype>* blob);
    void init_mkldnn_stream();
protected:
    MKLDNNLayer<Dtype>* _previous_mkldnn_layer;
private:
    shared_ptr<MKLDNNStream> _mkldnn_stream;
};



template <typename Dtype>
class MKLDNNMemoryDescriptorBase : public PrvMemDescr
        , public boost::enable_shared_from_this<MKLDNNMemoryDescriptorBase<Dtype> >
{
public:
    MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                ,shared_ptr<memory::primitive_desc> prv_memory_pd);
    ~MKLDNNMemoryDescriptorBase() {}
    // ---- PrvMemDescr virtual functions -----
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
    shared_ptr<reorder>  reorder_usr2prv() { return _reorder_usr2prv; }
    shared_ptr<reorder>  reorder_prv2usr() { return _reorder_prv2usr; }
    std::string name;  // for debugging purposes
    
    void set_mkldnn_layer(MKLDNNLayer<Dtype>* layer) { _mkldnn_layer = layer;  }
    MKLDNNLayer<Dtype>*  mkldnn_layer() const { return _mkldnn_layer;  }
    void set_mkldnn_stream(shared_ptr<MKLDNNStream> mkldnn_stream) { _mkldnn_stream = mkldnn_stream; }

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
    shared_ptr<reorder> _reorder_usr2prv;
    shared_ptr<reorder> _reorder_prv2usr;
    shared_ptr<memory> _prv_memory;
    Dtype* _internal_ptr;
    shared_ptr<memory> _usr_memory;
    void* _cpu_ptr;

    shared_ptr<MKLDNNStream> _mkldnn_stream;

    MKLDNNLayer<Dtype>* _mkldnn_layer;
};

template <typename Dtype, bool is_diff>
class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase<Dtype> {
public:
    MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd )
        : MKLDNNMemoryDescriptorBase<Dtype>(usr_memory_pd, prv_memory_pd ) {}

    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);

    virtual void create_reorder_from_prv(void* cpu_ptr);
    virtual void create_reorder_to_prv(void* cpu_ptr);

    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.
    shared_ptr<primitive> get_blob_prv_primitive(Blob<Dtype> * blob, bool set_prv_ptr, bool convert = true,
            MKLDNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd = NULL);
    void sync_blob_prv_data(Blob<Dtype> * blob, bool set_prv_ptr);
    shared_ptr<primitive> create_input(Blob<Dtype> * blob, bool set_prv_ptr);
    shared_ptr<memory> create_output_memory(Blob<Dtype> * blob);

    void set_mkldnn_primitive(shared_ptr<primitive> primitive) { _mkldnn_primitive = primitive;  }
    void set_linked_primitive(shared_ptr<primitive> primitive) { _linked_primitive = primitive;  }
    void set_primitives(shared_ptr<primitive> primitive, Blob<Dtype> * blob);
    shared_ptr<primitive>  mkldnn_primitive() const { return _mkldnn_primitive;  }
    shared_ptr<primitive>  linked_primitive() const { return _linked_primitive;  }
    
private:
    shared_ptr<primitive> _mkldnn_primitive;
    shared_ptr<primitive> _linked_primitive;
    
};

template <typename Dtype>
class MKLDNNData : public MKLDNNMemoryDescriptor<Dtype, false>
{
public:
    MKLDNNData(shared_ptr<memory::primitive_desc> usr_memory_pd
                ,shared_ptr<memory::primitive_desc> prv_memory_pd )
        : MKLDNNMemoryDescriptor<Dtype, false>(usr_memory_pd, prv_memory_pd ) {}
};

template <typename Dtype>
class MKLDNNDiff : public MKLDNNMemoryDescriptor<Dtype, true>
{
public:
    MKLDNNDiff(shared_ptr<memory::primitive_desc> usr_memory_pd
                , shared_ptr<memory::primitive_desc> prv_memory_pd )
        : MKLDNNMemoryDescriptor<Dtype, true>(usr_memory_pd, prv_memory_pd ) {}
};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_MEMORY_HPP_
