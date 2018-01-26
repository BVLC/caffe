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
                                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                                , std::vector<int>fl=std::vector<int>(1,0)
                                , bool is_sum=false);

    MKLDNNMemoryDescriptorBase(shared_ptr<memory::primitive_desc> usr_memory_pd
                                , shared_ptr<memory::primitive_desc> prv_memory_pd
                                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                                , bool is_float
                                , std::vector<float>scale=std::vector<float>(1,1.)
                                , bool is_sum=false); 


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

    float get_scale(int i) { return _scale[i]; }
    std::vector<float> get_scale() { return _scale; }
    void set_scale(std::vector<float> scale) { _scale.assign(scale.begin(),scale.end());}

    int get_fl(int i) { return _fl[i]; }
    std::vector<int> get_fl() { return _fl; }
    void set_fl(std::vector<int> fl) { _fl.assign(fl.begin(),fl.end());}

    void set_sum(bool is_sum) { _is_sum = is_sum; }
    bool get_sum() { return _is_sum; }

    void set_float(bool is_float) { _is_float = is_float; }
    bool get_float() { return _is_float; }

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
#ifdef USE_MLSL
          if (mn::is_multinode()) {
            auto mlsl_free = [](char* p) { mn::free((void*)p); };
            _mlsl_memory.reset(
              (char*)mn::alloc(_prv_memory_pd->get_size(), 64), mlsl_free);
            _prv_memory = shared_ptr<memory>(
              new memory(*_prv_memory_pd, (void*)_mlsl_memory.get()));
          } else {
#endif
            _prv_memory = shared_ptr<memory>(new memory(*_prv_memory_pd));
#ifdef USE_MLSL
          }
#endif
          _internal_ptr = (Dtype *)(_prv_memory->get_data_handle());
          // TODO: may need initialize memory by 0
        }
    }
    void set_prv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<float> scale)  {
        _prv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors(scale);
        }
    }

    void set_prv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<int> fl)  {
        _prv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors(fl);
        }
    }

    void set_extprv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<int> fl, std::vector<int> fl_ext, bool is_sum)  {
        _extprv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors(fl, fl_ext, is_sum);
        }
    }

    void set_extprv_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<float> scale, std::vector<float> scale_ext, bool is_sum)  {
        _extprv_memory_pd = memory_pd;
        if (_prv_memory_pd && _usr_memory_pd) {
            check_usr_with_prv_descriptors();
            this->create_reorder_descriptors(scale, scale_ext, is_sum);
        }
    }

    void set_usr_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<float> scale) {
        _usr_memory_pd = memory_pd;
    }

    void set_usr_memory_pd(shared_ptr<memory::primitive_desc> memory_pd, std::vector<int> fl) {
        _usr_memory_pd = memory_pd;
    }

    void create_reorder_descriptors(std::vector<float> scale, std::vector<float>scale_ext=std::vector<float>(1,1.), bool is_sum=false);

    void create_reorder_descriptors(std::vector<int> fl, std::vector<int>fl_ext=std::vector<int>(1,0), bool is_sum=false);

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
    void* _cpu_ptr;

#ifdef CO_SIM
    shared_ptr<memory> _usr_memory_cosim;
    MKLDNNPrimitive<Dtype> _reorder_prv2usr_cosim;
    //Used to save the copy of the private memory
    shared_ptr<memory> _prv_memory_cosim;
    //Wrapper for readonly_prv_memory
    shared_ptr<primitive::at>  at_prv_cosim;
#endif
    MKLDNNLayer<Dtype>* _mkldnn_layer;
    Blob<Dtype>* _blob;
    std::vector<float> _scale = std::vector<float>(1,1.);
    std::vector<int> _fl = std::vector<int>(1,0);
    bool _is_sum = false;
    bool _is_float = false;
#ifdef USE_MLSL
    shared_ptr<char> _mlsl_memory;
#endif
};

template <typename Dtype, bool is_diff>
class MKLDNNMemoryDescriptor : public MKLDNNMemoryDescriptorBase<Dtype> {
public:
    MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                        , std::vector<int> fl=std::vector<int>(1,0)
                        , bool is_sum=false);

    MKLDNNMemoryDescriptor(shared_ptr<memory::primitive_desc> usr_memory_pd
                        , shared_ptr<memory::primitive_desc> prv_memory_pd
                        , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                        , bool is_float
                        , std::vector<float> scale=std::vector<float>(1,1.)
                        , bool is_sum=false);

    virtual void convert_from_prv(void* cpu_ptr);
    virtual void convert_to_prv(void* cpu_ptr);
    virtual void convert_from_extprv(shared_ptr<primitive> aprimitive);
    virtual bool on_to_cpu();
#ifdef CO_SIM

    virtual void create_reorder_from_prv_cosim(void* cpu_ptr);
    virtual void convert_from_prv_cosim(void* cpu_ptr);
#endif

    virtual void create_reorder_from_prv(void* cpu_ptr);
    virtual void create_reorder_to_prv(void* cpu_ptr);
    virtual void create_reorder_from_extprv(shared_ptr<primitive> aprimitive);

    // The last get_blob_data_ptr() argument is a hack for reusing
    // in backward a conversion done already in the forward direction.
    shared_ptr<primitive> get_blob_prv_primitive(Blob<Dtype> * blob, bool set_prv_ptr, bool convert = true,
            MKLDNNMemoryDescriptor<Dtype, is_diff>* converted_in_fwd = NULL);

    void sync_before_read();
    void sync_before_write(bool inplace = false);

    shared_ptr<primitive> create_input(Blob<Dtype> * blob, bool set_prv_ptr);
    shared_ptr<memory> create_output_memory(Blob<Dtype> * blob, bool inplace = false);
    shared_ptr<primitive> create_input(bool set_prv_ptr);
    shared_ptr<memory> create_output_memory(bool inplace = false);
    Dtype* get_memory_ptr(long offset = 0);
    shared_ptr<memory::desc> get_memory_desc();
    size_t get_memory_count();
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
                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                , std::vector<int> fl=std::vector<int>(1,0)
                , bool is_sum=false)
        : MKLDNNMemoryDescriptor<Dtype, false>(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer, fl, is_sum) {}

    MKLDNNData(shared_ptr<memory::primitive_desc> usr_memory_pd
                , shared_ptr<memory::primitive_desc> prv_memory_pd
                , Blob<Dtype>* blob, MKLDNNLayer<Dtype>* mkldnn_layer
                , bool is_float
                , std::vector<float> scale=std::vector<float>(1,1.)
                , bool is_sum=false)
        : MKLDNNMemoryDescriptor<Dtype, false>(usr_memory_pd, prv_memory_pd, blob, mkldnn_layer, is_float, scale, is_sum) {}
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
