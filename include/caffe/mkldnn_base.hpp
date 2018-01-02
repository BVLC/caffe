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

#ifndef CAFFE_MKLDNN_BASE_HPP_
#define CAFFE_MKLDNN_BASE_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "mkldnn.hpp"
#include "caffe/quant/base_quant_layer.hpp"

using namespace mkldnn;

namespace caffe {

// =====  CpuEngine =======================================
// cpu_engine singleton
class CpuEngine
{
public:
    static CpuEngine & Instance()
    {
        // I's thread-safe in C++11.
        static CpuEngine myInstance;
        return myInstance;
    }
    CpuEngine(CpuEngine const&) = delete;             // Copy construct
    CpuEngine(CpuEngine&&) = delete;                  // Move construct
    CpuEngine& operator=(CpuEngine const&) = delete;  // Copy assign
    CpuEngine& operator=(CpuEngine &&) = delete;      // Move assign

    engine & get_engine() { return _cpu_engine; }
protected:
    CpuEngine() : _cpu_engine(engine::cpu, 0) {}
//    CpuEngine() : _cpu_engine(engine::cpu_lazy, 0) {}
    ~CpuEngine() {}
private:
    engine _cpu_engine;
};

#ifdef FPGA_ENABLED
// =====  FPGAEngine =======================================
// fpga_engine singleton
class FPGAEngine
{
public:
    static FPGAEngine & Instance()
    {
        // I's thread-safe in C++11.
        static FPGAEngine myInstance;
        return myInstance;
    }
    FPGAEngine(FPGAEngine const&) = delete;             // Copy construct
    FPGAEngine(FPGAEngine&&) = delete;                  // Move construct
    FPGAEngine& operator=(FPGAEngine const&) = delete;  // Copy assign
    FPGAEngine& operator=(FPGAEngine &&) = delete;      // Move assign

    engine & get_engine() { return _fpga_engine; }
protected:
    FPGAEngine() : _fpga_engine(engine::fpga, 0) {}
    ~FPGAEngine() {}
private:
    engine _fpga_engine;
};
#endif // #ifdef FPGA_ENABLED

#ifdef DLA_ENABLED
// =====  Deep Learning Accelerator =======================================
class DLAEngine
{
public:
    static DLAEngine & Instance()
    {
        // I's thread-safe in C++11.
        static DLAEngine myInstance;
        return myInstance;
    }
    DLAEngine(DLAEngine const&) = delete;             // Copy construct
    DLAEngine(DLAEngine&&) = delete;                  // Move construct
    DLAEngine& operator=(DLAEngine const&) = delete;  // Copy assign
    DLAEngine& operator=(DLAEngine &&) = delete;      // Move assign

    engine & get_engine() { return _dla_engine; }
protected:
    DLAEngine() : _dla_engine(engine::dla, 0) {}
    ~DLAEngine() {}
private:
    engine _dla_engine;
};


#endif // #ifdef DLA_ENABLED

// =====  MKLDNNStream =======================================
class MKLDNNStream {
public:
    explicit MKLDNNStream():_ready(false) { prepare(); }
    virtual ~MKLDNNStream() {}
    MKLDNNStream  &submit(std::vector<primitive> primitives) { _stream->submit(primitives); return *this; }
    bool wait(bool block = true) {
        VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : wait stream ";
        _ready = false;
        bool res = _stream->wait(block);
        VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : end of stream waiting ";
        return res;
    }
    bool ready() { return _ready; }
    void prepare() {
        if(_ready == false) {
            // stream just created or already executed
            // !! TODO: change below if stream will have method to reset its state
            VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : create new stream";
//            _stream.reset(new stream(stream::kind::any));
            _stream.reset(new stream(stream::kind::eager));
            // TODO: Enable when Unit tests work for this one
            //_stream.reset(new stream(stream::kind::lazy));
        }
        _ready = true;
    }
protected:
private:
    bool _ready;
    shared_ptr<stream> _stream;
};

// =====  StreamHolder =======================================
// singleton
class StreamHolder
{
public:
    static StreamHolder & Instance()
    {
        // I's thread-safe in C++11.
        static StreamHolder myInstance;
        return myInstance;
    }
    StreamHolder(StreamHolder const&) = delete;             // Copy construct
    StreamHolder(StreamHolder&&) = delete;                  // Move construct
    StreamHolder& operator=(StreamHolder const&) = delete;  // Copy assign
    StreamHolder& operator=(StreamHolder &&) = delete;      // Move assign

    shared_ptr<MKLDNNStream> get_stream();
    shared_ptr<MKLDNNStream> current_stream() { return _current_stream; }
    void prepare_mkldnn_stream(shared_ptr<MKLDNNStream> mkldnn_stream) {
        _current_stream = mkldnn_stream;
        _current_stream->prepare();
    }
protected:
    StreamHolder() : _current_stream() {}
    ~StreamHolder() {}
private:
    shared_ptr<MKLDNNStream> _current_stream;
};

// =====  MKLDNNLayer =======================================
template <typename Dtype>
class MKLDNNLayer : public BaseQuantLayer<Dtype> {
public:
    explicit MKLDNNLayer(const LayerParameter &param);
    virtual ~MKLDNNLayer() {}
protected:
    bool reshape;
};

// =====  MKLDNNPrimitive =======================================
template <typename Dtype>
class MKLDNNPrimitive {
public:
    explicit MKLDNNPrimitive():aprimitive(), mkldnn_stream() {}

    //API for initializing with shared_ptr<primitive>
    MKLDNNPrimitive(shared_ptr<primitive> aprimitive_input) {this->aprimitive = aprimitive_input;}

    virtual ~MKLDNNPrimitive() {}
    void reset(primitive* pprimitive) { this->aprimitive.reset(pprimitive);}
    shared_ptr<primitive> aprimitive;
    shared_ptr<MKLDNNStream> mkldnn_stream;
    shared_ptr<MKLDNNStream> get_mkldnn_stream();
    shared_ptr<MKLDNNStream> submit();
private:
};

}  // namespace caffe
#endif  // #ifndef CAFFE_MKLDNN_BASE_HPP_
