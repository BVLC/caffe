#ifndef CAFFE_MKLDNN_BASE_HPP_
#define CAFFE_MKLDNN_BASE_HPP_

#include <string>
#include <vector>

#include "boost/enable_shared_from_this.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "mkldnn.hpp"

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
//    CpuEngine() : _cpu_engine(engine::cpu, 0) {}
    CpuEngine() : _cpu_engine(engine::cpu_lazy, 0) {}
    ~CpuEngine() {}
private:
    engine _cpu_engine;
};

// =====  MKLDNNStream =======================================
class MKLDNNStream {
public:
    explicit MKLDNNStream():_ready(false) { prepare(); }
    virtual ~MKLDNNStream() {}
    MKLDNNStream  &submit(std::vector<primitive> primitives) { _stream->submit(primitives); return *this; }
    bool wait(bool block = true) {
        VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : execute stream (wait) ";
        _ready = false;
        return _stream->wait(block);
    }
    bool ready() { return _ready; }
    void prepare() {
        if(_ready == false) {
            // stream just created or already executed
            // !! TODO: change below if stream will have method to reset its state
            VLOG(1) << typeid(*this).name()<< " : " << __FUNCTION__ << " : create new stream";
            _stream.reset(new stream());
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
    StreamHolder() : _current_stream(NULL) {}
    ~StreamHolder() {}
private:
    shared_ptr<MKLDNNStream> _current_stream;
};

// =====  MKLDNNLayer =======================================
template <typename Dtype>
class MKLDNNLayer {
public:
    explicit MKLDNNLayer() {}
    virtual ~MKLDNNLayer() {}
};

// =====  MKLDNNPrimitive =======================================
template <typename Dtype>
class MKLDNNPrimitive {
public:
    explicit MKLDNNPrimitive():aprimitive(NULL), mkldnn_stream(NULL) {}
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
