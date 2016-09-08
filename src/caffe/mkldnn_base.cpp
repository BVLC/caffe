#ifdef MKLDNN_SUPPORTED
#include "caffe/mkldnn_memory.hpp"

namespace caffe {


shared_ptr<MKLDNNStream> StreamHolder::get_stream()
{
    if (this->_current_stream == NULL || !this->_current_stream->ready()) {
        _current_stream.reset(new MKLDNNStream());
    }
    return _current_stream;
}

template <typename Dtype>
shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::get_mkldnn_stream() {
    if(mkldnn_stream == NULL)
        mkldnn_stream = StreamHolder::Instance().get_stream();
    else
        StreamHolder::Instance().prepare_mkldnn_stream(mkldnn_stream);
    return mkldnn_stream;

}

template <typename Dtype>
shared_ptr<MKLDNNStream>  MKLDNNPrimitive<Dtype>::submit() {
    CHECK(this->aprimitive);
    this->get_mkldnn_stream()->submit({*(this->aprimitive)});
    return mkldnn_stream;
}

template class MKLDNNLayer<double>;
template class MKLDNNLayer<float>;
template class MKLDNNPrimitive<double>;
template class MKLDNNPrimitive<float>;
}  // namespace caffe
#endif  // #ifdef MKLDNN_SUPPORTED
