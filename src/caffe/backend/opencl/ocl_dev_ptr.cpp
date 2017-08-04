#include "caffe/common.hpp"
#include "caffe/backend/opencl/ocl_dev_ptr.hpp"

#ifdef USE_OPENCL

namespace caffe {

template<typename Mtype>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(cl_mem ocl_mem)
    : ocl_mem_(ocl_mem), off_(0) {
}

template<typename Mtype>
template<typename Otype>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<Otype> *other)
    : ocl_mem_(other->get_ocl_mem()), off_(0) {
  ocl_dev_ptr<Otype>
  off_ = (other->get_off() * sizeof(Otype)) / sizeof(Mtype);
}

template<typename Mtype>
void ocl_dev_ptr<Mtype>::increment(int_tp val) {
  off_ += val;
}

template<typename Mtype>
void ocl_dev_ptr<Mtype>::decrement(int_tp val) {
  off_ -= val;
}

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<int8_t>* other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<uint8_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<int16_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<uint16_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<int32_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<uint32_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<int64_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<uint64_t> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<half> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<float> *other);

template<typename Mtype>
template<>
ocl_dev_ptr<Mtype>::ocl_dev_ptr(dev_ptr<double> *other);


template<typename Mtype>
cl_mem ocl_dev_ptr<Mtype>::get_ocl_mem() {
  return ocl_mem_;
}

template<typename Mtype>
uint_tp ocl_dev_ptr<Mtype>::get_off() {
  return off_;
}

INSTANTIATE_POINTER_CLASS(ocl_dev_ptr)


}  // namespace caffe

#endif  // USE_OPENCL
