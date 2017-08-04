#include "caffe/common.hpp"
#include "caffe/backend/cuda/cuda_dev_ptr.hpp"

#ifdef USE_CUDA

namespace caffe {

template<typename Mtype>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(Mtype* raw_ptr)
    : raw_ptr_(raw_ptr) {
}

template<typename Mtype>
template<typename Otype>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<Otype> > other)
    : raw_ptr_(nullptr) {
  raw_ptr_ = reinterpret_cast<Mtype*>(other.get()->get());
}

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<int8_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<uint8_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<int16_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<uint16_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<int32_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<uint32_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<int64_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<uint64_t> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<half_float::half> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<float> > other);

template<typename Mtype>
template<>
cuda_dev_ptr<Mtype>::cuda_dev_ptr(std::shared_ptr<cuda_dev_ptr<double> > other);


template<typename Mtype>
void cuda_dev_ptr<Mtype>::increment(int_tp val) {
  raw_ptr_ += val;
}

template<typename Mtype>
void cuda_dev_ptr<Mtype>::decrement(int_tp val) {
  raw_ptr_ -= val;
}

template<typename Mtype>
Mtype* cuda_dev_ptr<Mtype>::get() {
  return raw_ptr_;
}


INSTANTIATE_POINTER_CLASS(cuda_dev_ptr)


}  // namespace caffe

#endif  // USE_CUDA
