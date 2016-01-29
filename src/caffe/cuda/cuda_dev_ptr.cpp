#include "caffe/cuda/cuda_dev_ptr.hpp"

#ifdef USE_CUDA

namespace caffe {

template<typename Type>
cuda_dev_ptr<Type>::cuda_dev_ptr(Type* ptr)
    : raw_ptr_(ptr) {
}

template<typename Type>
void* cuda_dev_ptr<Type>::get() {
  return raw_ptr_;
}

template<typename Type>
int_tp cuda_dev_ptr<Type>::off() {
  return 0;
}

}  // namespace caffe

#endif  // USE_GREENTEA
