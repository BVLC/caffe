#include "caffe/opencl/ocl_dev_ptr.hpp"

#ifdef USE_GREENTEA

namespace caffe {

template<typename Type>
ocl_dev_ptr<Type>::ocl_dev_ptr(cl_mem ocl_mem)
    : ocl_mem_(ocl_mem) {
}

template<typename Type>
Type* ocl_dev_ptr<Type>::get() {
  return nullptr;
}

template<typename Type>
std::size_t ocl_dev_ptr<Type>::off() {
  return 0;
}

}  // namespace caffe

#endif  // USE_GREENTEA
